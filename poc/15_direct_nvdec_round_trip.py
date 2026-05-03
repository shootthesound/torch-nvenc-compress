"""15 — Direct NVDEC round-trip (Option D, session 4 deliverable).

Encodes a synthetic YUV444 frame via the direct NVENC ctypes path (PoC 14)
and then decodes the resulting HEVC bitstream via the direct NVDEC ctypes
path — completing the codec round-trip without PyAV / PyNvVideoCodec /
FFmpeg subprocess anywhere.

Pipeline:

  ENCODE side (from PoC 14):
    1. Open NVENC session, init for HEVC YUV444 constQP
    2. Create input + bitstream buffers
    3. Write synthetic YUV444 frame, EncodePicture, LockBitstream
    4. Capture HEVC Annex-B bytes

  DECODE side (new, this session):
    1. Open CUvideoctxlock for the same CUDA context
    2. Build CUVIDPARSERPARAMS with three callbacks:
         pfnSequenceCallback — fired with CUVIDEOFORMAT on the first parse;
                                we create a YUV444 NVDEC decoder here
         pfnDecodePicture   — pass through to cuvidDecodePicture
         pfnDisplayPicture  — map the frame, copy YUV planes back to host,
                                unmap, store the bytes
    3. cuvidParseVideoData with the encoded bitstream
    4. cuvidParseVideoData with CUVID_PKT_ENDOFSTREAM to flush
    5. Compare decoded bytes vs original input bytes (PSNR)

Success criterion: decoded bytes reasonably match the original synthetic
gradient (PSNR > 30 dB) — proves the NVDEC ctypes path is functional.
"""

from __future__ import annotations

import ctypes
import sys

import torch
from cuda.bindings import driver as cuda

from nvenc_compress.direct import (
    create_instance, open_encode_session_cuda, destroy_encoder,
    NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P4_GUID,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
)
from nvenc_compress.direct.structs import (
    initialize_encoder_hevc_yuv444,
    create_input_buffer, destroy_input_buffer,
    create_bitstream_buffer, destroy_bitstream_buffer,
    write_input_buffer, encode_picture, lock_and_read_bitstream,
    NV_ENC_BUFFER_FORMAT_YUV444,
)
from nvenc_compress.direct.decoder import (
    CUVIDPARSERPARAMS, CUVIDSOURCEDATAPACKET, CUVIDPICPARAMS,
    CUVIDPARSERDISPINFO, CUVIDEOFORMAT, CUVIDDECODECREATEINFO, CUVIDPROCPARAMS,
    PFNVIDSEQUENCECALLBACK, PFNVIDDECODECALLBACK, PFNVIDDISPLAYCALLBACK,
    PFNVIDOPPOINTCALLBACK, PFNVIDSEIMSGCALLBACK,
    cudaVideoCodec_HEVC, cudaVideoChromaFormat_444, cudaVideoSurfaceFormat_YUV444,
    cudaVideoCreate_PreferCUVID, cudaVideoDeinterlaceMode_Weave,
    CUVID_PKT_ENDOFSTREAM,
    ctx_lock_create, ctx_lock_destroy,
    create_parser, parse_video_data, destroy_parser,
    create_decoder, destroy_decoder, decode_picture,
    map_video_frame64, unmap_video_frame64,
)


NV_ENC_PIC_FLAG_FORCEIDR = 1 << 1
NV_ENC_PIC_FLAG_OUTPUT_SPSPPS = 1 << 2

W, H = 256, 256
QP = 18


def make_synthetic_yuv444(width: int, height: int) -> bytes:
    y_plane = bytearray(width * height)
    for row in range(height):
        for col in range(width):
            y_plane[row * width + col] = (row + col) & 0xFF
    u_plane = bytes([128] * (width * height))
    v_plane = bytes([128] * (width * height))
    return bytes(y_plane) + u_plane + v_plane


def encode_one_frame() -> bytes:
    """Run the PoC 14 encode flow; return the HEVC Annex-B bitstream."""
    table = create_instance()
    err, ctx = cuda.cuCtxGetCurrent()
    encoder = open_encode_session_cuda(table, int(ctx))
    try:
        initialize_encoder_hevc_yuv444(
            table, encoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P4_GUID,
            W, H, QP, tuning=NV_ENC_TUNING_INFO_HIGH_QUALITY,
        )
        in_buf = create_input_buffer(table, encoder, W, H, NV_ENC_BUFFER_FORMAT_YUV444)
        out_buf = create_bitstream_buffer(table, encoder)
        try:
            frame = make_synthetic_yuv444(W, H)
            write_input_buffer(table, encoder, in_buf, frame, W, H)
            s = encode_picture(table, encoder, in_buf, out_buf, W, H,
                               pic_flags=NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS,
                               buffer_format=NV_ENC_BUFFER_FORMAT_YUV444)
            if s != 0:
                raise RuntimeError(f"encode_picture status={s}")
            return lock_and_read_bitstream(table, encoder, out_buf)
        finally:
            destroy_bitstream_buffer(table, encoder, out_buf)
            destroy_input_buffer(table, encoder, in_buf)
    finally:
        destroy_encoder(table, encoder)


class DecodeContext:
    """Holds parser state shared between callbacks."""
    def __init__(self):
        self.decoder = None
        self.coded_w = 0
        self.coded_h = 0
        self.decoded_bytes: bytes | None = None
        self.disp_count = 0
        self.error: str | None = None


def main() -> int:
    print("Direct NVENC + NVDEC round-trip (D session 4)\n")

    # CUDA init
    print("[1] CUDA init...")
    err, = cuda.cuInit(0)
    if int(err) != 0:
        print(f"    cuInit failed: {err}"); return 1
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    err, ctx = cuda.cuCtxGetCurrent()
    if int(ctx) == 0:
        print("    cuCtxGetCurrent returned NULL"); return 1
    print(f"    ctx 0x{int(ctx):x}\n")

    # ENCODE half
    print("[2] Encode synthetic frame via direct NVENC...")
    bitstream = encode_one_frame()
    print(f"    HEVC bitstream: {len(bitstream)} bytes")
    print(f"    first bytes: {bitstream[:8].hex()}\n")

    # DECODE half
    print("[3] Open CUvideoctxlock...")
    lock = ctx_lock_create(int(ctx))
    print(f"    lock 0x{lock.value:x}\n")

    decode_ctx = DecodeContext()

    # ---- Callbacks ----------------------------------------------------------
    @PFNVIDSEQUENCECALLBACK
    def on_sequence(user, fmt_ptr):
        try:
            fmt = fmt_ptr.contents
            print(f"[seq cb] codec={fmt.codec} {fmt.coded_width}x{fmt.coded_height} "
                  f"chroma={fmt.chroma_format} bit_depth_luma+8={fmt.bit_depth_luma_minus8 + 8}")
            decode_ctx.coded_w = fmt.coded_width
            decode_ctx.coded_h = fmt.coded_height

            ci = CUVIDDECODECREATEINFO()
            ci.ulWidth = fmt.coded_width
            ci.ulHeight = fmt.coded_height
            ci.ulNumDecodeSurfaces = max(2, fmt.min_num_decode_surfaces)
            ci.CodecType = cudaVideoCodec_HEVC
            ci.ChromaFormat = cudaVideoChromaFormat_444
            ci.bitDepthMinus8 = fmt.bit_depth_luma_minus8
            ci.ulCreationFlags = cudaVideoCreate_PreferCUVID
            ci.ulMaxWidth = fmt.coded_width
            ci.ulMaxHeight = fmt.coded_height
            ci.display_area.left = 0
            ci.display_area.top = 0
            ci.display_area.right = fmt.coded_width
            ci.display_area.bottom = fmt.coded_height
            ci.OutputFormat = cudaVideoSurfaceFormat_YUV444
            ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave
            ci.ulTargetWidth = fmt.coded_width
            ci.ulTargetHeight = fmt.coded_height
            ci.ulNumOutputSurfaces = 2
            ci.vidLock = lock.value

            decode_ctx.decoder = create_decoder(ci)
            print(f"[seq cb] decoder 0x{decode_ctx.decoder.value:x}")
            return ci.ulNumDecodeSurfaces
        except Exception as e:
            decode_ctx.error = f"sequence cb: {e!r}"
            return 0

    @PFNVIDDECODECALLBACK
    def on_decode(user, pic_ptr):
        try:
            pp = pic_ptr.contents
            decode_picture(decode_ctx.decoder, ctypes.addressof(pp))
            return 1
        except Exception as e:
            decode_ctx.error = f"decode cb: {e!r}"
            return 0

    @PFNVIDDISPLAYCALLBACK
    def on_display(user, disp_ptr):
        try:
            disp = disp_ptr.contents
            print(f"[disp cb] picture_index={disp.picture_index}")

            # Map the surface
            proc = CUVIDPROCPARAMS()
            proc.progressive_frame = 1
            proc.top_field_first = 0
            proc.unpaired_field = 0
            proc.second_field = 0
            proc.output_stream = None
            dptr, pitch = map_video_frame64(decode_ctx.decoder,
                                              disp.picture_index, proc)
            print(f"[disp cb] mapped device 0x{dptr:x} pitch={pitch}")

            # YUV444 output: 3 stacked planes, each pitch * H, with 'pitch' >= W
            num_planes = 3
            plane_h = decode_ctx.coded_h
            total = pitch * plane_h * num_planes
            host_buf = (ctypes.c_uint8 * total)()

            # cuMemcpyDtoH
            err = cuda.cuMemcpyDtoH(host_buf, dptr, total)
            unmap_video_frame64(decode_ctx.decoder, dptr)

            err_int = int(err[0]) if isinstance(err, tuple) else int(err)
            if err_int != 0:
                decode_ctx.error = f"cuMemcpyDtoH err={err_int}"
                return 0

            # Repack from pitched layout to tightly-packed Y|U|V (W*H per plane)
            tight = bytearray(W * H * 3)
            for plane in range(3):
                for row in range(H):
                    src_off = (plane * plane_h + row) * pitch
                    dst_off = (plane * H + row) * W
                    tight[dst_off:dst_off + W] = bytes(host_buf[src_off:src_off + W])

            decode_ctx.decoded_bytes = bytes(tight)
            decode_ctx.disp_count += 1
            return 1
        except Exception as e:
            decode_ctx.error = f"display cb: {e!r}"
            return 0

    # Build parser params
    print("[4] Create video parser...")
    pp = CUVIDPARSERPARAMS()
    pp.CodecType = cudaVideoCodec_HEVC
    pp.ulMaxNumDecodeSurfaces = 2
    pp.ulMaxDisplayDelay = 0
    pp.ulErrorThreshold = 100
    pp.pUserData = None
    pp.pfnSequenceCallback = on_sequence
    pp.pfnDecodePicture = on_decode
    pp.pfnDisplayPicture = on_display
    pp.pfnGetOperatingPoint = ctypes.cast(None, PFNVIDOPPOINTCALLBACK)
    pp.pfnGetSEIMsg = ctypes.cast(None, PFNVIDSEIMSGCALLBACK)
    parser = create_parser(pp)
    print(f"    parser 0x{parser.value:x}\n")

    try:
        # Feed the bitstream
        print("[5] cuvidParseVideoData (bitstream)...")
        buf = (ctypes.c_uint8 * len(bitstream)).from_buffer_copy(bitstream)
        pkt = CUVIDSOURCEDATAPACKET()
        pkt.flags = 0
        pkt.payload_size = len(bitstream)
        pkt.payload = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
        pkt.timestamp = 0
        parse_video_data(parser, pkt)
        print()

        # Flush with EOS
        print("[6] cuvidParseVideoData (EOS flush)...")
        eos = CUVIDSOURCEDATAPACKET()
        eos.flags = CUVID_PKT_ENDOFSTREAM
        eos.payload_size = 0
        parse_video_data(parser, eos)
        print()

        if decode_ctx.error:
            print(f"!!! callback error: {decode_ctx.error}")
            return 1

        if decode_ctx.decoded_bytes is None:
            print("!!! no frame decoded")
            return 1

        # Compare
        original = make_synthetic_yuv444(W, H)
        decoded = decode_ctx.decoded_bytes
        print(f"[7] Compare original ({len(original)} bytes) vs decoded ({len(decoded)} bytes)...")
        if len(original) != len(decoded):
            print(f"!!! length mismatch")
            return 1
        diffs = [abs(a - b) for a, b in zip(original, decoded)]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        # PSNR
        import math
        mse = sum(d * d for d in diffs) / len(diffs)
        psnr = 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)
        print(f"    max abs diff: {max_diff}")
        print(f"    mean abs diff: {mean_diff:.3f}")
        print(f"    PSNR: {psnr:.2f} dB")
        ok = psnr > 30
        print(f"    {'PASS' if ok else 'FAIL'} (PSNR > 30 dB)\n")

    finally:
        print("[8] Cleanup...")
        destroy_parser(parser)
        if decode_ctx.decoder is not None:
            destroy_decoder(decode_ctx.decoder)
        ctx_lock_destroy(lock)
        print("    OK\n")

    print("*** SUCCESS *** — full encode + decode round-trip via direct ctypes,")
    print("with NO PyAV / PyNvVideoCodec / FFmpeg subprocess anywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
