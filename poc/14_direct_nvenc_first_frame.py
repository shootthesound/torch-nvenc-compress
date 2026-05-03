"""14 — Direct NVENC first-frame encode (Option D, session 3 deliverable).

Builds on `13_direct_nvenc_scaffold.py`. Where the scaffold proved the
open/query/destroy lifecycle works, this PoC exercises the full encode path:

  1. Initialise CUDA + open encode session (as before)
  2. nvEncInitializeEncoder for HEVC YUV444 at constQP — the previously
     blocked call (NV_ENC_ERR_INVALID_VERSION) is now unblocked using
     SDK-13-correct version constants discovered from FFmpeg/nv-codec-headers
  3. nvEncCreateInputBuffer  — system-memory frame buffer
  4. nvEncCreateBitstreamBuffer — output bitstream buffer
  5. nvEncLockInputBuffer + memcpy a synthetic YUV444 gradient + Unlock
  6. nvEncEncodePicture  (force IDR for self-decodable first frame)
  7. nvEncLockBitstream + readout + Unlock — verify >0 bytes returned
  8. Cleanup all handles

If this PoC prints a non-zero bitstream length, the direct backend
codec path is functionally complete in software and we can move to
real activation tensor input + the NVDEC half of the round-trip.
"""

from __future__ import annotations

import sys

import torch
from cuda.bindings import driver as cuda

from nvenc_compress.direct import (
    create_instance,
    open_encode_session_cuda,
    destroy_encoder,
    get_last_error_string,
    NV_ENC_CODEC_HEVC_GUID,
    NV_ENC_PRESET_P4_GUID,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
)
from nvenc_compress.direct.structs import (
    initialize_encoder_hevc_yuv444,
    create_input_buffer,
    destroy_input_buffer,
    create_bitstream_buffer,
    destroy_bitstream_buffer,
    write_input_buffer,
    encode_picture,
    lock_and_read_bitstream,
    NV_ENC_BUFFER_FORMAT_YUV444,
)


# NV_ENC_PIC_FLAGS bits (from nvEncodeAPI.h)
NV_ENC_PIC_FLAG_FORCEINTRA = 1 << 0
NV_ENC_PIC_FLAG_FORCEIDR = 1 << 1
NV_ENC_PIC_FLAG_OUTPUT_SPSPPS = 1 << 2
NV_ENC_PIC_FLAG_EOS = 1 << 3

# NVENCSTATUS values we care about
NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = 14


def make_synthetic_yuv444(width: int, height: int) -> bytes:
    """Synthetic test frame: gradient on Y, mid-grey on U/V.

    Returns planar YUV444 layout (Y plane, then U plane, then V plane),
    each plane width*height uint8 bytes — total 3*width*height bytes.
    """
    y_plane = bytearray(width * height)
    for row in range(height):
        for col in range(width):
            # Simple diagonal gradient — easy for codec to compress
            y_plane[row * width + col] = (row + col) & 0xFF
    u_plane = bytes([128] * (width * height))
    v_plane = bytes([128] * (width * height))
    return bytes(y_plane) + u_plane + v_plane


def main() -> int:
    print("Direct NVENC first-frame encode (D session 3)\n")
    W, H = 256, 256
    QP = 18

    # 1: CUDA init
    print("[1] CUDA init...")
    err, = cuda.cuInit(0)
    if int(err) != 0:
        print(f"    cuInit failed: {err}")
        return 1
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    err, ctx = cuda.cuCtxGetCurrent()
    if int(ctx) == 0:
        print("    cuCtxGetCurrent returned NULL")
        return 1
    print(f"    ctx 0x{int(ctx):x}\n")

    # 2: NVENC instance + session
    print("[2] NVENC API instance + session...")
    table = create_instance()
    encoder = open_encode_session_cuda(table, int(ctx))
    print(f"    encoder 0x{encoder.value:x}\n")

    try:
        # 3: Initialise encoder
        print(f"[3] nvEncInitializeEncoder HEVC YUV444 {W}x{H} QP={QP}...")
        init, config = initialize_encoder_hevc_yuv444(
            table, encoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P4_GUID,
            W, H, QP, tuning=NV_ENC_TUNING_INFO_HIGH_QUALITY,
        )
        print(f"    init OK (chroma bitfield 0x{config.encodeCodecConfig.hevcConfig.flag_bitfield:x})\n")

        # 4: Create input buffer
        print("[4] nvEncCreateInputBuffer...")
        in_buf = create_input_buffer(table, encoder, W, H, NV_ENC_BUFFER_FORMAT_YUV444)
        print(f"    input buffer 0x{in_buf.value:x}\n")

        # 5: Create bitstream buffer
        print("[5] nvEncCreateBitstreamBuffer...")
        out_buf = create_bitstream_buffer(table, encoder)
        print(f"    bitstream buffer 0x{out_buf.value:x}\n")

        # 6: Synth frame + write to input buffer
        print("[6] Generate synthetic YUV444 frame + write to input buffer...")
        frame_bytes = make_synthetic_yuv444(W, H)
        print(f"    frame size {len(frame_bytes)} bytes (3*{W}*{H})")
        write_input_buffer(table, encoder, in_buf, frame_bytes, W, H)
        print("    written + unlocked\n")

        # 7: Encode picture (force IDR — self-decodable)
        print("[7] nvEncEncodePicture FORCEIDR...")
        flags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS
        s = encode_picture(table, encoder, in_buf, out_buf, W, H,
                            pic_flags=flags, buffer_format=NV_ENC_BUFFER_FORMAT_YUV444)
        if s == NV_ENC_SUCCESS:
            print("    status SUCCESS — bitstream ready\n")
            need_flush = False
        elif s == NV_ENC_ERR_NEED_MORE_INPUT:
            print("    status NEED_MORE_INPUT — encoder buffering, will flush\n")
            need_flush = True
        else:
            err = get_last_error_string(table, encoder)
            print(f"    nvEncEncodePicture failed: status={s}")
            print(f"    driver error string: {err!r}")
            return 1

        # If buffering, send EOS to flush
        if need_flush:
            print("[7b] Flushing encoder with EOS...")
            s = encode_picture(table, encoder, None, out_buf, W, H,
                                pic_flags=NV_ENC_PIC_FLAG_EOS,
                                buffer_format=NV_ENC_BUFFER_FORMAT_YUV444)
            print(f"    EOS status={s}\n")

        # 8: Lock + read bitstream
        print("[8] nvEncLockBitstream + read...")
        bitstream = lock_and_read_bitstream(table, encoder, out_buf)
        print(f"    bitstream length: {len(bitstream)} bytes")
        if len(bitstream) > 0:
            head = bitstream[:16].hex()
            print(f"    first 16 bytes: {head}")
            # HEVC NAL units start with 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01 (Annex B)
            looks_hevc = bitstream[:4] == b"\x00\x00\x00\x01" or bitstream[:3] == b"\x00\x00\x01"
            print(f"    Annex B start code: {'YES' if looks_hevc else 'NO (not Annex B?)'}\n")
        else:
            print("    EMPTY bitstream — encode silently produced nothing\n")
            return 1

        # 9: Cleanup
        print("[9] Cleanup buffers...")
        destroy_bitstream_buffer(table, encoder, out_buf)
        destroy_input_buffer(table, encoder, in_buf)
        print("    OK\n")

    finally:
        print("[10] nvEncDestroyEncoder...")
        destroy_encoder(table, encoder)
        print("    OK\n")

    print("*** SUCCESS *** — first frame encoded via direct NVENC ctypes path,")
    print("with NO PyAV / PyNvVideoCodec / FFmpeg subprocess in the codec path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
