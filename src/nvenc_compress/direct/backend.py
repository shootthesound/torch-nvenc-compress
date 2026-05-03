"""DirectBackend — drop-in replacement for CodecSession using direct ctypes.

Same encode_frames / decode_frames interface as CodecSession, but the codec
path is pure ctypes against the driver-shipped NVENC + NVDEC DLLs:
  - no PyAV subprocess
  - no PyNvVideoCodec dependency
  - no FFmpeg subprocess

The encode side keeps one persistent NVENC session open across calls
(matching CodecSession's amortised init behaviour). The decode side
re-creates the parser per `decode_frames` call because cuvidParser doesn't
cleanly reset between independent IDR-led streams (same constraint as
CodecSession's PyAV path).

Limitations relative to CodecSession:
- Currently uses NVENC's system-memory input buffer (write_input_buffer
  copies frame bytes through host RAM). The CUDA-pointer zero-copy path
  via nvEncRegisterResource is session-6 work; until then the host->device
  copy is a real cost on the encode hot path.
- The decode path also goes through host memory (cuMemcpyDtoH after
  cuvidMapVideoFrame64). Zero-copy would map the device ptr to torch.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np
import torch
from cuda.bindings import driver as cuda

from .api import (
    create_instance, open_encode_session_cuda, destroy_encoder,
    NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P4_GUID,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
)
from .structs import (
    initialize_encoder_hevc_yuv444,
    create_input_buffer, destroy_input_buffer,
    create_bitstream_buffer, destroy_bitstream_buffer,
    write_input_buffer, encode_picture, lock_and_read_bitstream,
    register_cuda_resource, unregister_resource,
    map_input_resource, unmap_input_resource,
    set_io_cuda_streams,
    NV_ENC_BUFFER_FORMAT_YUV444,
)
from .decoder import (
    CUVIDPARSERPARAMS, CUVIDSOURCEDATAPACKET, CUVIDPROCPARAMS,
    CUVIDDECODECREATEINFO,
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


def _ensure_cuda_ctx() -> int:
    """Make sure a CUDA context exists (via torch) and return its handle."""
    err, = cuda.cuInit(0)
    if int(err) != 0:
        raise RuntimeError(f"cuInit failed: {err}")
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    err, ctx = cuda.cuCtxGetCurrent()
    if int(ctx) == 0:
        raise RuntimeError("cuCtxGetCurrent returned NULL context")
    return int(ctx)


class DirectBackend:
    """Encode / decode HEVC YUV444 via direct ctypes into NVENC + NVDEC.

    Matches CodecSession's encode_frames / decode_frames signature so it
    can substitute in the high-level compress() / decompress() pipeline
    helpers without further changes.
    """

    def __init__(self, height: int, width: int, qp: int = 18,
                  cuda_stream: Optional[int] = None,
                  output_pool_size: int = 8):
        """If `cuda_stream` is given (an integer CUstream handle), encoder
        input fetch and bitstream copy are bound to it via
        nvEncSetIOCudaStreams, and per-frame cuMemcpyDtoDAsync uses the
        same stream. This is the parallel-path entry point — pass torch's
        own stream handle to interleave encode with model compute.

        `output_pool_size` controls how many output bitstream buffers we
        allocate as a ring. Up to that many frames can be in flight on
        NVENC concurrently — the rest of the encode pipeline blocks on
        lock_bitstream. Default 8 is a reasonable trade between memory
        and pipelining depth (each buffer is small, ~few KB)."""
        self.height = height
        self.width = width
        self.qp = qp
        self._user_stream = cuda_stream
        self._stream_handle_storage = None  # ctypes c_void_p kept alive for the encoder
        self._output_pool_size = max(1, int(output_pool_size))

        self._cuda_ctx = _ensure_cuda_ctx()

        # Persistent encoder session
        self._table = create_instance()
        self._encoder = open_encode_session_cuda(self._table, self._cuda_ctx)
        try:
            initialize_encoder_hevc_yuv444(
                self._table, self._encoder,
                NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P4_GUID,
                width, height, qp, tuning=NV_ENC_TUNING_INFO_HIGH_QUALITY,
            )
            self._in_buf = create_input_buffer(
                self._table, self._encoder, width, height, NV_ENC_BUFFER_FORMAT_YUV444
            )
            # Ring of output bitstream buffers. encode_tensor_frames cycles
            # through them; lock_and_read_bitstream is deferred until the
            # ring would otherwise wrap, so up to (pool - 1) encodes can be
            # in flight before we block.
            self._out_pool = [
                create_bitstream_buffer(self._table, self._encoder)
                for _ in range(self._output_pool_size)
            ]
            # Backwards-compat alias for the original single-buffer code
            # path (encode_frames host-buf path still uses one buffer).
            self._out_buf = self._out_pool[0]

            if self._user_stream is not None:
                # Encoder will queue input-fetch + output-bitstream-copy on
                # the user's stream. NV_ENC_CUSTREAM_PTR is documented as
                # CUstream*, so we hand it the address of a c_void_p that
                # holds the stream handle — must keep this alive for the
                # encoder's lifetime.
                self._stream_handle_storage = ctypes.c_void_p(self._user_stream)
                set_io_cuda_streams(
                    self._table, self._encoder,
                    ctypes.addressof(self._stream_handle_storage),
                    ctypes.addressof(self._stream_handle_storage),
                )
        except Exception:
            destroy_encoder(self._table, self._encoder)
            raise

        # Lazy zero-copy resources (allocated on first encode_tensor_frames call).
        # Each ring slot owns its own staging buffer + registration so multiple
        # frames can be in flight concurrently without overwriting each other's
        # encoder input.
        self._cuda_bufs: list[int] = []           # K CUdeviceptrs
        self._cuda_buf_size = 0
        self._registered_res_pool: list = []      # K registered_resource handles

    # ---- encode side ------------------------------------------------------

    def encode_frames(self, frames: np.ndarray) -> list[bytes]:
        """Encode N frames [N, 3, H, W] uint8 -> list of HEVC packet bytes.

        Frame 0 is forced IDR with inline SPS/PPS so the bitstream is
        self-decodable without external extradata.
        """
        if frames.dtype != np.uint8:
            raise ValueError(f"expected uint8 frames, got {frames.dtype}")
        N, C, H, W = frames.shape
        if (C, H, W) != (3, self.height, self.width):
            raise ValueError(
                f"frame shape {C}x{H}x{W} doesn't match session "
                f"3x{self.height}x{self.width}"
            )

        out: list[bytes] = []
        for i in range(N):
            # Pack YUV444 planes contiguously: Y plane, then U plane, then V plane
            f = np.ascontiguousarray(frames[i])
            yuv_bytes = f.tobytes()  # 3 * H * W bytes in plane order

            write_input_buffer(self._table, self._encoder, self._in_buf,
                                yuv_bytes, self.width, self.height)

            flags = (NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) if i == 0 else 0
            s = encode_picture(self._table, self._encoder, self._in_buf,
                                self._out_buf, self.width, self.height,
                                pic_flags=flags,
                                buffer_format=NV_ENC_BUFFER_FORMAT_YUV444)
            if s != 0 and s != 14:  # 14 = NV_ENC_ERR_NEED_MORE_INPUT
                raise RuntimeError(f"encode_picture status={s}")
            if s == 0:
                pkt = lock_and_read_bitstream(self._table, self._encoder, self._out_buf)
                out.append(pkt)
            # else: encoder is buffering; output will come on a later call
        return out

    # ---- encode side, zero-copy from torch CUDA tensor -------------------

    def _ensure_cuda_buf(self) -> None:
        """Allocate K per-frame YUV444 staging buffers on the GPU and
        register each with NVENC. K matches the output ring size so each
        in-flight frame has its own input region.

        Lazy: only happens on first call to encode_tensor_frames."""
        if self._cuda_bufs:
            return
        size = 3 * self.height * self.width  # YUV444 8-bit, pitch == width
        self._cuda_buf_size = size
        for _ in range(self._output_pool_size):
            err, dptr = cuda.cuMemAlloc(size)
            if int(err) != 0:
                raise RuntimeError(f"cuMemAlloc({size}) failed: {err}")
            buf = int(dptr)
            self._cuda_bufs.append(buf)
            self._registered_res_pool.append(register_cuda_resource(
                self._table, self._encoder,
                cuda_ptr=buf,
                width=self.width, height=self.height,
                pitch=self.width,
                buffer_format=NV_ENC_BUFFER_FORMAT_YUV444,
            ))

    def encode_tensor_frames(self, tensor: torch.Tensor) -> list[bytes]:
        """Encode N frames from a CUDA torch tensor — no host-side copy.

        Tensor shape: [N, 3, H, W] uint8, contiguous, on CUDA.
        Per frame the path is:
          1. cuMemcpyDtoD from tensor[i].data_ptr to our registered staging buffer
          2. nvEncMapInputResource → mappedResource
          3. nvEncEncodePicture (FORCEIDR on i==0)
          4. nvEncLockBitstream / read / unlock
          5. nvEncUnmapInputResource

        The D2D memcpy is GPU-internal (no PCIe) so this is the closest the
        encoder gets to "true" zero-copy without us hijacking the tensor's
        own pointer (which we can't, because the encoder needs the buffer
        registered for its full lifetime).
        """
        if not tensor.is_cuda:
            raise ValueError("tensor must be on CUDA")
        if tensor.dtype != torch.uint8:
            raise ValueError(f"tensor must be uint8, got {tensor.dtype}")
        if tensor.dim() != 4 or tensor.shape[1:] != (3, self.height, self.width):
            raise ValueError(
                f"tensor shape {tuple(tensor.shape)} doesn't match "
                f"[N, 3, {self.height}, {self.width}]"
            )
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self._ensure_cuda_buf()

        N = tensor.shape[0]
        per_frame_bytes = self._cuda_buf_size
        K = self._output_pool_size

        # Per-frame state we need to remember across the ring window:
        #   in_flight[ring_slot] = (mapped_resource, frame_index_for_ordering)
        in_flight: list[Optional[tuple]] = [None] * K
        # Output packets indexed by frame number — encoder may emit them
        # in different order than we issue (esp. with B-frames), but we
        # disabled B-frames so order is preserved. Use a list anyway for
        # safety.
        out: list[Optional[bytes]] = [None] * N

        def drain_slot(slot: int) -> None:
            """Lock+read the bitstream buffer at ring[slot], record its
            packet, and unmap the input it was using."""
            entry = in_flight[slot]
            if entry is None:
                return
            mapped, frame_idx = entry
            pkt = lock_and_read_bitstream(self._table, self._encoder,
                                            self._out_pool[slot])
            out[frame_idx] = pkt
            unmap_input_resource(self._table, self._encoder, mapped)
            in_flight[slot] = None

        for i in range(N):
            slot = i % K

            # If this slot still holds an in-flight frame, drain it before
            # we overwrite — this is the only blocking sync point.
            if in_flight[slot] is not None:
                drain_slot(slot)

            # GPU-side copy from tensor[i] into THIS slot's registered
            # staging buffer. Slots have independent buffers so frames in
            # other slots are not corrupted while their encodes are in flight.
            src_ptr = int(tensor[i].data_ptr())
            slot_dst = self._cuda_bufs[slot]
            if self._user_stream is not None:
                err = cuda.cuMemcpyDtoDAsync(slot_dst, src_ptr,
                                              per_frame_bytes, self._user_stream)
            else:
                err = cuda.cuMemcpyDtoD(slot_dst, src_ptr, per_frame_bytes)
            err_int = int(err[0]) if isinstance(err, tuple) else int(err)
            if err_int != 0:
                raise RuntimeError(f"cuMemcpy(D2D) failed: {err_int}")

            mapped = map_input_resource(self._table, self._encoder,
                                          self._registered_res_pool[slot])
            flags = (NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) if i == 0 else 0
            s = encode_picture(self._table, self._encoder, mapped,
                                self._out_pool[slot], self.width, self.height,
                                pic_flags=flags,
                                buffer_format=NV_ENC_BUFFER_FORMAT_YUV444)
            if s != 0 and s != 14:
                # Best-effort cleanup of the unmapped input
                unmap_input_resource(self._table, self._encoder, mapped)
                raise RuntimeError(f"encode_picture status={s}")
            if s == 0:
                in_flight[slot] = (mapped, i)
            else:
                # NEED_MORE_INPUT — encoder is buffering. Unmap input now;
                # later encodes will produce the deferred output. Since we
                # disabled B-frames this shouldn't happen but guard anyway.
                unmap_input_resource(self._table, self._encoder, mapped)

        # Drain any remaining in-flight frames.
        for slot in range(K):
            if in_flight[slot] is not None:
                drain_slot(slot)

        return [p for p in out if p is not None]

    # ---- decode side ------------------------------------------------------

    def decode_frames(self, packets: list[bytes], n_frames: int) -> np.ndarray:
        """Decode HEVC packets -> [N, 3, H, W] uint8 numpy array."""
        out = np.empty((n_frames, 3, self.height, self.width), dtype=np.uint8)

        # State shared between callbacks
        st = {
            "decoder": None,
            "coded_w": 0,
            "coded_h": 0,
            "seen": 0,
            "error": None,
        }

        @PFNVIDSEQUENCECALLBACK
        def on_sequence(_user, fmt_ptr):
            try:
                fmt = fmt_ptr.contents
                st["coded_w"] = fmt.coded_width
                st["coded_h"] = fmt.coded_height
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
                st["decoder"] = create_decoder(ci)
                return ci.ulNumDecodeSurfaces
            except Exception as e:
                st["error"] = f"sequence cb: {e!r}"
                return 0

        @PFNVIDDECODECALLBACK
        def on_decode(_user, pic_ptr):
            try:
                pp = pic_ptr.contents
                decode_picture(st["decoder"], ctypes.addressof(pp))
                return 1
            except Exception as e:
                st["error"] = f"decode cb: {e!r}"
                return 0

        @PFNVIDDISPLAYCALLBACK
        def on_display(_user, disp_ptr):
            try:
                disp = disp_ptr.contents
                if st["seen"] >= n_frames:
                    return 1  # silently skip extras
                proc = CUVIDPROCPARAMS()
                proc.progressive_frame = 1
                dptr, pitch = map_video_frame64(st["decoder"], disp.picture_index, proc)

                plane_h = st["coded_h"]
                total = pitch * plane_h * 3
                host_buf = (ctypes.c_uint8 * total)()
                err = cuda.cuMemcpyDtoH(host_buf, dptr, total)
                unmap_video_frame64(st["decoder"], dptr)
                err_int = int(err[0]) if isinstance(err, tuple) else int(err)
                if err_int != 0:
                    st["error"] = f"cuMemcpyDtoH err={err_int}"
                    return 0

                # Reshape into a numpy view of the pitched buffer, then crop W per row
                pitched = np.frombuffer(host_buf, dtype=np.uint8).reshape(3, plane_h, pitch)
                out[st["seen"]] = pitched[:, : self.height, : self.width]
                st["seen"] += 1
                return 1
            except Exception as e:
                st["error"] = f"display cb: {e!r}"
                return 0

        # Build parser + lock
        lock = ctx_lock_create(self._cuda_ctx)
        try:
            pp = CUVIDPARSERPARAMS()
            pp.CodecType = cudaVideoCodec_HEVC
            pp.ulMaxNumDecodeSurfaces = 2
            pp.ulMaxDisplayDelay = 0
            pp.ulErrorThreshold = 100
            pp.pfnSequenceCallback = on_sequence
            pp.pfnDecodePicture = on_decode
            pp.pfnDisplayPicture = on_display
            pp.pfnGetOperatingPoint = ctypes.cast(None, PFNVIDOPPOINTCALLBACK)
            pp.pfnGetSEIMsg = ctypes.cast(None, PFNVIDSEIMSGCALLBACK)
            parser = create_parser(pp)
            try:
                # Concatenate packets — NVENC produces self-delimited Annex-B NAL
                # units, so the parser can split them itself.
                blob = b"".join(packets)
                buf = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
                pkt = CUVIDSOURCEDATAPACKET()
                pkt.flags = 0
                pkt.payload_size = len(blob)
                pkt.payload = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
                parse_video_data(parser, pkt)

                # Flush
                eos = CUVIDSOURCEDATAPACKET()
                eos.flags = CUVID_PKT_ENDOFSTREAM
                parse_video_data(parser, eos)

                if st["error"]:
                    raise RuntimeError(st["error"])
                if st["seen"] != n_frames:
                    raise RuntimeError(
                        f"decoded {st['seen']} frames, expected {n_frames}"
                    )
                return out
            finally:
                destroy_parser(parser)
                if st["decoder"] is not None:
                    destroy_decoder(st["decoder"])
        finally:
            ctx_lock_destroy(lock)

    # ---- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self._encoder is not None:
            for rr in self._registered_res_pool:
                try:
                    unregister_resource(self._table, self._encoder, rr)
                except Exception:
                    pass
            self._registered_res_pool = []
            for buf in self._cuda_bufs:
                try:
                    cuda.cuMemFree(buf)
                except Exception:
                    pass
            self._cuda_bufs = []
            for buf in self._out_pool:
                try:
                    destroy_bitstream_buffer(self._table, self._encoder, buf)
                except Exception:
                    pass
            self._out_pool = []
            self._out_buf = None
            try:
                destroy_input_buffer(self._table, self._encoder, self._in_buf)
            except Exception:
                pass
            try:
                destroy_encoder(self._table, self._encoder)
            except Exception:
                pass
            self._encoder = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
