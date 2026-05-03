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
import os
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
    # Lower-level struct types + version constants for the pre-bound fast path
    NV_ENC_PIC_PARAMS, NV_ENC_PIC_PARAMS_VER,
    NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_BITSTREAM_VER,
    NV_ENC_MAP_INPUT_RESOURCE, NV_ENC_MAP_INPUT_RESOURCE_VER,
)
from .api import NVENCSTATUS
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

        # ---- Pre-bound ctypes function wrappers + reusable structs --------
        # Wrapping table.nvEncXxx with CFUNCTYPE allocates ~hundreds of bytes
        # and runs ctypes meta-machinery; doing it per call costs a meaningful
        # fraction of per-frame latency. Bind once here, reuse forever.
        from ctypes import CFUNCTYPE, POINTER, c_void_p
        self._fn_map = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_MAP_INPUT_RESOURCE))(
            self._table.nvEncMapInputResource
        )
        self._fn_unmap = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(
            self._table.nvEncUnmapInputResource
        )
        self._fn_encode = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_PIC_PARAMS))(
            self._table.nvEncEncodePicture
        )
        self._fn_lock = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_LOCK_BITSTREAM))(
            self._table.nvEncLockBitstream
        )
        self._fn_unlock = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(
            self._table.nvEncUnlockBitstream
        )
        # Persistent NV_ENC_PIC_PARAMS — most fields are the same every call.
        # Only inputBuffer / outputBitstream / encodePicFlags change per frame.
        self._pic = NV_ENC_PIC_PARAMS()
        self._pic.version = NV_ENC_PIC_PARAMS_VER
        self._pic.inputWidth = width
        self._pic.inputHeight = height
        self._pic.inputPitch = width
        self._pic.bufferFmt = NV_ENC_BUFFER_FORMAT_YUV444
        self._pic.pictureStruct = 1  # NV_ENC_PIC_STRUCT_FRAME
        # Persistent lock-bitstream struct.
        self._lock = NV_ENC_LOCK_BITSTREAM()
        self._lock.version = NV_ENC_LOCK_BITSTREAM_VER
        # Per-slot map-input-resource structs (filled lazily after registration).
        self._map_structs: list[NV_ENC_MAP_INPUT_RESOURCE] = []

        # Optional native (C) hot-loop helper. OPT-IN via env var
        # NVENC_DIRECT_NATIVE=1 — the C path is correct but doesn't beat
        # the pre-bound Python loop on this hot path because NVENC's own
        # submission-latency floor (~100us per EncodePicture call)
        # dominates over the ctypes overhead. Kept in tree for future
        # workloads where the pool depth or call rate make Python overhead
        # the bottleneck again.
        if os.environ.get("NVENC_DIRECT_NATIVE", "0") == "1":
            from . import _native
            self._native_lib = _native.get_lib()
        else:
            self._native_lib = None
        self._native_ctx = None  # built on first encode (after _ensure_cuda_buf)
        # Persistent destination buffer for native encode output. Grows on
        # demand — zeroing 100+ MB per call would dominate per-frame time.
        self._native_dest_buf = None
        self._native_dest_cap = 0
        self._native_offsets_arr = None
        self._native_sizes_arr = None
        self._native_offsets_cap = 0

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
            rr = register_cuda_resource(
                self._table, self._encoder,
                cuda_ptr=buf,
                width=self.width, height=self.height,
                pitch=self.width,
                buffer_format=NV_ENC_BUFFER_FORMAT_YUV444,
            )
            self._registered_res_pool.append(rr)
            # Pre-fill the per-slot map struct (registeredResource is the only
            # field that needs to be set; mappedResource is filled by the call).
            mp = NV_ENC_MAP_INPUT_RESOURCE()
            mp.version = NV_ENC_MAP_INPUT_RESOURCE_VER
            mp.registeredResource = rr.value if hasattr(rr, "value") else rr
            self._map_structs.append(mp)

    def _build_native_ctx(self) -> None:
        """Construct the C-side EncodeContext after the CUDA staging buffers
        have been allocated and registered. Called once on the first encode."""
        from . import _native
        # Find function pointer addresses. ctypes function objects can be
        # converted to raw addresses via ctypes.cast(fn, c_void_p).value.
        def addr(fn):
            return ctypes.cast(fn, ctypes.c_void_p).value or 0

        # Resolve cuda driver memcpy functions through cuda-python
        # The cuMemcpy*/cuStreamSync etc. APIs are ctypes-bound at the lib
        # level; we want raw addresses. cuda.bindings.driver provides them
        # but doesn't expose addresses directly. Easiest: load nvcuda.dll
        # ourselves and resolve symbols via GetProcAddress.
        nvcuda = ctypes.CDLL("nvcuda.dll")
        fn_memcpy_dtod = ctypes.cast(nvcuda.cuMemcpyDtoD_v2, ctypes.c_void_p).value or 0
        fn_memcpy_dtod_async = ctypes.cast(nvcuda.cuMemcpyDtoDAsync_v2, ctypes.c_void_p).value or 0

        # NVENC function pointers come from the API table (already integers)
        # via the same table we pre-bound CFUNCTYPE wrappers from.
        # The table holds raw void* pointers in its struct fields.
        t = self._table

        ctx = _native.EncodeContext()
        ctx.fn_map = t.nvEncMapInputResource
        ctx.fn_unmap = t.nvEncUnmapInputResource
        ctx.fn_encode = t.nvEncEncodePicture
        ctx.fn_lock = t.nvEncLockBitstream
        ctx.fn_unlock = t.nvEncUnlockBitstream
        ctx.fn_memcpy_dtod = fn_memcpy_dtod
        ctx.fn_memcpy_dtod_async = fn_memcpy_dtod_async

        ctx.encoder = self._encoder.value if hasattr(self._encoder, "value") else self._encoder
        ctx.cuda_stream = self._user_stream if self._user_stream else 0

        ctx.pool_size = self._output_pool_size
        ctx.per_frame_bytes = self._cuda_buf_size

        # Per-slot CUDA dst pointers (uint64 array)
        n = self._output_pool_size
        DstArrT = ctypes.c_uint64 * n
        self._native_dst_arr = DstArrT(*self._cuda_bufs)
        ctx.slot_dst_ptrs = self._native_dst_arr

        # Per-slot output bitstream handles
        OutArrT = ctypes.c_void_p * n
        self._native_out_arr = OutArrT(
            *(b.value if hasattr(b, "value") else b for b in self._out_pool)
        )
        ctx.out_buffers = self._native_out_arr

        # Per-slot map struct pointers (addresses of pre-allocated structs)
        MapArrT = ctypes.c_void_p * n
        self._native_map_arr = MapArrT(
            *(ctypes.addressof(m) for m in self._map_structs)
        )
        ctx.map_struct_ptrs = self._native_map_arr
        ctx.map_mapped_resource_offset = NV_ENC_MAP_INPUT_RESOURCE.mappedResource.offset

        # Pic params struct + field offsets
        ctx.pic_struct_ptr = ctypes.addressof(self._pic)
        ctx.pic_inputBuffer_offset = NV_ENC_PIC_PARAMS.inputBuffer.offset
        ctx.pic_outputBitstream_offset = NV_ENC_PIC_PARAMS.outputBitstream.offset
        ctx.pic_encodePicFlags_offset = NV_ENC_PIC_PARAMS.encodePicFlags.offset

        # Lock struct + field offsets
        ctx.lock_struct_ptr = ctypes.addressof(self._lock)
        ctx.lock_outputBitstream_offset = NV_ENC_LOCK_BITSTREAM.outputBitstream.offset
        ctx.lock_bitstreamSizeInBytes_offset = NV_ENC_LOCK_BITSTREAM.bitstreamSizeInBytes.offset
        ctx.lock_bitstreamBufferPtr_offset = NV_ENC_LOCK_BITSTREAM.bitstreamBufferPtr.offset

        # IDR + SPSPPS flags constant
        ctx.flags_idr = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS

        self._native_ctx = ctx

    def _encode_tensor_frames_native(self, tensor, N, per_frame_bytes, K):
        """Run the encode hot loop in C. Returns list of packet bytes."""
        if self._native_ctx is None:
            self._build_native_ctx()

        # Build src_ptrs array vectorized via numpy
        base = int(tensor.data_ptr())
        stride = per_frame_bytes
        src_np = np.arange(N, dtype=np.uint64) * np.uint64(stride) + np.uint64(base)
        src_ptr = src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

        # Reuse persistent dest buffer. Grow on demand. Heuristic: assume
        # at least 4x compression — N * per_frame_bytes / 4 is a generous
        # cap (real ratio at QP=18 is 30x+; this gives 7x headroom). Any
        # call that overflows will retry with the input-sized cap.
        needed_cap = max(N * per_frame_bytes // 4, 4 * 1024 * 1024)
        if self._native_dest_buf is None or self._native_dest_cap < needed_cap:
            self._native_dest_buf = (ctypes.c_uint8 * needed_cap)()
            self._native_dest_cap = needed_cap
        if self._native_offsets_arr is None or self._native_offsets_cap < N:
            self._native_offsets_arr = (ctypes.c_uint32 * N)()
            self._native_sizes_arr = (ctypes.c_uint32 * N)()
            self._native_offsets_cap = N

        dest_buf = self._native_dest_buf
        dest_cap = self._native_dest_cap
        offsets = self._native_offsets_arr
        sizes = self._native_sizes_arr
        dest_addr = ctypes.addressof(dest_buf)

        rc = self._native_lib.encode_batch(
            ctypes.byref(self._native_ctx),
            N,
            src_ptr,
            ctypes.cast(dest_buf, ctypes.c_void_p),
            dest_cap,
            offsets,
            sizes,
        )
        del src_np  # keep alive across call
        if rc == 1:
            # Retry once with full input-sized cap
            needed_cap = N * per_frame_bytes
            self._native_dest_buf = (ctypes.c_uint8 * needed_cap)()
            self._native_dest_cap = needed_cap
            dest_buf = self._native_dest_buf
            dest_addr = ctypes.addressof(dest_buf)
            rc = self._native_lib.encode_batch(
                ctypes.byref(self._native_ctx),
                N, src_ptr,
                ctypes.cast(dest_buf, ctypes.c_void_p),
                needed_cap, offsets, sizes,
            )
            if rc != 0:
                raise RuntimeError(f"native encode_batch failed (after grow) rc={rc}")
        elif rc != 0:
            raise RuntimeError(f"native encode_batch failed with code {rc}")

        out: list[bytes] = [
            ctypes.string_at(dest_addr + offsets[i], sizes[i])
            for i in range(N)
        ]
        return out

    def encode_tensor_frames(self, tensor: torch.Tensor) -> list[bytes]:
        """Encode N frames from a CUDA torch tensor — no host-side copy.

        Tensor shape: [N, 3, H, W] uint8, contiguous, on CUDA.

        Uses the native C encode-loop helper when available (closes the
        per-frame Python overhead); falls back to the pure-Python pre-bound
        loop otherwise.

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

        # Native fast path — runs the per-frame loop in C, bypassing the
        # 6 ctypes calls per frame in the Python loop below.
        if self._native_lib is not None:
            return self._encode_tensor_frames_native(tensor, N, per_frame_bytes, K)

        # Bind everything we need into local names — local lookups are
        # ~5x faster than attribute lookups in CPython's hot loop.
        encoder = self._encoder
        cuda_bufs = self._cuda_bufs
        out_pool = self._out_pool
        map_structs = self._map_structs
        pic = self._pic
        lock_struct = self._lock
        fn_map = self._fn_map
        fn_unmap = self._fn_unmap
        fn_encode = self._fn_encode
        fn_lock = self._fn_lock
        fn_unlock = self._fn_unlock
        stream = self._user_stream
        cuMemcpyDtoDAsync = cuda.cuMemcpyDtoDAsync
        cuMemcpyDtoD = cuda.cuMemcpyDtoD
        from ctypes import byref, string_at, c_void_p
        flags_idr = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS

        # in_flight[slot] = frame_index, or -1 if empty
        in_flight = [-1] * K
        out: list[Optional[bytes]] = [None] * N

        def drain_slot(slot: int) -> None:
            frame_idx = in_flight[slot]
            if frame_idx < 0:
                return
            buf = out_pool[slot]
            lock_struct.outputBitstream = buf
            s = fn_lock(encoder, byref(lock_struct))
            if s != 0:
                raise RuntimeError(f"nvEncLockBitstream failed: status={s}")
            out[frame_idx] = string_at(lock_struct.bitstreamBufferPtr,
                                        lock_struct.bitstreamSizeInBytes)
            s = fn_unlock(encoder, buf)
            if s != 0:
                raise RuntimeError(f"nvEncUnlockBitstream failed: status={s}")
            # Unmap the input that was used for this slot
            mp = map_structs[slot]
            s = fn_unmap(encoder, mp.mappedResource)
            if s != 0:
                raise RuntimeError(f"nvEncUnmapInputResource failed: status={s}")
            mp.mappedResource = None
            in_flight[slot] = -1

        for i in range(N):
            slot = i % K

            # If this slot still holds an in-flight frame, drain it before reuse
            if in_flight[slot] >= 0:
                drain_slot(slot)

            # GPU-side copy from tensor[i] into THIS slot's registered staging buffer
            src_ptr = int(tensor[i].data_ptr())
            slot_dst = cuda_bufs[slot]
            if stream is not None:
                err = cuMemcpyDtoDAsync(slot_dst, src_ptr, per_frame_bytes, stream)
            else:
                err = cuMemcpyDtoD(slot_dst, src_ptr, per_frame_bytes)
            err_int = int(err[0]) if isinstance(err, tuple) else int(err)
            if err_int != 0:
                raise RuntimeError(f"cuMemcpy(D2D) failed: {err_int}")

            # Map input resource for this slot (re-uses the per-slot struct)
            mp = map_structs[slot]
            s = fn_map(encoder, byref(mp))
            if s != 0:
                raise RuntimeError(f"nvEncMapInputResource failed: status={s}")

            # Update only the changing fields of the persistent pic struct
            pic.inputBuffer = mp.mappedResource
            pic.outputBitstream = out_pool[slot]
            pic.encodePicFlags = flags_idr if i == 0 else 0

            s = fn_encode(encoder, byref(pic))
            if s != 0 and s != 14:
                fn_unmap(encoder, mp.mappedResource)
                mp.mappedResource = None
                raise RuntimeError(f"nvEncEncodePicture failed: status={s}")
            if s == 0:
                in_flight[slot] = i
            else:
                # NEED_MORE_INPUT — shouldn't happen with bf=0 but unmap to be safe
                fn_unmap(encoder, mp.mappedResource)
                mp.mappedResource = None

        # Drain any remaining in-flight frames
        for slot in range(K):
            if in_flight[slot] >= 0:
                drain_slot(slot)

        return [p for p in out if p is not None]

    # ---- decode side ------------------------------------------------------

    def decode_frames(self, packets: list[bytes], n_frames: int) -> np.ndarray:
        """Decode HEVC packets -> [N, 3, H, W] uint8 numpy array.

        For zero-copy decode that returns a torch CUDA tensor (no host
        round-trip), use decode_frames_cuda() instead — it's substantially
        faster on the decode path."""
        out_t = self.decode_frames_cuda(packets, n_frames)
        return out_t.cpu().numpy()

    def decode_frames_cuda(self, packets: list[bytes], n_frames: int) -> torch.Tensor:
        """Decode HEVC packets directly into a torch CUDA tensor [N, 3, H, W] uint8.

        Skips the host round-trip entirely: each decoded frame is copied via
        cuMemcpy2DAsync from the NVDEC-mapped pitched surface to the
        appropriate slice of a pre-allocated torch CUDA tensor. The output
        tensor stays on the GPU; caller can keep working with it there or
        pull to CPU explicitly with .cpu().
        """
        # Pre-allocate the destination on GPU
        out = torch.empty((n_frames, 3, self.height, self.width),
                           device="cuda", dtype=torch.uint8)
        out_base = int(out.data_ptr())
        # Strides: row stride within a plane = width (uint8), plane stride = H*W,
        # frame stride = 3*H*W. All in bytes since dtype=uint8.
        plane_bytes = self.height * self.width
        frame_bytes = 3 * plane_bytes

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
                if self._user_stream is not None:
                    proc.output_stream = self._user_stream
                dptr, pitch = map_video_frame64(st["decoder"], disp.picture_index, proc)

                plane_h = st["coded_h"]
                # NVDEC YUV444 surface: 3 planes stacked, each pitch * plane_h bytes.
                # Issue 3 cuMemcpy2DAsync (one per plane), copying width bytes per
                # row, self.height rows, srcPitch=pitch, dstPitch=self.width.
                # Output destination is the appropriate slice in our torch tensor.
                seen = st["seen"]
                frame_off = out_base + seen * frame_bytes
                for plane in range(3):
                    src = dptr + plane * pitch * plane_h
                    dst = frame_off + plane * plane_bytes
                    m2d = cuda.CUDA_MEMCPY2D()
                    m2d.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
                    m2d.srcDevice = src
                    m2d.srcPitch = pitch
                    m2d.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
                    m2d.dstDevice = dst
                    m2d.dstPitch = self.width
                    m2d.WidthInBytes = self.width
                    m2d.Height = self.height
                    if self._user_stream is not None:
                        err = cuda.cuMemcpy2DAsync(m2d, self._user_stream)
                    else:
                        err = cuda.cuMemcpy2D(m2d)
                    err_int = int(err[0]) if isinstance(err, tuple) else int(err)
                    if err_int != 0:
                        unmap_video_frame64(st["decoder"], dptr)
                        st["error"] = f"cuMemcpy2D plane {plane} err={err_int}"
                        return 0
                unmap_video_frame64(st["decoder"], dptr)
                st["seen"] = seen + 1
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
                blob = b"".join(packets)
                buf = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
                pkt = CUVIDSOURCEDATAPACKET()
                pkt.flags = 0
                pkt.payload_size = len(blob)
                pkt.payload = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
                parse_video_data(parser, pkt)

                eos = CUVIDSOURCEDATAPACKET()
                eos.flags = CUVID_PKT_ENDOFSTREAM
                parse_video_data(parser, eos)

                if st["error"]:
                    raise RuntimeError(st["error"])
                if st["seen"] != n_frames:
                    raise RuntimeError(
                        f"decoded {st['seen']} frames, expected {n_frames}"
                    )
                # If async copies were issued on a stream, sync it here
                if self._user_stream is not None:
                    err = cuda.cuStreamSynchronize(self._user_stream)
                    err_int = int(err[0]) if isinstance(err, tuple) else int(err)
                    if err_int != 0:
                        raise RuntimeError(f"cuStreamSynchronize err={err_int}")
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
