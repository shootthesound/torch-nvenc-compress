"""17 — Parallel-path demo: NVENC encode runs concurrently with GEMM compute.

The project's headline claim is that NVENC silicon is independent of CUDA
SM compute and the PCIe controller. This PoC demonstrates that empirically:

  - Stream A: heavy GEMM workload (saturates SMs)
  - Stream B: DirectBackend.encode_tensor_frames bound to stream B via
              nvEncSetIOCudaStreams (input fetch + bitstream copy queued
              on stream B)

If the streams are truly independent, total wall-clock should approach
max(compute_time, encode_time), not their sum.

Bench protocol:
  1. Time GEMM-only baseline.
  2. Time encode-only baseline.
  3. Time GEMM + encode SERIALIZED on default stream.
  4. Time GEMM + encode PARALLEL on streams A and B.
  5. Compute overlap_factor = (sequential_time / parallel_time).
     1.0 = no overlap; max(gemm,enc) / sum(gemm,enc) is the theoretical ceiling.
"""

from __future__ import annotations

import sys
import time

import torch
from cuda.bindings import driver as cuda

from nvenc_compress.direct.backend import DirectBackend


W, H = 256, 256
QP = 18
N_FRAMES = 64

# GEMM workload sized to take comparable time to the encode batch
GEMM_DIM = 4096
GEMM_ITERS = 30


def _stream_handle(s: torch.cuda.Stream) -> int:
    return s.cuda_stream


def gemm_workload(a: torch.Tensor, b: torch.Tensor, iters: int) -> torch.Tensor:
    out = a
    for _ in range(iters):
        out = out @ b
    return out


def main() -> int:
    print(f"Parallel-path demo — {N_FRAMES} frames {W}x{H} YUV444 + {GEMM_ITERS}x{GEMM_DIM}^2 matmul\n")

    # CUDA bring-up
    err, = cuda.cuInit(0)
    if int(err) != 0:
        print(f"cuInit failed: {err}"); return 1
    torch.cuda.init()

    # Compute streams
    stream_a = torch.cuda.Stream()  # GEMM
    stream_b = torch.cuda.Stream()  # NVENC
    print(f"stream A handle: 0x{_stream_handle(stream_a):x}  (compute)")
    print(f"stream B handle: 0x{_stream_handle(stream_b):x}  (encode)\n")

    # Stage tensors on default stream — done once, not timed
    a = torch.randn(GEMM_DIM, GEMM_DIM, device="cuda", dtype=torch.float16)
    b = torch.randn(GEMM_DIM, GEMM_DIM, device="cuda", dtype=torch.float16)

    # Synthetic uint8 frames already on GPU
    frames = torch.empty(N_FRAMES, 3, H, W, device="cuda", dtype=torch.uint8)
    rr = torch.arange(H, device="cuda")[:, None]
    cc = torch.arange(W, device="cuda")[None, :]
    for i in range(N_FRAMES):
        frames[i, 0] = ((rr + cc + i * 7) & 0xFF).to(torch.uint8)
    frames[:, 1] = 128
    frames[:, 2] = 128
    torch.cuda.synchronize()

    # Build the encoder bound to stream B
    print("Building DirectBackend bound to stream B...")
    backend = DirectBackend(height=H, width=W, qp=QP, cuda_stream=_stream_handle(stream_b))

    try:
        # ---- 1. GEMM-only baseline ----
        print("\n[1] GEMM-only (stream A)...")
        # Warmup
        with torch.cuda.stream(stream_a):
            _ = gemm_workload(a, b, 5)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream_a):
            r = gemm_workload(a, b, GEMM_ITERS)
        torch.cuda.synchronize()
        gemm_only_ms = (time.perf_counter() - t0) * 1000
        # Force materialization
        _ = r.sum().item()
        print(f"    {gemm_only_ms:.1f} ms")

        # ---- 2. Encode-only baseline (stream B) ----
        print("\n[2] Encode-only (stream B)...")
        # Warmup
        _ = backend.encode_tensor_frames(frames[:1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        packets_solo = backend.encode_tensor_frames(frames)
        torch.cuda.synchronize()
        encode_only_ms = (time.perf_counter() - t0) * 1000
        print(f"    {encode_only_ms:.1f} ms ({encode_only_ms / N_FRAMES:.2f} ms/frame)")

        # ---- 3. Serialized: encode then GEMM (default stream pattern) ----
        print("\n[3] Serialized (encode -> wait -> GEMM)...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = backend.encode_tensor_frames(frames)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream_a):
            r = gemm_workload(a, b, GEMM_ITERS)
        torch.cuda.synchronize()
        serialized_ms = (time.perf_counter() - t0) * 1000
        _ = r.sum().item()
        print(f"    {serialized_ms:.1f} ms")

        # ---- 4. Parallel: kick off GEMM on stream A, encode on stream B ----
        print("\n[4] Parallel (GEMM on A + encode on B simultaneously)...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream_a):
            r = gemm_workload(a, b, GEMM_ITERS)
        # encode_tensor_frames runs on backend's bound stream B; the encode
        # API calls themselves are CPU-issued from the main thread but the
        # cuMemcpyDtoDAsync + nvEncEncodePicture are queued on B
        _ = backend.encode_tensor_frames(frames)
        torch.cuda.synchronize()
        parallel_ms = (time.perf_counter() - t0) * 1000
        _ = r.sum().item()
        print(f"    {parallel_ms:.1f} ms")

        # ---- Summary ----
        print("\n--- summary ---")
        print(f"GEMM only:        {gemm_only_ms:.1f} ms")
        print(f"Encode only:      {encode_only_ms:.1f} ms")
        print(f"Sum (no overlap): {gemm_only_ms + encode_only_ms:.1f} ms")
        print(f"Max (full overlap floor): {max(gemm_only_ms, encode_only_ms):.1f} ms")
        print(f"Serialized:       {serialized_ms:.1f} ms  (pays full cost, no overlap)")
        print(f"Parallel:         {parallel_ms:.1f} ms")
        if parallel_ms > 0:
            speedup = serialized_ms / parallel_ms
            ceiling = (gemm_only_ms + encode_only_ms) / max(gemm_only_ms, encode_only_ms)
            print(f"\nSpeedup parallel vs serialized: {speedup:.2f}x")
            print(f"Theoretical ceiling:            {ceiling:.2f}x")
            overlap_pct = 100 * (1 - (parallel_ms - max(gemm_only_ms, encode_only_ms))
                                 / (gemm_only_ms + encode_only_ms - max(gemm_only_ms, encode_only_ms))) \
                if (gemm_only_ms + encode_only_ms - max(gemm_only_ms, encode_only_ms)) > 0 else 0
            print(f"Overlap fraction realized:      {overlap_pct:.1f}%")

    finally:
        backend.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
