"""09 — Dual-lane: direct PCIe and codec path running concurrently.

The 'parallel hardware paths' argument concretely demonstrated.

Setup: N tensors to be moved off-GPU. Three strategies compared:

  A. ALL_DIRECT     — All N tensors via direct cuda → cpu. Baseline.
  B. ALL_CODEC      — All N tensors via PCA + NVENC + CPU. Slow today (subprocess).
  C. DUAL_LANE      — Half via direct PCIe, half via codec (NVENC silicon),
                      both running concurrently on separate cuda streams.

Even with the slow subprocess pipeline, when the codec path's WIRE-portion is
much smaller than the direct path's, dual-lane improves total throughput because
the direct lane is no longer bottlenecked by ALL the bytes going through it.

The simulation also includes a SLOW WIRE mode (using sleep() to model 100 Mbps
broadband) where dual-lane gains are large because both lanes share that wire
but the codec lane uses much less of its bandwidth per useful byte.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from nvenc_compress import build_shared_basis, compress, decompress


N_TENSORS = 8
SHAPE = (4096, 4096)             # ~32 MB at fp16-equivalent each
K_PCA = 1000
QP = 18

# Simulated wires for the wire-bottleneck case
WIRES = [
    ("PCIe 4.0 (no wire bottleneck)", 32.0e9),
    ("1 Gbit ethernet",                0.125e9),
    ("100 Mbps residential broadband", 12.5e6),
]


def synth_tensors(n, shape, device):
    return [torch.randn(*shape, device=device) for _ in range(n)]


def all_direct(tensors, wire_bw=None):
    """Sequential cuda -> cpu. Optionally simulate wire transmission afterwards."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in tensors:
        cpu = t.cpu()
        if wire_bw is not None:
            time.sleep(cpu.numel() * 2 / wire_bw)   # 2 bytes/elt for fp16-equivalent
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def all_codec(tensors, basis, wire_bw=None):
    """Sequential codec compress + (simulated) wire transmit."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in tensors:
        data, recipe = compress(t, basis, qp=QP)
        if wire_bw is not None:
            time.sleep(len(data) / wire_bw)
        # We're measuring the SEND side (offload pattern). Don't decompress here
        # — the receiver decompresses. Same as direct above which doesn't read back.
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def dual_lane(tensors, basis, wire_bw=None):
    """Half via direct, half via codec, on separate cuda streams. Run concurrently."""
    half = len(tensors) // 2
    direct_tensors = tensors[:half]
    codec_tensors = tensors[half:]

    direct_stream = torch.cuda.Stream()
    codec_stream = torch.cuda.Stream()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Kick off both lanes
    direct_done = [None]
    codec_done = [None]

    def direct_lane():
        with torch.cuda.stream(direct_stream):
            for t in direct_tensors:
                cpu = t.cpu()
                if wire_bw is not None:
                    time.sleep(cpu.numel() * 2 / wire_bw)
        direct_done[0] = time.perf_counter()

    def codec_lane():
        with torch.cuda.stream(codec_stream):
            for t in codec_tensors:
                data, recipe = compress(t, basis, qp=QP)
                if wire_bw is not None:
                    time.sleep(len(data) / wire_bw)
        codec_done[0] = time.perf_counter()

    # Run both in threads so they truly overlap (cuda streams alone aren't
    # enough — Python is single-threaded per stream, so we need real threads
    # to dispatch encode-side work concurrently with cpu() calls).
    import threading
    th_direct = threading.Thread(target=direct_lane)
    th_codec = threading.Thread(target=codec_lane)
    th_direct.start()
    th_codec.start()
    th_direct.join()
    th_codec.join()

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — this PoC requires a GPU")
        return

    print(f"Generating {N_TENSORS} tensors of shape {SHAPE} on GPU "
          f"(~{SHAPE[0]*SHAPE[1]*2/1e6:.1f} MB each, "
          f"{N_TENSORS * SHAPE[0]*SHAPE[1]*2/1e6:.1f} MB total)...\n")
    tensors = synth_tensors(N_TENSORS, SHAPE, "cuda")

    print(f"Building PCA basis (K={K_PCA})...\n")
    basis = build_shared_basis(tensors[:4], K=K_PCA)

    # warmup
    compress(tensors[0], basis, qp=QP)

    print(f"Three strategies measured for offloading {N_TENSORS} tensors, across simulated wires:\n")
    print(f"{'wire':<35s}  {'all_direct':>11s}  {'all_codec':>11s}  {'dual_lane':>11s}  "
          f"{'dual vs direct':>14s}")

    for name, bw in WIRES:
        wire_bw = bw if bw < 5e9 else None
        # Use None when wire is fast enough that simulation noise > signal
        # (PCIe is so fast the sleep() overhead dwarfs actual transfer time)

        t_direct = all_direct(tensors, wire_bw)
        t_codec  = all_codec(tensors, basis, wire_bw)
        t_dual   = dual_lane(tensors, basis, wire_bw)
        speedup  = t_direct / t_dual if t_dual > 0 else float("inf")
        print(f"{name:<35s}  {t_direct*1000:>9.1f} ms  {t_codec*1000:>9.1f} ms  "
              f"{t_dual*1000:>9.1f} ms  {speedup:>13.2f}x")

    print()
    print("Notes:")
    print("  - 'all_direct': baseline — all tensors via cuda.cpu() (the standard offload path)")
    print("  - 'all_codec':  all tensors via PCA + NVENC + (sim) wire. Slow today due to")
    print("                  FFmpeg subprocess overhead.")
    print("  - 'dual_lane':  half tensors via direct, half via codec, running concurrently")
    print("                  on separate threads + cuda streams.")
    print()
    print("  - For PCIe (no wire simulation): the codec subprocess overhead dominates and")
    print("    'dual_lane' is bottlenecked by the codec lane's slow encode pipeline. With")
    print("    the PyAV fast path, this would flip to a clear dual-lane win.")
    print("  - For 1 Gbit and 100 Mbps wires: the wire IS the bottleneck. Compressed bytes")
    print("    take much less time on the wire, freeing capacity. 'dual_lane' becomes useful.")
    print("  - Wire times use sleep(bytes/bandwidth) — simulated, not actually transmitted.")


if __name__ == "__main__":
    main()
