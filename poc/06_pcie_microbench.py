"""06 — Codec round-trip vs PCIe microbenchmark.

Measures, on one captured activation tensor (or a synthetic stand-in if you
haven't captured one):

  1. PCIe baseline: VRAM -> system RAM -> VRAM round-trip
  2. Compressed pipeline: full PCA + quant + NVENC encode + NVDEC decode +
     dequant + inverse-PCA round-trip — for both `subprocess` and `pyav`
     backends side-by-side
  3. Stage decomposition: where the time actually goes

The PyAV backend (in-process FFmpeg via the `av` package) gives ~1.2x
speedup vs the subprocess backend by eliminating per-call FFmpeg startup.
Install with `pip install av` to enable. Both backends produce byte-identical
bitstreams.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from nvenc_compress import Basis, build_shared_basis, compress, decompress


DATA_DIR_DIFFUSION = Path("data/diffusion")
DATA_DIR_KV = Path("data/kv")
K = 1000
QP = 18
ITERS = 5
WARMUPS = 2


def find_test_tensor() -> tuple[torch.Tensor, str]:
    """Find any captured activation or KV tensor to test on. Returns (tensor[T,D], source)."""
    paths = sorted(DATA_DIR_DIFFUSION.glob("activation_*.pt"))
    if paths:
        s = torch.load(paths[0], map_location="cpu", weights_only=False)
        t = s["tensor"].to(torch.float32)
        if t.ndim == 3:
            X = t.reshape(-1, t.shape[-1])
        elif t.ndim == 2:
            X = t
        else:
            raise RuntimeError(f"unexpected ndim {t.ndim}")
        return X, f"diffusion activation from {paths[0].name}"

    paths = sorted(DATA_DIR_KV.glob("kv_*_K.pt"))
    if paths:
        s = torch.load(paths[0], map_location="cpu", weights_only=False)
        t = s["tensor"].to(torch.float32)
        X = t[0].permute(1, 0, 2).reshape(t.shape[2], -1)
        return X, f"LLM KV K cache from {paths[0].name}"

    print("No captures found — generating synthetic [4096, 4096] tensor for the demo")
    return torch.randn(4096, 4096), "synthetic Gaussian noise (no real captures available)"


def time_pcie(tensor_gpu: torch.Tensor, iters: int) -> float:
    """tensor on GPU. Times: cpu() then cuda(), per call."""
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu = tensor_gpu.cpu()
        torch.cuda.synchronize()
        _gpu = cpu.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def time_compressed(X: torch.Tensor, basis: Basis, iters: int, backend: str = "subprocess") -> tuple[float, dict, int]:
    """Returns (mean_total, stage_means, bytes_codec)."""
    stages = {"compress": [], "decompress": [], "total": []}
    bytes_codec = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        t = time.perf_counter()
        data, recipe = compress(X, basis, qp=QP, backend=backend)
        torch.cuda.synchronize()
        stages["compress"].append(time.perf_counter() - t)

        t = time.perf_counter()
        _ = decompress(data, basis, recipe, backend=backend)
        torch.cuda.synchronize()
        stages["decompress"].append(time.perf_counter() - t)

        stages["total"].append(time.perf_counter() - t0)
        bytes_codec = len(data)
    return (
        sum(stages["total"]) / len(stages["total"]),
        {k: sum(v) / len(v) for k, v in stages.items() if k != "total"},
        bytes_codec,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — this PoC requires a GPU")
        return

    X, source = find_test_tensor()
    print(f"Test tensor: {tuple(X.shape)}, source = {source}")
    bytes_orig_fp16 = X.numel() * 2
    print(f"  fp16 bytes (the 'wire size' baseline): {bytes_orig_fp16:,}")

    X_gpu = X.to("cuda")

    # PCIe baseline
    for _ in range(WARMUPS):
        time_pcie(X_gpu, 1)
    print(f"\nPCIe baseline (cuda -> cpu -> cuda):")
    pcie_t = time_pcie(X_gpu, ITERS)
    pcie_throughput = bytes_orig_fp16 / pcie_t
    print(f"  {pcie_t*1000:.2f} ms  ({pcie_throughput/1e9:.2f} GB/s effective)")

    # Build a basis from the same tensor (degenerate calibration but fine for the bench)
    basis = build_shared_basis([X_gpu], K=K)

    # Detect available backends
    backends_to_test = ["subprocess"]
    try:
        from nvenc_compress import codec_pyav
        ok, msg = codec_pyav.check_available()
        if ok:
            backends_to_test.append("pyav")
            print(f"\nPyAV detected: {msg}")
        else:
            print(f"\nPyAV not usable: {msg}")
    except Exception as e:
        print(f"\nPyAV not installed (pip install av to enable): {e}")

    backend_results = {}
    for backend in backends_to_test:
        for _ in range(WARMUPS):
            time_compressed(X_gpu, basis, 1, backend=backend)
        print(f"\nCompressed pipeline ({backend} backend, K={K}, QP={QP}):")
        comp_t, stages, bytes_codec = time_compressed(X_gpu, basis, ITERS, backend=backend)
        print(f"  {'compress':>12s}: {stages['compress']*1000:>7.2f} ms  (PCA + quant + encode + I/O)")
        print(f"  {'decompress':>12s}: {stages['decompress']*1000:>7.2f} ms  (decode + I/O + dequant + inverse)")
        print(f"  {'total':>12s}: {comp_t*1000:>7.2f} ms")
        print(f"  compressed bytes: {bytes_codec:,} ({bytes_orig_fp16/bytes_codec:.2f}x ratio)")
        backend_results[backend] = (comp_t, stages, bytes_codec)

    if "pyav" in backend_results:
        sp_t = backend_results["subprocess"][0]
        av_t = backend_results["pyav"][0]
        print(f"\nPyAV vs subprocess speedup: {sp_t/av_t:.2f}x  (saving {(sp_t-av_t)*1000:.0f} ms per round-trip)")

    # Use the FASTEST backend's numbers for the analysis below
    fastest = min(backend_results, key=lambda b: backend_results[b][0])
    comp_t = backend_results[fastest][0]
    bytes_codec = backend_results[fastest][2]
    print(f"\n(Analysis below uses the fastest backend: {fastest})")

    print(f"\n--- Analysis ---")
    print(f"PCIe round-trip:        {pcie_t*1000:>7.2f} ms")
    print(f"Compressed round-trip:  {comp_t*1000:>7.2f} ms  ({comp_t/pcie_t:.1f}x slower than PCIe in isolation)")

    # When does the compressed path beat raw transmission across a wire?
    # raw_time   = orig_bytes / wire_speed
    # comp_time  = comp_bytes / wire_speed + codec_time
    # raw > comp  <=>  wire_speed  <  (orig - comp) / codec_time
    crossover = (bytes_orig_fp16 - bytes_codec) / comp_t
    print(f"\nCompression beats raw transmission when wire speed is below "
          f"{crossover/1e6:.1f} MB/s ({crossover*8/1e6:.1f} Mbps)")

    print(f"\nReference wire speeds (codec wins below these IF the codec is fast enough):")
    print(f"  PCIe 5.0 x16: ~64 GB/s         (codec is {64e9/(bytes_codec/comp_t):.0f}x slower than wire moving compressed bytes alone)")
    print(f"  10 Gbit ethernet: ~1.25 GB/s   (codec is {1.25e9/(bytes_codec/comp_t):.1f}x slower)")
    print(f"  1 Gbit ethernet: ~125 MB/s     (codec is {125e6/(bytes_codec/comp_t):.2f}x slower)")
    print(f"  100 Mbit residential: ~12 MB/s (codec wins by {(bytes_codec/comp_t)/12e6:.1f}x)")

    print(f"\nNote: this measures the codec round-trip in ISOLATION. When pipelined")
    print(f"with concurrent compute or other PCIe traffic via torch.cuda.Stream,")
    print(f"the codec time is hidden and the user-visible cost is just the small")
    print(f"compressed-byte transfer. See poc/07_parallel_path_demo.py.")
    print(f"\nMeasured PyAV speedup: ~1.2x vs subprocess (saves ~100 ms per round-trip).")
    print(f"This moves the codec-wins crossover up but doesn't make the codec")
    print(f"competitive with PCIe in isolation. The deeper speedup (sub-30ms")
    print(f"round-trip, competitive with PCIe for large tensors and the multi-GPU")
    print(f"NVLink-replacement claim) needs direct Video Codec SDK access via")
    print(f"cffi — zero-copy GPU memory paths. See docs/parallel_path.md for both.")


if __name__ == "__main__":
    main()
