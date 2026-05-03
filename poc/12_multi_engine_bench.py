"""12 — MultiEngineCodecSession: parallel NVENC across multiple GPU engines.

Modern NVIDIA GPUs ship with multiple NVENC encoder engines on the same die.
The RTX 5090 has 3, H100 has 4, A100 has 1. They run as independent hardware
lanes. This PoC distributes tensor encodes across them in parallel.

Stacks on top of CodecSession's ~1.77x speedup over subprocess for batch
workloads:

    Subprocess per-call:                ~302 ms/tensor   (baseline)
    PyAV per-call:                      ~243 ms/tensor   (1.24x)
    CodecSession:                       ~177 ms/tensor   (1.71x)
    MultiEngineCodecSession (3 engines):~111 ms/tensor   (2.72x)   <-- new

Note: this is a BATCH API. Pass N tensors, get back N (packets, recipe)
tuples. Streaming single-tensor encodes don't benefit (use CodecSession
or compress() instead).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from nvenc_compress import (
    build_shared_basis, compress, decompress, metrics,
    CodecSession, MultiEngineCodecSession,
)


DATA_DIR_DIFFUSION = Path("data/diffusion")
DATA_DIR_KV = Path("data/kv")
# Optional fallback: if the new public capture dirs are empty, look in
# RING0_DATA_DIR (used by the project author's internal research scratchpad).
import os as _os
RING0_FALLBACK = Path(_os.environ.get("RING0_DATA_DIR", "ring0/data"))
K = 1000
QP = 18
BATCH_SIZE = 12
N_ENGINES_TO_TEST = [1, 2, 3]


def find_samples():
    for d in (DATA_DIR_DIFFUSION, DATA_DIR_KV, RING0_FALLBACK):
        paths = sorted(d.glob("activation*.pt") if d != DATA_DIR_KV else d.glob("kv_*_K.pt"))
        out = []
        for p in paths[:32]:
            try:
                s = torch.load(p, map_location="cpu", weights_only=False)
                t = s["tensor"].to(torch.float32)
                if t.ndim == 3:   X = t.reshape(-1, t.shape[-1])
                elif t.ndim == 4: X = t[0].permute(1, 0, 2).reshape(t.shape[2], -1)
                elif t.ndim == 2: X = t
                else: continue
                out.append(X.cuda())
            except Exception:
                continue
        if len(out) >= 5:
            print(f"Using {len(out)} captures from {d}/")
            return out
    print("No captures found.")
    return []


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    samples = find_samples()
    if len(samples) < BATCH_SIZE + 4:
        print(f"Need >= {BATCH_SIZE + 4} samples; have {len(samples)}")
        return
    basis = build_shared_basis(samples[:4], K=K)
    test = samples[4 : 4 + BATCH_SIZE]
    print(f"Batch size: {BATCH_SIZE}, K={K}, QP={QP}\n")

    # Get frame size
    _data, _recipe = compress(test[0], basis, qp=QP, backend="pyav")
    ph, pw = _recipe.padded_h, _recipe.padded_w

    # Quality verification on one tensor (single-engine vs multi-engine should produce same quality)
    print("Quality verification (one held-out tensor through 3-engine session):")
    with MultiEngineCodecSession(height=ph, width=pw, qp=QP, n_engines=3) as multi:
        results = multi.compress_batch([test[0]], basis)
        recon_tensors = multi.decompress_batch(results, basis)
    recon = recon_tensors[0].cuda()
    m = metrics.compute(
        test[0].T.unsqueeze(0).unsqueeze(-1).cpu(),
        recon.T.unsqueeze(0).unsqueeze(-1).cpu(),
    )
    print(f"  multi-engine (1 tensor): cos={m['cos_mean']:.4f}\n")

    # Bench across engine counts
    print(f"Benchmark — {BATCH_SIZE} tensors:\n")
    print(f"{'config':<30s}  {'total':>10s}  {'per-tensor':>12s}  {'vs subprocess':>14s}")

    # Reference: subprocess per-call
    compress(test[0], basis, qp=QP, backend="subprocess")
    t0 = time.perf_counter()
    for X in test:
        compress(X, basis, qp=QP, backend="subprocess")
    torch.cuda.synchronize()
    t_sub = time.perf_counter() - t0
    print(f"{'subprocess (per-call)':<30s}  {t_sub*1000:>8.0f} ms  {t_sub/BATCH_SIZE*1000:>10.0f} ms  {1.00:>13.2f}x")

    # PyAV per-call
    compress(test[0], basis, qp=QP, backend="pyav")
    t0 = time.perf_counter()
    for X in test:
        compress(X, basis, qp=QP, backend="pyav")
    torch.cuda.synchronize()
    t_pyav = time.perf_counter() - t0
    print(f"{'PyAV (per-call)':<30s}  {t_pyav*1000:>8.0f} ms  {t_pyav/BATCH_SIZE*1000:>10.0f} ms  {t_sub/t_pyav:>13.2f}x")

    # CodecSession single
    with CodecSession(height=ph, width=pw, qp=QP) as session:
        t0 = time.perf_counter()
        for X in test:
            session.compress(X, basis)
        torch.cuda.synchronize()
        t_session = time.perf_counter() - t0
    print(f"{'CodecSession (1 engine)':<30s}  {t_session*1000:>8.0f} ms  {t_session/BATCH_SIZE*1000:>10.0f} ms  {t_sub/t_session:>13.2f}x")

    # MultiEngineCodecSession across N
    for n_eng in N_ENGINES_TO_TEST:
        if n_eng == 1:
            continue  # already covered
        with MultiEngineCodecSession(height=ph, width=pw, qp=QP, n_engines=n_eng) as multi:
            t0 = time.perf_counter()
            multi.compress_batch(test, basis)
            torch.cuda.synchronize()
            t_multi = time.perf_counter() - t0
        label = f"MultiEngine ({n_eng} engines)"
        print(f"{label:<30s}  {t_multi*1000:>8.0f} ms  {t_multi/BATCH_SIZE*1000:>10.0f} ms  {t_sub/t_multi:>13.2f}x")

    print()
    print("Notes:")
    print("  - The 5090 has 3 NVENC engines. H100 has 4. A100 has 1 (no parallelism win).")
    print("  - Speedup scales sub-linearly because (a) Python GIL contention, (b) PCA matmul")
    print("    on GPU contends with itself across threads, (c) small per-frame Python overhead.")
    print("  - For batch / multi-tensor workloads (model parallelism, KV-spill decode batches,")
    print("    distributed inference), MultiEngineCodecSession is the best backend we ship.")


if __name__ == "__main__":
    main()
