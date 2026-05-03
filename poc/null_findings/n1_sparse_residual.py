"""n1 — Sparse residual: doesn't help much.

Hypothesis: take a lossy codec output, identify the few positions with the
largest reconstruction error, transmit only those positions as sparse
corrections. Should cheaply close the gap to lossless.

Reality: reconstruction error is approximately uniformly distributed across
positions. Top-1% positions hold only slightly more error than random 1%.
Even sending 10% of positions back as fp16 (4-byte index + 2-byte value =
~10 MB extra for a 32 MB tensor) only improves cos by ~0.02 because most
of the unfixed error is spread thinly across the remaining 90% of positions.

Reference numbers (LOO across N=8, K=500, QP=18, FLUX.2 Klein activations):
    0.0% residual:  37.3x ratio, cos 0.943
    1.0% residual:  17.6x ratio, cos 0.947  (tiny improvement, big ratio cost)
    10%  residual:   3.1x ratio, cos 0.966  (still doesn't reach 0.99)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from nvenc_compress import Basis, build_shared_basis, compress, decompress, metrics


# Use whatever captures we have — diffusion preferred for the wider tensor.
DATA_DIRS = [Path("data/diffusion"), Path("data/kv")]
PERCENTAGES = [0.0, 0.1, 1.0, 5.0, 10.0]
BYTES_PER_CORRECTION = 4 + 2                   # uint32 idx + fp16 value


def find_samples() -> list[torch.Tensor]:
    for d in DATA_DIRS:
        paths = sorted(d.glob("*.pt"))
        samples = []
        for p in paths:
            try:
                s = torch.load(p, map_location="cpu", weights_only=False)
                t = s["tensor"].to(torch.float32)
                if t.ndim == 3:
                    X = t.reshape(-1, t.shape[-1])
                elif t.ndim == 4:                  # KV tensor
                    X = t[0].permute(1, 0, 2).reshape(t.shape[2], -1)
                elif t.ndim == 2:
                    X = t
                else:
                    continue
                samples.append(X)
            except Exception:
                continue
        if len(samples) >= 3:
            print(f"Using {len(samples)} samples from {d}/")
            return samples
    return []


def apply_sparse_residual(X_recon, residual, percentage):
    """Pick top-p% positions by |residual|, transmit at fp16 precision, apply."""
    if percentage <= 0:
        return X_recon, 0
    flat = residual.reshape(-1)
    n_total = flat.numel()
    n_keep = max(1, int(n_total * percentage / 100.0))
    abs_flat = flat.abs()
    _vals, idx = torch.topk(abs_flat, n_keep)
    transmitted = flat[idx].to(torch.float16).to(torch.float32)
    correction = torch.zeros_like(flat)
    correction[idx] = transmitted
    return X_recon + correction.reshape(X_recon.shape), n_keep


def main() -> None:
    samples = find_samples()
    if not samples:
        print("No captures found. Run a capture script first (scripts/capture_diffusion.py "
              "or scripts/capture_llm_kv.py).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = [s.to(device) for s in samples]
    D = samples[0].shape[1]
    K = max(50, min(500, D // 4))                # arbitrary lossy operating point
    QP = 18
    print(f"D={D}, using K={K}, QP={QP}\n")

    rows = []
    for i, X in enumerate(samples):
        train = samples[:i] + samples[i+1:]
        basis = build_shared_basis(train, K=K)
        bytes_orig_fp16 = X.numel() * 2

        data, recipe = compress(X, basis, qp=QP)
        X_recon = decompress(data, basis, recipe).to(device)
        bytes_codec = len(data)
        residual = X - X_recon

        for p in PERCENTAGES:
            X_corrected, n_corr = apply_sparse_residual(X_recon, residual, p)
            bytes_residual = n_corr * BYTES_PER_CORRECTION
            ratio = bytes_orig_fp16 / (bytes_codec + bytes_residual)
            m = metrics.compute(
                X.T.unsqueeze(0).unsqueeze(-1).cpu(),
                X_corrected.T.unsqueeze(0).unsqueeze(-1).cpu(),
            )
            rows.append((p, ratio, m["cos_mean"], m["cos_p1"], n_corr))

    print(f"{'pct':>6s}  {'mean_ratio':>10s}  {'mean_cos':>9s}  {'mean_p1':>9s}  {'mean_corrs':>11s}")
    for p in PERCENTAGES:
        cell = [r for r in rows if r[0] == p]
        if not cell:
            continue
        print(f"{p:>5.2f}%  "
              f"{sum(r[1] for r in cell)/len(cell):>9.2f}x  "
              f"{sum(r[2] for r in cell)/len(cell):>9.4f}  "
              f"{sum(r[3] for r in cell)/len(cell):>9.4f}  "
              f"{sum(r[4] for r in cell)/len(cell):>11,.0f}")

    print("\nVerdict: residual is uniformly distributed; sparse correction barely")
    print("improves cos. Don't ship sparse residual as a default mode for this content.")


if __name__ == "__main__":
    main()
