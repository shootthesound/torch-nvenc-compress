"""03 — Diffusion activation LOO PCA + truncation + codec sweep.

For each captured activation, build a shared PCA basis from the OTHER N-1
samples (leave-one-out, the honest generalisation test), compress + decompress
the held-out one through PCA + truncation + NVENC HEVC at multiple (K, QP)
operating points, measure quality.

Reference numbers we measured on FLUX.2 Klein 9B (different model than
FLUX.1-schnell that the diffusers capture script uses, so your numbers will
differ slightly — but the qualitative shape of the Pareto should match):

    K=2000  QP=10:   6.1x at cos 0.991  (lossless)
    K=1000  QP=10:  12.3x at cos 0.975
    K=500   QP=18:  37.3x at cos 0.943  (aggressive)
"""

from __future__ import annotations

from pathlib import Path

import torch

from nvenc_compress import (
    Basis,
    compress, decompress, metrics,
)
from nvenc_compress.pca import LeaveOneOutBasisBuilder


DATA_DIR = Path("data/diffusion")
KS = [500, 1000, 2000, 4096]
QPS = [10, 18, 26]


def main() -> None:
    paths = sorted(DATA_DIR.glob("activation_*.pt"))
    if not paths:
        print(f"No activation_*.pt found in {DATA_DIR}/. Run scripts/capture_diffusion.py first.")
        return
    if len(paths) < 3:
        print(f"Need >=3 captures for LOO; found {len(paths)}. "
              "Re-run scripts/capture_diffusion.py with --num-prompts >= 3.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = []
    names = []
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        t = s["tensor"].to(torch.float32)
        if t.ndim == 3:
            X = t.reshape(-1, t.shape[-1])
        elif t.ndim == 2:
            X = t
        else:
            print(f"  skipping {p.name}: unexpected ndim {t.ndim}")
            continue
        samples.append(X.to(device))
        names.append(p.stem)

    N = len(samples)
    D = samples[0].shape[1]
    KS_filt = [k for k in KS if k <= D]
    print(f"Loaded {N} captures, D={D}, LOO sweep across "
          f"{len(KS_filt)} K values x {len(QPS)} QP values\n")

    print("Precomputing per-sample covariances...")
    builder = LeaveOneOutBasisBuilder(samples)

    cells = {(K, QP): [] for K in KS_filt for QP in QPS}

    for i, (X, name) in enumerate(zip(samples, names)):
        # Build shared basis at the largest K once (full eigendecomp);
        # smaller K's just slice the columns.
        max_K = max(KS_filt)
        basis_full = builder.basis_excluding(i, max_K)
        bytes_orig_fp16 = X.numel() * 2
        for K in KS_filt:
            basis = Basis(mean=basis_full.mean, V_K=basis_full.V_K[:, :K].contiguous())
            for QP in QPS:
                data, recipe = compress(X, basis, qp=QP)
                X_recon = decompress(data, basis, recipe)
                m = metrics.compute(
                    X.T.unsqueeze(0).unsqueeze(-1).cpu(),
                    X_recon.T.unsqueeze(0).unsqueeze(-1).cpu(),
                )
                ratio = bytes_orig_fp16 / len(data)
                cells[(K, QP)].append((ratio, m))
        print(f"  hold-out {i+1}/{N}: {name}")

    print(f"\nLOO results (mean across {N} hold-outs):")
    print(f"{'K':>5s} {'QP':>3s}  {'mean_ratio':>10s}  "
          f"{'mean_cos':>9s}  {'min_cos':>8s}  {'mean_p1':>8s}  {'min_p1':>8s}")
    for K in KS_filt:
        for QP in QPS:
            results = cells[(K, QP)]
            ratios = [r for r, _ in results]
            cosm = [m["cos_mean"] for _, m in results]
            cosp1 = [m["cos_p1"] for _, m in results]
            print(f"{K:>5d} {QP:>3d}  {sum(ratios)/len(ratios):>9.2f}x  "
                  f"{sum(cosm)/len(cosm):>9.4f}  {min(cosm):>8.4f}  "
                  f"{sum(cosp1)/len(cosp1):>8.4f}  {min(cosp1):>8.4f}")


if __name__ == "__main__":
    main()
