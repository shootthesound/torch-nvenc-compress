"""02 — Diffusion activation channel-covariance spectrum.

Loads activation_*.pt files captured by `scripts/capture_diffusion.py`,
combines them into one sample matrix, computes the channel covariance, and
prints the eigenvalue spectrum (cumulative variance fractions, effective rank).

A heavy-tailed spectrum (most variance in top-K << D channels) is what makes
PCA-then-codec compression work for these tensors. If the spectrum is flat,
PCA can't help — see docs/findings.md for what we measured on FLUX.2 Klein.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


DATA_DIR = Path("data/diffusion")


def main() -> None:
    paths = sorted(DATA_DIR.glob("activation_*.pt"))
    if not paths:
        print(f"No activation_*.pt found in {DATA_DIR}/. Run scripts/capture_diffusion.py first.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = []
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        t = s["tensor"].to(torch.float32)
        # diffusers FLUX block output is typically [B, T, D]. Flatten batch and
        # treat each token as a sample.
        if t.ndim == 3:
            X = t.reshape(-1, t.shape[-1])
        elif t.ndim == 2:
            X = t
        else:
            print(f"  WARNING: {p.name} has unexpected ndim {t.ndim}, shape {tuple(t.shape)}; skipping")
            continue
        samples.append(X)

    if not samples:
        return

    D = samples[0].shape[1]
    X = torch.cat(samples, dim=0).to(device)
    T_total, D_check = X.shape
    assert D_check == D
    print(f"Loaded {len(samples)} captures into combined sample matrix [{T_total}, {D}]\n")

    mean = X.mean(dim=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (T_total - 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0).cpu()
    total = eigvals.sum().item()

    print(f"  total variance:      {total:.4f}")
    print(f"  largest eigenvalue:  {eigvals[0].item():.4f}")
    print(f"  median eigenvalue:   {eigvals[D // 2].item():.6e}")
    print(f"  ratio max/median:    {eigvals[0].item() / max(eigvals[D // 2].item(), 1e-30):.2f}")

    p = eigvals / total
    ent = -(p * (p + 1e-30).log()).sum()
    eff_rank = ent.exp().item()
    print(f"  effective rank:      {eff_rank:.1f} of {D}  (concentration {D/eff_rank:.2f}x)\n")

    print("Cumulative variance held by top-K components:")
    cum = (eigvals.cumsum(0) / total)
    candidate_ks = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, D]
    for k in candidate_ks:
        if k <= D:
            print(f"  top {k:>5d} ({100*k/D:5.1f}%):  {100 * cum[k - 1].item():6.2f}%")

    print("\nIf the top 1-10% of channels hold >75% of variance, PCA + truncation will work.")
    print("Run `python poc/03_diffusion_pareto.py` for the full LOO Pareto sweep.")


if __name__ == "__main__":
    main()
