"""n3 — Channel reordering for HEVC P-frame coherence: doesn't help.

Hypothesis: HEVC's GOP structure uses P-frames that predict from the
previous frame. If we order PCA-rotated channels by similarity (so adjacent
frames hold similar channels), the codec's temporal prediction has more to
work with and compression should improve.

Reality: PCA orthogonalises by construction. The whole point of PCA is to
produce decorrelated dimensions. Greedy nearest-neighbour reordering finds
residual non-linear correlations between channels, but they're too weak for
the codec's linear prediction model to exploit.

Reference numbers (LOO across N=64 FLUX.2 Klein activations, K=500, QP=18):
    PCA-eigenvalue order:    37.25x  cos 0.943   (baseline)
    Greedy similarity order: 37.04x  cos 0.942   (slightly worse, within noise)

The two are within measurement noise. Channel ordering is not a meaningful
lever for this pipeline.

Why it might still be a lever in a different setup: if you're NOT using PCA
(so channels in the standard basis DO have cross-channel correlations), or
if you use a different basis that intentionally leaves correlations in,
reordering might help. We didn't test those.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nvenc_compress import Basis, build_shared_basis, compress, decompress, metrics


DATA_DIRS = [Path("data/diffusion"), Path("data/kv")]
QP = 18


def find_samples():
    for d in DATA_DIRS:
        paths = sorted(d.glob("*.pt"))
        samples = []
        for p in paths:
            try:
                s = torch.load(p, map_location="cpu", weights_only=False)
                t = s["tensor"].to(torch.float32)
                if t.ndim == 3:
                    X = t.reshape(-1, t.shape[-1])
                elif t.ndim == 4:
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


def greedy_channel_order(R: torch.Tensor) -> torch.Tensor:
    """R: [T, K]. Returns permutation [K] that orders channels by greedy
    nearest-neighbour cosine similarity from channel 0."""
    chans = R.T
    norms = chans.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_sim = (chans / norms) @ (chans / norms).T
    K_ = cos_sim.shape[0]
    visited = torch.zeros(K_, dtype=torch.bool, device=R.device)
    order = torch.empty(K_, dtype=torch.long, device=R.device)
    order[0] = 0
    visited[0] = True
    for i in range(1, K_):
        prev = int(order[i - 1].item())
        sims = cos_sim[prev].clone()
        sims[visited] = -float("inf")
        nxt = int(sims.argmax().item())
        order[i] = nxt
        visited[nxt] = True
    return order


def main() -> None:
    samples = find_samples()
    if not samples:
        print("No captures found. Run a capture script first.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = [s.to(device) for s in samples]
    D = samples[0].shape[1]
    K = max(50, min(500, D // 4))
    print(f"D={D}, K={K}, QP={QP}\n")

    baseline_results = []
    greedy_results = []

    for i, X in enumerate(samples):
        train = samples[:i] + samples[i+1:]
        basis = build_shared_basis(train, K=K)
        bytes_orig_fp16 = X.numel() * 2

        # Baseline: PCA-eigenvalue order (the default in our pipeline)
        data, recipe = compress(X, basis, qp=QP)
        X_recon = decompress(data, basis, recipe).to(device)
        m = metrics.compute(
            X.T.unsqueeze(0).unsqueeze(-1).cpu(),
            X_recon.T.unsqueeze(0).unsqueeze(-1).cpu(),
        )
        baseline_results.append((bytes_orig_fp16 / len(data), m["cos_mean"]))

        # Greedy reordering: permute the basis columns
        R = basis.project(X)
        order = greedy_channel_order(R)
        permuted_basis = Basis(
            mean=basis.mean,
            V_K=basis.V_K[:, order].contiguous(),
        )
        data2, recipe2 = compress(X, permuted_basis, qp=QP)
        X_recon2 = decompress(data2, permuted_basis, recipe2).to(device)
        m2 = metrics.compute(
            X.T.unsqueeze(0).unsqueeze(-1).cpu(),
            X_recon2.T.unsqueeze(0).unsqueeze(-1).cpu(),
        )
        # The order itself must be transmitted; account for it (~K bytes at uint16)
        order_bytes = K * 2
        greedy_results.append((bytes_orig_fp16 / (len(data2) + order_bytes), m2["cos_mean"]))

    print(f"{'condition':<22s}  {'mean_ratio':>10s}  {'mean_cos':>9s}")
    for label, results in (("baseline (PCA order)", baseline_results),
                            ("greedy similarity",     greedy_results)):
        ratios = [r for r, _ in results]
        cosms = [c for _, c in results]
        print(f"{label:<22s}  {sum(ratios)/len(ratios):>9.2f}x  {sum(cosms)/len(cosms):>9.4f}")

    print("\nVerdict: PCA already removed linear correlations. The codec's P-frame")
    print("prediction has nothing to exploit. Channel reordering doesn't help.")


if __name__ == "__main__":
    main()
