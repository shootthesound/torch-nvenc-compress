"""05 — LLM KV cache LOO PCA + truncation + codec sweep.

Same shape of analysis as poc/03 but for KV cache. Builds a shared PCA basis
per kind (K, V) via leave-one-out, runs through the full pipeline at multiple
(K_keep, QP) operating points, reports per-cell quality.

Reference numbers measured on Mistral 7B v0.3 (1024 KV channels per layer):

    K_keep=1024  QP=10:   2.7x at cos 0.999  (lossless)
    K_keep=800   QP=18:   5.1x at cos 0.986 (K) / 0.965 (V)
    K_keep=400   QP=18:    10x at cos 0.954 (K) / 0.870 (V)

Note: KV caches are LESS compressible than diffusion activations at the
same quality level. For LLM inference, stay near-lossless — KV errors compound
across the decode loop in a way diffusion-step errors do not.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nvenc_compress import Basis, compress, decompress, metrics
from nvenc_compress.pca import LeaveOneOutBasisBuilder


DATA_DIR = Path("data/kv")
KS = [200, 400, 800, 1024]
QPS = [10, 18]


def run_kind(kind: str, paths: list[Path]) -> None:
    if not paths or len(paths) < 3:
        print(f"\nSkipping {kind} cache: need >=3 captures, have {len(paths)}")
        return

    print(f"\n=== {kind} cache LOO sweep ({len(paths)} hold-outs) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = []
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        t = s["tensor"]                                        # [1, kv_heads, seq, head_dim]
        X = t[0].permute(1, 0, 2).reshape(t.shape[2], -1).to(device)
        samples.append(X)

    N = len(samples)
    D = samples[0].shape[1]
    KS_filt = [k for k in KS if k <= D]
    print(f"D={D}, sweep K in {KS_filt} x QP in {QPS}")

    builder = LeaveOneOutBasisBuilder(samples)
    cells = {(K, QP): [] for K in KS_filt for QP in QPS}

    for i, X in enumerate(samples):
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

    print(f"\n{'K':>5s} {'QP':>3s}  {'mean_ratio':>10s}  "
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


def main() -> None:
    if not DATA_DIR.exists():
        print(f"{DATA_DIR}/ does not exist. Run scripts/capture_llm_kv.py first.")
        return

    run_kind("K", sorted(DATA_DIR.glob("kv_*_K.pt")))
    run_kind("V", sorted(DATA_DIR.glob("kv_*_V.pt")))


if __name__ == "__main__":
    main()
