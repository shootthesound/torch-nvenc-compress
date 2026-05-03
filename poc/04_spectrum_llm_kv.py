"""04 — LLM KV cache channel-covariance spectrum (K and V separately).

Loads kv_*_layer{L}_K.pt and kv_*_layer{L}_V.pt files captured by
`scripts/capture_llm_kv.py`, computes per-cache (K vs V) spectrum, prints
cumulative variance fractions and effective rank.

K and V have asymmetric structure: K cache is more concentrated than V cache
(measured on Mistral 7B v0.3: K eff_rank 86/1024, V eff_rank 148/1024).
At lossy operating points this means K reconstructs better than V.
"""

from __future__ import annotations

from pathlib import Path

import torch


DATA_DIR = Path("data/kv")


def analyze_kind(kind: str, paths: list[Path]) -> None:
    if not paths:
        print(f"\n  no {kind} cache files found")
        return
    print(f"\n=== {kind} cache spectrum ({len(paths)} samples) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xs = []
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        t = s["tensor"]                    # [1, num_kv_heads, seq_len, head_dim]
        # rearrange to [seq_len, num_kv_heads * head_dim]
        if t.ndim != 4:
            print(f"  skipping {p.name}: ndim {t.ndim}")
            continue
        X = t[0].permute(1, 0, 2).reshape(t.shape[2], -1)
        Xs.append(X)

    if not Xs:
        return
    D = Xs[0].shape[1]
    X = torch.cat(Xs, dim=0).to(device)
    T_total = X.shape[0]
    print(f"  combined sample matrix: [{T_total}, {D}]")

    mean = X.mean(dim=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (T_total - 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0).cpu()
    total = eigvals.sum().item()

    print(f"  largest eigval:    {eigvals[0].item():.4f}")
    print(f"  median eigval:     {eigvals[D // 2].item():.6e}")
    print(f"  ratio max/median:  {eigvals[0].item() / max(eigvals[D // 2].item(), 1e-30):.2f}")

    cum = (eigvals.cumsum(0) / total)
    print(f"  cumulative variance:")
    for k in [10, 25, 50, 100, 200, 400, 800, D]:
        if k <= D:
            print(f"    top {k:>4d} ({100*k/D:5.1f}%): {100 * cum[k - 1].item():6.2f}%")

    p = eigvals / total
    ent = -(p * (p + 1e-30).log()).sum()
    eff_rank = ent.exp().item()
    print(f"  effective rank:    {eff_rank:.1f} of {D}  (concentration {D/eff_rank:.2f}x)")


def main() -> None:
    if not DATA_DIR.exists():
        print(f"{DATA_DIR}/ does not exist. Run scripts/capture_llm_kv.py first.")
        return

    k_paths = sorted(DATA_DIR.glob("kv_*_K.pt"))
    v_paths = sorted(DATA_DIR.glob("kv_*_V.pt"))
    if not (k_paths or v_paths):
        print(f"No kv_*_K.pt or kv_*_V.pt files in {DATA_DIR}/. "
              "Run scripts/capture_llm_kv.py first.")
        return

    analyze_kind("K", k_paths)
    analyze_kind("V", v_paths)

    print("\nIf top ~10% of channels hold ~80% of variance for both K and V, "
          "PCA + truncation on KV will work — see poc/05_llm_kv_pareto.py.")


if __name__ == "__main__":
    main()
