"""22 — Long-horizon codec drift (the no-butterfly-effect check for ML).

Answers the question that comes after PoC 20's per-frame latency: *if the
codec sits inside a feedback loop where each step modifies the tensor
slightly (iterative inference, activation checkpointing inside training),
does codec quantization noise compound across many steps, or does it
stay bounded?*

This is the same shape as the long-horizon soak test the sibling vortex
repo runs on Navier-Stokes solver state — for ML data instead of fluid
fields. Mirrors that test's dual-path harness: two paths share the same
initial activation and the same per-step perturbation; one path runs
the codec round-trip on every step, the other doesn't. Per-step L2
between the paths is measured. If error stays bounded (q4/q1 ≈ 1×) the
codec is safe for in-loop use; if error grows exponentially, the codec
destabilises the trajectory.

Methodology:
  1. Synthesize a FLUX-shape activation (heavy-tailed channel covariance,
     ~25 MB, packed as [N, 3, H, W] uint8 YUV frames) — matches the
     scale + structure of the real captures used elsewhere in this repo.
  2. Per step:
       - Apply a small Gaussian perturbation in float space, requantize
         to uint8. Same perturbation seeded identically for both paths.
       - Path A (baseline): keep the perturbed uint8 frames as-is.
       - Path B (codec-in-loop): encode -> decode the same frames
         through DirectBackend; the decoded uint8 is what feeds the
         next step's perturbation.
  3. Per step, measure max abs |path_A - path_B| in uint8 units.
  4. Repeat for lossless / QP=10 / QP=18 / QP=28 over 1000 steps each.
  5. Report q4/q1 ratio (last-quarter-mean / first-quarter-mean of
     steady-state) — the "is error bounded?" headline.

Runtime: ~5 min on RTX 5090 for all 4 modes.

Saves docs/figures/long_horizon_codec_drift.png.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from nvenc_compress.direct.backend import DirectBackend


# ---- Configuration -------------------------------------------------
H, W = 256, 256
N_FRAMES = 32                 # 32 × 192 KB = ~6 MB activation per step
N_STEPS = 500                 # soak length (vortex's reference test was 5000;
                              # 500 with per-step codec round-trip is enough
                              # to see whether drift saturates or grows)
PERTURB_STD = 1.5             # uint8 units; small per-step modification
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = REPO_ROOT / "docs" / "figures" / "long_horizon_codec_drift.png"

MODES = [
    # (label, lossless, qp, plot_color)
    ("Lossless (bit-exact)",  True,  18, "#3b3b3b"),
    ("QP=10 (near-lossless)", False, 10, "#2c8a4f"),
    ("QP=18 (standard)",      False, 18, "#1f4ed8"),
    ("QP=28 (high compress)", False, 28, "#d8541f"),
]


def synth_flux_shaped_activation(seed: int = 0) -> np.ndarray:
    """Build a [N_FRAMES, 3, H, W] uint8 tensor that mimics a FLUX-style
    activation: heavy-tailed channel covariance + low-frequency spatial
    structure, the empirical pattern documented in the project log
    (top 100 of 4096 channels hold 75% of variance, etc.).

    Each Y/U/V plane gets a different scale to mimic the per-channel
    std range (0.26 to 12.6 in real captures); spatial content is a
    sum of a few 2D sinusoids per frame for low-rank structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = np.empty((N_FRAMES, 3, H, W), dtype=np.float32)
    # Per-channel variance: heavy-tailed (a few "outlier" channels)
    channel_scales = rng.lognormal(mean=0.0, sigma=1.5, size=(N_FRAMES, 3))
    for f in range(N_FRAMES):
        for c in range(3):
            field = np.zeros((H, W), dtype=np.float32)
            for _ in range(3):  # low-rank structure
                kx = rng.uniform(0.5, 3.0)
                ky = rng.uniform(0.5, 3.0)
                phase = rng.uniform(0, 2 * np.pi)
                field += rng.uniform(20, 100) * np.sin(
                    2 * np.pi * (kx * xx / W + ky * yy / H) + phase
                )
            field += rng.standard_normal((H, W)) * 8.0
            out[f, c] = field * channel_scales[f, c]
    out = (out - out.min()) / (out.max() - out.min())
    return (out * 255).astype(np.uint8)


def perturb_in_place(frames_float: np.ndarray, std: float, seed: int):
    """Add Gaussian perturbation (in float space) and clip to uint8 range.
    Deterministic per step via seed so both paths see the same noise."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(frames_float.shape).astype(np.float32) * std
    frames_float += noise
    np.clip(frames_float, 0, 255, out=frames_float)


def run_mode(label: str, lossless: bool, qp: int) -> dict:
    """Run the dual-path soak for one codec mode. Returns dict with
    per-step max-diff array + summary stats.

    Per-step harness (mirrors vortex's dual-path test):
      1. Apply identical Gaussian perturbation to both paths (same seed).
      2. Cast both paths to uint8 — IDENTICAL quantization step on both
         sides. This is the key trick: without matching quantization,
         path A drifts in float forever while path B gets snapped to
         uint8 every step via the codec, and they diverge for
         representation reasons rather than codec reasons.
      3. Path A keeps its uint8 directly. Path B encodes -> decodes
         through the codec — for lossless this is bit-exact; for lossy
         this injects per-step codec quantization noise.
      4. Both paths' uint8 state becomes input to next step's float
         perturbation. Lossless mode: path_a uint8 == path_b uint8 at
         every step (drift = 0). Lossy mode: drift starts at codec-
         noise floor and either saturates (good) or grows (bad).
    """
    print(f"\n[{label}]")
    initial = synth_flux_shaped_activation(seed=0)
    print(f"  shape: {initial.shape}  raw bytes: {initial.nbytes/1e6:.1f} MB/step")

    # Two paths, both start identical (uint8)
    state_a = initial.copy()
    state_b = initial.copy()

    backend = DirectBackend(height=H, width=W, qp=qp, lossless=lossless)
    diffs = np.empty(N_STEPS, dtype=np.float32)
    t0 = time.perf_counter()
    last_print = [0]
    try:
        for step in range(N_STEPS):
            # Apply identical Gaussian perturbation to both paths
            float_a = state_a.astype(np.float32)
            float_b = state_b.astype(np.float32)
            perturb_in_place(float_a, std=PERTURB_STD, seed=step)
            perturb_in_place(float_b, std=PERTURB_STD, seed=step)
            # Identical quantization on both sides — this is what makes
            # the dual-path test fair (only the codec round-trip differs)
            state_a = float_a.astype(np.uint8)
            state_b = float_b.astype(np.uint8)

            # Codec round-trip on path B only
            packets = backend.encode_frames(state_b)
            decoded = backend.decode_frames(packets, N_FRAMES)
            state_b = decoded  # decoded uint8 feeds next step's perturbation

            # Per-step drift: max abs diff between the two paths in uint8 units
            diffs[step] = float(
                np.abs(state_a.astype(int) - state_b.astype(int)).max()
            )

            if step - last_print[0] >= 100 or step == N_STEPS - 1:
                last_print[0] = step
                elapsed = time.perf_counter() - t0
                rate = (step + 1) / max(elapsed, 1e-6)
                eta = (N_STEPS - step - 1) / max(rate, 1e-6)
                print(f"    step {step+1:>4}/{N_STEPS}  diff={diffs[step]:>5.1f}  "
                      f"({rate:.1f} steps/s, ETA {eta:.0f}s)")
    finally:
        backend.close()

    elapsed = time.perf_counter() - t0
    # Steady-state region: drop first 10% as transient
    skip = max(1, N_STEPS // 10)
    ss = diffs[skip:]
    quarter = max(1, len(ss) // 4)
    q1 = float(ss[:quarter].mean())
    q4 = float(ss[-quarter:].mean())
    growth = q4 / max(q1, 1e-9)

    print(f"  steady-state q1 mean: {q1:.3f}  q4 mean: {q4:.3f}  q4/q1: {growth:.2f}x")
    print(f"  done in {elapsed:.0f}s")
    return {
        "label": label,
        "diffs": diffs,
        "q1": q1,
        "q4": q4,
        "growth": growth,
        "max_overall": float(diffs.max()),
    }


def main() -> int:
    print("=" * 72)
    print("Long-horizon codec drift — dual-path soak on FLUX-shape activation")
    print("=" * 72)
    print(f"  steps: {N_STEPS}")
    print(f"  per-step perturbation: Gaussian std={PERTURB_STD} (uint8 units)")
    print(f"  activation shape: [{N_FRAMES}, 3, {H}, {W}] uint8 = "
          f"{N_FRAMES*3*H*W/1e6:.1f} MB/step")

    results = []
    for label, lossless, qp, _color in MODES:
        results.append(run_mode(label, lossless, qp))

    # ---- Plot ---------------------------------------------------------
    print("\nGenerating figure ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140, constrained_layout=True)
    for r, (label, _l, _q, color) in zip(results, MODES):
        x = np.arange(N_STEPS)
        # log-y; floor zeros at 1e-2 so lossless (always 0) renders
        y = np.maximum(r["diffs"], 1e-2)
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.85,
                 label=f"{label}  q4/q1 = {r['growth']:.2f}x")

    ax.set_yscale("log")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("per-step max abs |path_A - path_B|  (uint8 units)")
    ax.set_title(
        f"Long-horizon codec drift on FLUX-shape activations "
        f"({N_STEPS} steps with per-step perturbation σ={PERTURB_STD})\n"
        "Bounded traces (q4/q1 < ~3) = codec safe for in-loop use; "
        "lossless is bit-exact (rendered at floor 0.01)",
        fontsize=11,
    )
    ax.axvspan(N_STEPS // 10, N_STEPS, alpha=0.06, color="gray")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_FIG}")

    # ---- Summary table -----------------------------------------------
    print()
    print("=" * 72)
    print(f"Summary — {N_STEPS}-step soak, steady-state region (steps {N_STEPS//10}–{N_STEPS})")
    print("=" * 72)
    print(f"{'mode':<26s}  {'q1 mean':>10s}  {'q4 mean':>10s}  {'q4/q1':>8s}  "
          f"{'max overall':>12s}")
    print("-" * 72)
    for r in results:
        print(f"{r['label']:<26s}  {r['q1']:>10.3f}  {r['q4']:>10.3f}  "
              f"{r['growth']:>7.2f}x  {r['max_overall']:>12.1f}")
    print()
    print("Reading: q4/q1 << ~3x means error is bounded across many steps —")
    print("codec safe for in-loop use (activation checkpointing, iterative")
    print("inference, etc.). Exponential drift would show q4/q1 in the 100s.")
    print("Lossless is bit-exact across the whole soak (q4/q1 == 1.0 trivially).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
