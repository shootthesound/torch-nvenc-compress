"""21 — Von Kármán vortex street: NVENC's P-frame chain on advection-dominated CFD.

`poc/20_heat_equation_pframe.py` showed NVENC's P-frame chain extracts ~24× extra
compression beyond intra-only HEVC for a 1000-step heat-equation trajectory. But
heat is a *soft* test — the field just diffuses smoothly with no actual motion.
A real video codec's motion-estimation engine has nothing to track in pure
diffusion; it just sees the field shrinking uniformly.

This PoC pushes the temporal-compression argument onto a *hard* test: a 2D Von
Kármán vortex street (incompressible flow past a cylinder, Reynolds ~150). The
field is genuinely *moving* — vortices shed off the cylinder and translate
downstream, exactly the pattern HEVC's motion estimation was designed for. But
the field also evolves nonlinearly: vortices are *created* and *destroyed* at
the cylinder, and they rotate, so motion estimation has to handle pattern
changes, not just translations.

If the P-frame chain still gives a strong win on this (5×+) it validates the
claim for the CFD / aerospace / weather audiences. If the win is small or
negative we've learned that motion estimation is calibrated for natural-video
patterns and doesn't transfer to small/fast/rotating structures — which is
itself a useful finding.

Solver: Stam "Stable Fluids" semi-Lagrangian incompressible Navier-Stokes
implemented in pure PyTorch on GPU. ~300 lines. Same comparison structure as
poc/20: Mode A (I-frames only) vs Mode B (I + P-frame chain).

Out of scope (future work):
  - Extracting + visualising the codec's actual motion vectors (requires
    parsing HEVC NAL units below the API)
  - 3D Navier-Stokes (2D suffices to validate the temporal claim)
  - Comparison vs domain-specific scientific compressors (SZ, ZFP, MGARD)
  - CFL-adaptive timestepping
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nvenc_compress.direct.backend import DirectBackend


# ---- simulation params ----------------------------------------------------
W_GRID = 512                  # spatial grid width (downstream direction)
H_GRID = 256                  # spatial grid height (cross-stream direction)
N_STEPS = 1500                # ~12 vortex-shedding cycles at our params
CYL_DIAMETER = 24             # cylinder diameter (grid cells) — smaller helps shed faster
CYL_CENTER = (128, H_GRID // 2 + 1)  # off-axis by 1 cell — breaks reflection symmetry
U_INFLOW = 1.0                # inflow velocity (left → right)
RE = 200.0                    # Reynolds number (well into shedding regime)
NU = U_INFLOW * CYL_DIAMETER / RE   # kinematic viscosity from Re
# dt chosen so α_diffuse = ν·dt/dx² ≈ 0.05 (mild diffusion per step) AND
# Courant u·dt/dx ≈ 0.4 (semi-Lagrangian is stable for any Courant, but
# bilinear interpolation introduces numerical viscosity ∝ Courant — keep small).
DT = 0.4
DX = 1.0
N_DIFFUSE_ITER = 20           # Jacobi iterations for implicit viscous diffusion
N_PRESSURE_ITER = 50          # Jacobi iterations for pressure projection
PERTURB_STEPS = 60            # number of initial steps to apply explicit asymmetric forcing
PERTURB_AMP = 0.5             # transverse-velocity kick amplitude in perturb region

# ---- codec params ---------------------------------------------------------
QP = 18                       # near-lossless

# ---- output paths --------------------------------------------------------
FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STRIP_PNG = FIG_DIR / "vortex_street_strip.png"
BYTES_PNG = FIG_DIR / "vortex_pframe_bytes_per_step.png"
RECON_PNG = FIG_DIR / "vortex_street_recon_vs_original.png"


def make_obstacle_mask(device) -> torch.Tensor:
    """Return [H, W] bool mask where True means 'inside the cylinder'."""
    yy, xx = torch.meshgrid(
        torch.arange(H_GRID, device=device, dtype=torch.float32),
        torch.arange(W_GRID, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = CYL_DIAMETER / 2.0
    cx, cy = CYL_CENTER
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def make_perturbed_inflow(device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initial velocity field plus a perturbation-mask tensor used by the
    main loop for `PERTURB_STEPS` steps to break wake symmetry.

    Returns (u, v, perturb_v_field). The perturb field is a vorticity-dipole-
    shaped transverse-velocity pattern just behind the cylinder: positive
    above, negative below. Adding a fraction of this to v each step for the
    first ~60 steps is the standard "kick the wake" trick to bypass the
    long transient before naturally-occurring shedding develops at moderate Re.
    """
    u = torch.full((H_GRID, W_GRID), U_INFLOW, device=device, dtype=torch.float32)
    v = torch.zeros((H_GRID, W_GRID), device=device, dtype=torch.float32)

    cx, cy = CYL_CENTER
    yy, xx = torch.meshgrid(
        torch.arange(H_GRID, device=device, dtype=torch.float32),
        torch.arange(W_GRID, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Dipole pattern: positive v above the cylinder centerline, negative below,
    # localised within a few diameters downstream of the cylinder.
    radial = torch.exp(
        -((xx - (cx + 1.5 * CYL_DIAMETER)) ** 2 + (yy - cy) ** 2)
        / (CYL_DIAMETER ** 2)
    )
    perturb_v = PERTURB_AMP * radial * torch.sign(yy - cy)
    return u, v, perturb_v


# 2D field shift helpers — replicate-edge boundary via index slicing.
# Names follow the convention "shift in the +X / -X / +Y / -Y direction":
#   shift_xp(f)[i, j]  ≈  f[i, j+1]  (neighbour one step in the +X / right direction)
def shift_xp(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f[:, 1:], f[:, -1:]], dim=1)
def shift_xm(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f[:, :1], f[:, :-1]], dim=1)
def shift_yp(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f[1:, :], f[-1:, :]], dim=0)
def shift_ym(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f[:1, :], f[:-1, :]], dim=0)


def apply_bc(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place boundary conditions:
       - left:  inflow (u=U, v=0)
       - right: zero-gradient outflow
       - top/bottom: free-slip (∂u/∂y=0, v=0)
       - cylinder: no-slip (u=v=0)"""
    u[:, 0] = U_INFLOW
    v[:, 0] = 0.0
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[0, :] = 0.0
    v[-1, :] = 0.0
    u = torch.where(mask, torch.zeros_like(u), u)
    v = torch.where(mask, torch.zeros_like(v), v)
    return u, v


def advect_semilagrangian(field: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Semi-Lagrangian advection of `field` by velocity (u, v).
    Backtraces each grid cell by -dt*velocity, samples old field via
    grid_sample (bilinear + edge clamp). Unconditionally stable."""
    H, W = field.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=field.device, dtype=torch.float32),
        torch.arange(W, device=field.device, dtype=torch.float32),
        indexing="ij",
    )
    # Backtrace
    src_x = xx - DT * u / DX
    src_y = yy - DT * v / DX
    # Normalise to [-1, 1] for grid_sample's expected coord system
    norm_x = 2.0 * src_x / (W - 1) - 1.0
    norm_y = 2.0 * src_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)   # [1, H, W, 2]
    field_in = field.unsqueeze(0).unsqueeze(0)                   # [1, 1, H, W]
    out = F.grid_sample(field_in, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
    return out.squeeze(0).squeeze(0)


def diffuse_jacobi(field: torch.Tensor, alpha: float, n_iter: int) -> torch.Tensor:
    """Implicit viscous diffusion: solve (I - α∇²) field_new = field via Jacobi.
    α = ν * dt / dx²."""
    f = field.clone()
    for _ in range(n_iter):
        f = (field + alpha * (shift_xp(f) + shift_xm(f) + shift_yp(f) + shift_ym(f))) \
            / (1.0 + 4.0 * alpha)
    return f


def project_incompressible(u: torch.Tensor, v: torch.Tensor, n_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pressure projection: enforce ∇·u = 0.
    Solve ∇²p = (1/dt) ∇·u via Jacobi, then subtract dt*∇p from velocity."""
    div = ((shift_xp(u) - shift_xm(u)) + (shift_yp(v) - shift_ym(v))) / (2 * DX)
    p = torch.zeros_like(u)
    rhs = div * (DX ** 2) / DT
    for _ in range(n_iter):
        p = (shift_xp(p) + shift_xm(p) + shift_yp(p) + shift_ym(p) - rhs) / 4.0
    dp_dx = (shift_xp(p) - shift_xm(p)) / (2 * DX)
    dp_dy = (shift_yp(p) - shift_ym(p)) / (2 * DX)
    return u - DT * dp_dx, v - DT * dp_dy


def vorticity(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ω = ∂v/∂x − ∂u/∂y (central differences)."""
    dv_dx = (shift_xp(v) - shift_xm(v)) / (2 * DX)
    du_dy = (shift_yp(u) - shift_ym(u)) / (2 * DX)
    return dv_dx - du_dy


def simulate() -> np.ndarray:
    """Run the vortex-street simulation. Returns vorticity history [N, H, W] fp32."""
    device = "cuda"
    mask = make_obstacle_mask(device)
    u, v, perturb_v = make_perturbed_inflow(device)
    u, v = apply_bc(u, v, mask)

    alpha_diff = NU * DT / (DX ** 2)

    history = np.empty((N_STEPS, H_GRID, W_GRID), dtype=np.float32)
    print_every = max(1, N_STEPS // 10)
    for t in range(N_STEPS):
        # Apply persistent symmetry-breaking forcing for the first PERTURB_STEPS
        if t < PERTURB_STEPS:
            v = v + perturb_v * (1.0 - t / PERTURB_STEPS)

        # 1. Advect
        u_a = advect_semilagrangian(u, u, v)
        v_a = advect_semilagrangian(v, u, v)
        u_a, v_a = apply_bc(u_a, v_a, mask)

        # 2. Viscous diffusion
        u_d = diffuse_jacobi(u_a, alpha_diff, N_DIFFUSE_ITER)
        v_d = diffuse_jacobi(v_a, alpha_diff, N_DIFFUSE_ITER)
        u_d, v_d = apply_bc(u_d, v_d, mask)

        # 3. Pressure projection
        u, v = project_incompressible(u_d, v_d, N_PRESSURE_ITER)
        u, v = apply_bc(u, v, mask)

        # 4. Compute and store vorticity
        omega = vorticity(u, v)
        history[t] = omega.cpu().numpy()

        if (t + 1) % print_every == 0:
            print(f"    step {t+1:>5}/{N_STEPS}  "
                  f"vorticity range [{omega.min().item():.2f}, {omega.max().item():.2f}]  "
                  f"|v|max={v.abs().max().item():.3f}")

    return history


def quantize_global_robust(history: np.ndarray,
                            pclip: float = 1.0
                            ) -> tuple[np.ndarray, float, float]:
    """Quantize to uint8 with percentile-clipped global min/max — protects
    against extreme outliers right at the cylinder surface skewing the
    dynamic range and starving the rest of the field of bits."""
    lo = float(np.percentile(history, pclip))
    hi = float(np.percentile(history, 100 - pclip))
    scale = 255.0 / (hi - lo + 1e-12)
    q = np.clip((history - lo) * scale, 0, 255).astype(np.uint8)
    return q, lo, hi


def dequantize_global(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return q.astype(np.float32) * (hi - lo) / 255.0 + lo


def make_yuv_frames(q_y: np.ndarray) -> torch.Tensor:
    """Wrap [N, H, W] uint8 single-channel field into [N, 3, H, W] YUV444
    with mid-grey U/V."""
    N, H, W = q_y.shape
    out = np.empty((N, 3, H, W), dtype=np.uint8)
    out[:, 0] = q_y
    out[:, 1] = 128
    out[:, 2] = 128
    return torch.from_numpy(out).cuda().contiguous()


def bench_iframe_only_fast(frames: torch.Tensor) -> tuple[int, float, list[int]]:
    """Mode A: each frame encoded as its own IDR. Returns total bytes,
    wall-clock ms, per-step bytes."""
    backend = DirectBackend(height=H_GRID, width=W_GRID, qp=QP)
    per_step = []
    try:
        _ = backend.encode_tensor_frames(frames[:1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(frames.shape[0]):
            pkts = backend.encode_tensor_frames(frames[i:i+1])
            per_step.append(sum(len(p) for p in pkts))
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        backend.close()
    return sum(per_step), elapsed_ms, per_step


def bench_pframe_chain(frames: torch.Tensor) -> tuple[int, list[bytes], float, list[int]]:
    """Mode B: all frames in one batch. Returns total bytes, raw packet list,
    wall-clock ms, per-step bytes."""
    backend = DirectBackend(height=H_GRID, width=W_GRID, qp=QP)
    try:
        _ = backend.encode_tensor_frames(frames[:1])
        torch.cuda.synchronize()
        backend.close()
        backend = DirectBackend(height=H_GRID, width=W_GRID, qp=QP)
        t0 = time.perf_counter()
        pkts = backend.encode_tensor_frames(frames)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_step = [len(p) for p in pkts]
        decoded = backend.decode_frames_cuda(pkts, frames.shape[0])
    finally:
        backend.close()
    return sum(per_step), decoded, elapsed_ms, per_step


def save_strip_png(history: np.ndarray, lo: float, hi: float,
                    sample_steps: list[int]) -> None:
    """Save a horizontal strip of vorticity heatmaps at the given timesteps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(sample_steps)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.7 * n), dpi=140,
                              constrained_layout=True)
    if n == 1:
        axes = [axes]
    vmax = max(abs(lo), abs(hi))
    for ax, t in zip(axes, sample_steps):
        ax.imshow(history[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal")
        ax.set_title(f"timestep {t}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Von Kármán vortex street (Re={int(RE)}, {W_GRID}×{H_GRID} grid) — "
        f"vorticity field at sample timesteps",
        fontsize=11,
    )
    fig.savefig(STRIP_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {STRIP_PNG}")


def save_recon_comparison(history: np.ndarray, history_recon: np.ndarray,
                            lo: float, hi: float, t: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = max(abs(lo), abs(hi))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), dpi=140,
                              constrained_layout=True)
    axes[0].imshow(history[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[0].set_title(f"original (timestep {t})")
    axes[1].imshow(history_recon[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title("after I+P-frame round-trip")
    diff = np.abs(history[t] - history_recon[t])
    im = axes[2].imshow(diff, cmap="magma", vmin=0, aspect="equal")
    axes[2].set_title(f"|diff|  (max {diff.max():.4f})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "P-frame chain reconstruction quality on the vortex street",
        fontsize=11,
    )
    fig.savefig(RECON_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {RECON_PNG}")


def save_bytes_per_step(per_step_a: list[int], per_step_b: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140, constrained_layout=True)
    x = np.arange(len(per_step_a))
    ax.plot(x, per_step_a, color="#9aa5b1", linewidth=0.7, label="Mode A: I-frames only")
    ax.plot(x, per_step_b, color="#1f4ed8", linewidth=0.7, label="Mode B: I + P-frame chain")
    ax.set_yscale("log")
    ax.set_xlabel("timestep")
    ax.set_ylabel("bytes per encoded frame (log scale)")
    ax.set_title("Per-frame bitstream size — vortex-street trajectory\n"
                 "(spikes in Mode B at steps 250/500/750/1000/1250 are auto-IDR refresh)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(BYTES_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {BYTES_PNG}")


def main() -> int:
    print(f"Vortex-street P-frame demonstration")
    print(f"  grid {W_GRID}×{H_GRID}, cylinder D={CYL_DIAMETER} at {CYL_CENTER}")
    print(f"  Re={RE}, ν={NU:.4f}, dt={DT:.3f}, U_inflow={U_INFLOW}")
    print(f"  {N_STEPS} timesteps, codec QP={QP}\n")

    print("[1] Simulating Navier-Stokes (Stam stable fluids)...")
    t0 = time.perf_counter()
    history = simulate()
    sim_ms = (time.perf_counter() - t0) * 1000
    print(f"    {N_STEPS} steps in {sim_ms:.0f} ms  ({sim_ms/N_STEPS:.1f} ms/step)")
    print(f"    vorticity range: [{history.min():.4f}, {history.max():.4f}]")
    print(f"    vorticity std: {history.std():.4f}")
    print(f"    raw fp32 trajectory: {history.nbytes:,} bytes "
          f"({history.nbytes/1e6:.2f} MB)\n")

    print("[2] Quantizing to uint8 with 1st/99th-percentile-clipped global range...")
    q, lo, hi = quantize_global_robust(history, pclip=1.0)
    print(f"    uint8 trajectory: {q.nbytes:,} bytes ({q.nbytes/1e6:.2f} MB)")
    print(f"    quant range (clipped): [{lo:.4f}, {hi:.4f}]\n")

    frames = make_yuv_frames(q)

    print("[3] Mode A: each timestep as its own IDR...")
    bytes_a, ms_a, per_step_a = bench_iframe_only_fast(frames)
    print(f"    {bytes_a:,} bytes total ({bytes_a/N_STEPS:.0f} bytes/frame)")
    print(f"    encode wall-clock {ms_a:.0f} ms\n")

    print("[4] Mode B: one IDR + P-frame chain...")
    bytes_b, decoded, ms_b, per_step_b = bench_pframe_chain(frames)
    print(f"    {bytes_b:,} bytes total ({bytes_b/N_STEPS:.0f} bytes/frame)")
    print(f"    encode wall-clock {ms_b:.0f} ms\n")

    print("[5] Reconstruction quality...")
    decoded_y = decoded[:, 0].cpu().numpy()
    history_recon = dequantize_global(decoded_y, lo, hi)
    diff = np.abs(history - history_recon)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = 99.0 if mse == 0 else float(20 * np.log10((hi - lo) / np.sqrt(mse)))
    per_step_err = diff.mean(axis=(1, 2))
    err_first10 = float(per_step_err[:10].mean())
    err_last10 = float(per_step_err[-10:].mean())
    print(f"    max abs error:  {max_abs:.6f}")
    print(f"    mean abs error: {mean_abs:.6f}")
    print(f"    PSNR:           {psnr:.2f} dB")
    print(f"    per-step error first 10 vs last 10: {err_first10:.6f} → {err_last10:.6f}\n")

    print("[6] Saving figures...")
    sample_steps = [N_STEPS // 12, N_STEPS // 4, N_STEPS // 2,
                     3 * N_STEPS // 4, N_STEPS - 1]
    save_strip_png(history, lo, hi, sample_steps)
    save_recon_comparison(history, history_recon, lo, hi, N_STEPS // 2)
    save_bytes_per_step(per_step_a, per_step_b)
    print()

    raw_bytes = history.nbytes
    quant_bytes = q.nbytes
    ratio_a_vs_quant = quant_bytes / bytes_a
    ratio_b_vs_quant = quant_bytes / bytes_b
    pframe_win = bytes_a / bytes_b

    print("=" * 64)
    print("RESULT")
    print("=" * 64)
    print(f"Raw fp32 trajectory:               {raw_bytes:>12,} bytes "
          f"({raw_bytes/1e6:.2f} MB)")
    print(f"After uint8 quantization:          {quant_bytes:>12,} bytes "
          f"({quant_bytes/1e6:.2f} MB, {raw_bytes/quant_bytes:.1f}× vs raw)")
    print(f"Mode A (I-frames only):            {bytes_a:>12,} bytes  "
          f"({raw_bytes/bytes_a:>5.1f}× vs raw, "
          f"{ratio_a_vs_quant:>5.1f}× vs uint8)")
    print(f"Mode B (I + P-frame chain):        {bytes_b:>12,} bytes  "
          f"({raw_bytes/bytes_b:>5.1f}× vs raw, "
          f"{ratio_b_vs_quant:>5.1f}× vs uint8)")
    print()
    print(f"P-frame win over I-only: ** {pframe_win:.2f}× **")
    print()

    print("Comparison context (poc/20 heat-equation: 24.15× P-frame win).")
    if pframe_win >= 5:
        print(f"==> Strong win on advection-dominated CFD. NVENC's motion estimation")
        print(f"    is genuinely tracking the vortices — {pframe_win:.1f}× extra")
        print(f"    compression beyond intra-only. Validates the temporal-")
        print(f"    compression argument for the CFD / aerospace / weather audience.")
    elif pframe_win >= 2:
        print(f"==> Real but modest win. Motion estimation helps with the")
        print(f"    translation of vortices but the non-linear shedding limits")
        print(f"    how much the codec can predict step-to-step. Still a useful")
        print(f"    primitive for these workloads.")
    else:
        print(f"==> Negligible win — interesting null finding. HEVC's motion")
        print(f"    estimation appears not to transfer well to small/fast/")
        print(f"    rotating structures. Worth understanding why.")
    print()
    print("Output figures:")
    print(f"  {STRIP_PNG.relative_to(STRIP_PNG.parent.parent.parent)}")
    print(f"  {RECON_PNG.relative_to(RECON_PNG.parent.parent.parent)}")
    print(f"  {BYTES_PNG.relative_to(BYTES_PNG.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
