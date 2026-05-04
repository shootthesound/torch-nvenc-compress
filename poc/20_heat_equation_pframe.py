"""20 — Heat equation P-frame chain: NVENC's temporal compression on a non-ML workload.

The architectural argument: NVENC has TWO compression modes — intra (I-frame,
each frame independent) and inter (P-frame, each frame as a delta from the
previous via motion vectors + residuals). Most of video's compression magic
lives in the P-frame side; typical video gets 5–10× more compression from
P-frames than from I-frames alone.

The current `DirectBackend` already produces P-frames within a single
encode_tensor_frames() call (only frame 0 forces IDR; frames 1..N-1 default
to encoder-chosen frame type, which is P-frame given frameIntervalP=1). What
the project's existing benchmarks haven't exercised is feeding it *temporally
coherent* data — the FLUX activations and Mistral KV cache PoCs treat
sequential YUV frames as packed channels rather than time-series, and
PCA-rotated channels are orthogonal by construction (see null_findings/n3
for why channel reordering doesn't help).

This PoC is the first project benchmark where the frame sequence has real
temporal coherence: a 2D heat equation simulator. ∂u/∂t = α∇²u. Each
timestep is a small diffusion-driven perturbation of the previous — exactly
the pattern P-frames were designed for.

Two compression modes compared on the same 1000-step simulation:
  - Mode A (I-frames only): each timestep encoded as its own IDR. Achieved
                            by calling encode_tensor_frames() per-timestep
                            so each call's frame 0 forces a fresh keyframe.
  - Mode B (I + P-frame chain): all timesteps in one encode_tensor_frames()
                                 call. Frame 0 IDR, rest are P-frames.
                                 Encoder default gopLength=250 inserts a
                                 keyframe refresh every 250 frames.

If Mode B's bitstream is much smaller than Mode A's, P-frame temporal
compression is doing real work — and the broader claim holds: NVENC's
inter-frame mode is a free delta-codec for any GPU workload that produces
temporally-coherent state (CFD, FEM, MD, weather sims, progressive
renderers, sci-viz streaming).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from nvenc_compress.direct.backend import DirectBackend


# ---- simulation params ----------------------------------------------------
GRID = 256              # spatial grid size (NVENC HEVC min ~144 on Blackwell)
N_STEPS = 1000          # number of timesteps to simulate + encode
ALPHA = 0.1             # thermal diffusivity
DT = 0.2                # timestep — alpha*dt/dx^2 < 0.25 for 2D Forward-Euler stability
DX = 1.0
N_HOTSPOTS = 8          # initial gaussian bumps

# ---- codec params ---------------------------------------------------------
QP = 18                 # near-lossless


def make_initial_field(rng: np.random.Generator) -> np.ndarray:
    """N_HOTSPOTS gaussian bumps on a GRID×GRID field, scaled to [0, 1]."""
    field = np.zeros((GRID, GRID), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    for _ in range(N_HOTSPOTS):
        cy, cx = rng.uniform(GRID // 6, 5 * GRID // 6, 2)
        sigma = rng.uniform(GRID / 24, GRID / 12)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return (field / field.max()).astype(np.float32)


def step_heat(u: torch.Tensor, alpha: float, dt: float, dx: float) -> torch.Tensor:
    """One forward-Euler timestep on the 2D heat equation.
    Dirichlet zero boundary (no flux out of the grid)."""
    lap = torch.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1]
        + u[1:-1, 2:] + u[1:-1, :-2]
        - 4 * u[1:-1, 1:-1]
    )
    return u + (alpha * dt / dx ** 2) * lap


def simulate(seed: int = 42) -> np.ndarray:
    """Run the heat equation forward N_STEPS timesteps. Returns [N, H, W] fp32."""
    rng = np.random.default_rng(seed)
    u = torch.from_numpy(make_initial_field(rng)).cuda()
    history = np.empty((N_STEPS, GRID, GRID), dtype=np.float32)
    for t in range(N_STEPS):
        history[t] = u.cpu().numpy()
        u = step_heat(u, ALPHA, DT, DX)
    return history


def quantize_global(history: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantize the whole [N, H, W] trajectory to uint8 with a SHARED global
    min/max. Critical detail: per-frame min/max would inject artificial
    frame-to-frame noise and destroy P-frame compression. The whole point
    is that consecutive frames look almost identical to the codec."""
    lo = float(history.min())
    hi = float(history.max())
    scale = 255.0 / (hi - lo + 1e-12)
    q = np.clip((history - lo) * scale, 0, 255).astype(np.uint8)
    return q, lo, hi


def dequantize_global(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return q.astype(np.float32) * (hi - lo) / 255.0 + lo


def make_yuv_frames(q_y: np.ndarray) -> torch.Tensor:
    """Wrap [N, H, W] uint8 single-channel field into [N, 3, H, W] YUV444
    with mid-grey U/V (no chroma signal — heat field is monochrome)."""
    N, H, W = q_y.shape
    out = np.empty((N, 3, H, W), dtype=np.uint8)
    out[:, 0] = q_y
    out[:, 1] = 128
    out[:, 2] = 128
    return torch.from_numpy(out).cuda().contiguous()


def bench_iframe_only(frames: torch.Tensor) -> tuple[int, float]:
    """Mode A: each frame encoded as its own IDR. One encode_tensor_frames
    call per frame, fresh encoder context each time so no inter-frame
    references can form."""
    total_bytes = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(frames.shape[0]):
        # Re-create the backend per-frame to guarantee no GOP carryover.
        # This is the worst-case "no temporal compression at all" baseline.
        backend = DirectBackend(height=GRID, width=GRID, qp=QP)
        try:
            pkts = backend.encode_tensor_frames(frames[i:i+1])
            total_bytes += sum(len(p) for p in pkts)
        finally:
            backend.close()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return total_bytes, elapsed_ms


def bench_iframe_only_fast(frames: torch.Tensor) -> tuple[int, float]:
    """Mode A (faster variant): same encoder reused, but each call is a
    1-frame batch so encode_tensor_frames forces IDR on its frame 0. The
    encoder context persists but each call's bitstream starts with a fresh
    keyframe — no P-frame referencing across calls."""
    total_bytes = 0
    backend = DirectBackend(height=GRID, width=GRID, qp=QP)
    try:
        # Warmup
        _ = backend.encode_tensor_frames(frames[:1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(frames.shape[0]):
            pkts = backend.encode_tensor_frames(frames[i:i+1])
            total_bytes += sum(len(p) for p in pkts)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        backend.close()
    return total_bytes, elapsed_ms


def bench_pframe_chain(frames: torch.Tensor) -> tuple[int, list[bytes], float]:
    """Mode B: all frames in one batch. Frame 0 forced IDR; frames 1..N-1
    default to P-frames (frameIntervalP=1, no B-frames). Encoder's default
    gopLength=250 also inserts auto-IDRs at frames 250, 500, 750."""
    backend = DirectBackend(height=GRID, width=GRID, qp=QP)
    try:
        # Warmup
        _ = backend.encode_tensor_frames(frames[:1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Re-create after warmup so the GOP starts clean
        backend.close()
        backend = DirectBackend(height=GRID, width=GRID, qp=QP)
        pkts = backend.encode_tensor_frames(frames)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_bytes = sum(len(p) for p in pkts)
        decoded = backend.decode_frames_cuda(pkts, frames.shape[0])
    finally:
        backend.close()
    return total_bytes, decoded, elapsed_ms


def main() -> int:
    print(f"Heat-equation P-frame demonstration")
    print(f"  grid {GRID}×{GRID}, {N_STEPS} timesteps, α={ALPHA}, QP={QP}\n")

    print("[1] Simulating heat equation...")
    t0 = time.perf_counter()
    history = simulate()
    sim_ms = (time.perf_counter() - t0) * 1000
    print(f"    {N_STEPS} steps in {sim_ms:.0f} ms")
    print(f"    field range: [{history.min():.4f}, {history.max():.4f}]")
    print(f"    raw fp32 trajectory: {history.nbytes:,} bytes "
          f"({history.nbytes/1e6:.2f} MB)\n")

    print("[2] Quantizing to uint8 with shared global min/max...")
    q, lo, hi = quantize_global(history)
    print(f"    uint8 trajectory: {q.nbytes:,} bytes ({q.nbytes/1e6:.2f} MB)")
    print(f"    quant range: [{lo:.4f}, {hi:.4f}]\n")

    frames = make_yuv_frames(q)

    print("[3] Mode A: each timestep as its own IDR (no temporal compression)...")
    bytes_a, ms_a = bench_iframe_only_fast(frames)
    print(f"    {bytes_a:,} bytes total ({bytes_a/N_STEPS:.0f} bytes/frame)")
    print(f"    encode wall-clock {ms_a:.0f} ms\n")

    print("[4] Mode B: one IDR + P-frame chain (temporal compression on)...")
    bytes_b, decoded, ms_b = bench_pframe_chain(frames)
    print(f"    {bytes_b:,} bytes total ({bytes_b/N_STEPS:.0f} bytes/frame)")
    print(f"    encode wall-clock {ms_b:.0f} ms\n")

    print("[5] Reconstruct from P-frame bitstream and verify quality...")
    decoded_y = decoded[:, 0].cpu().numpy()
    history_recon = dequantize_global(decoded_y, lo, hi)
    diff = np.abs(history - history_recon)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = 99.0 if mse == 0 else float(20 * np.log10((hi - lo) / np.sqrt(mse)))
    # Per-step error to see if it grows over the trajectory
    per_step_err = np.abs(history - history_recon).mean(axis=(1, 2))
    err_first10 = float(per_step_err[:10].mean())
    err_last10 = float(per_step_err[-10:].mean())
    print(f"    max abs error:  {max_abs:.6f}")
    print(f"    mean abs error: {mean_abs:.6f}")
    print(f"    PSNR:           {psnr:.2f} dB")
    print(f"    per-step error first 10 vs last 10: {err_first10:.6f} → {err_last10:.6f}")
    print()

    raw_bytes = history.nbytes
    quant_bytes = q.nbytes
    ratio_a_vs_raw = raw_bytes / bytes_a
    ratio_b_vs_raw = raw_bytes / bytes_b
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
          f"({ratio_a_vs_raw:>5.1f}× vs raw, "
          f"{ratio_a_vs_quant:>5.1f}× vs uint8)")
    print(f"Mode B (I + P-frame chain):        {bytes_b:>12,} bytes  "
          f"({ratio_b_vs_raw:>5.1f}× vs raw, "
          f"{ratio_b_vs_quant:>5.1f}× vs uint8)")
    print()
    print(f"P-frame win over I-only: ** {pframe_win:.2f}× **")
    print()
    if pframe_win >= 5:
        print(f"==> Strong win. NVENC's P-frame chain captures "
              f"{pframe_win:.1f}× more")
        print(f"    of the temporal coherence in this trajectory than intra-only.")
        print(f"    Validates the broader claim that the codec's temporal mode")
        print(f"    is exploitable for any GPU workload that produces frame-by-")
        print(f"    frame coherent state (CFD, FEM, weather, MD, progressive")
        print(f"    rendering, sci-viz streaming).")
    elif pframe_win >= 2:
        print(f"==> Real but modest win. P-frames help but the trajectory may")
        print(f"    have more variation than expected for this α/dt combination.")
    else:
        print(f"==> Negligible win. Worth investigating: the encoder may be")
        print(f"    inserting more IDRs than expected, or the QP is so loose")
        print(f"    that I-frame quantization dominates.")
    print()
    print("Notes:")
    print(f"  - Mode B includes auto-IDR refresh every gopLength=250 frames")
    print(f"    (4 IDRs total in 1000 steps). For a pure single-IDR test,")
    print(f"    override gopLength in initialize_encoder_hevc_yuv444.")
    print(f"  - Reconstruction error above is end-to-end fp32 → uint8 quantize")
    print(f"    → HEVC encode → HEVC decode → uint8 → fp32. The uint8 step is")
    print(f"    the dominant error contributor at this QP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
