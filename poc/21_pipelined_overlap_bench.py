"""21 — Pipelined codec + offload bench (does the bandwidth multiplier survive?)

Tests the load-bearing claim — that compressing activations on dedicated
NVENC silicon, while compute runs on a separate CUDA stream, produces a
useful bandwidth multiplier on PCIe-class wires. PoC 20 showed that
sequential codec usage is too slow to win on PCIe Gen4. The question
this PoC answers: with proper overlap, does the win come back?

The wire we can actually measure on a single-GPU rig is **PCIe transit
between VRAM and pinned host memory** (the offload pattern). That's
PCIe-bound exactly like cross-GPU peer-to-peer would be, and it's the
most relevant wire for the LoRA-training-pitch scenario (12 GB cards
training Flux LoRAs) — when training-loop offload to system RAM
dominates step time, this is the path that gets compressed.

We measure each stage independently, then compute three pipelined
wall-clock scenarios for a representative training-loop pattern of
N iterations of [compute layer → offload activation → reload → compute]:

  S0. Raw offload, no codec
       — compute on stream A || (D->H pinned) on stream B || (H->D)
       Baseline: PCIe is the only thing carrying activation bytes.

  S1. Sequential codec offload (no overlap)
       — compute → encode → D->H compressed → H->D compressed → decode
       Each stage waits for the previous. PoC 20 showed this loses on
       fast wires. Reported here for reference.

  S2. Overlapped codec offload (the load-bearing scenario)
       — encode on stream B runs concurrent with compute on stream A
         on the OUTGOING side; decode on stream B runs concurrent with
         compute on stream A on the INCOMING side. Critical-path
         wall-clock per iter = max(compute, encode) + transit_compressed
         + max(compute, decode), pipelined across iters.

Compute proxy: 30×4096² fp16 GEMM, matching poc/17's setup. Roughly
representative of a Flux block forward on the 5090.

Reports:
- Per-stage wall-clock measurements
- Pipeline wall-clock for each scenario
- Effective offload bandwidth = (N × raw_bytes) / wall_clock
- Honest verdict: does the codec multiply effective bandwidth, or not?
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from nvenc_compress.direct.backend import DirectBackend
from nvenc_compress.direct.multi_backend import MultiEngineDirectBackend


# ---- Configuration -------------------------------------------------
H, W = 256, 256
N_FRAMES = 128                     # 128 × 192 KB ≈ 25 MB activation
RAW_BYTES = N_FRAMES * 3 * H * W   # ≈ 25 MB
N_ITERS = 8                        # pipeline iterations to measure overlap
GEMM_DIM = 4096                    # 4096^2 fp16 GEMM
GEMM_REPS = 30                     # 30× repeat to match poc/17's compute proxy


# ---- Helpers -------------------------------------------------------
def synth_activation(seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = np.empty((N_FRAMES, 3, H, W), dtype=np.float32)
    for f in range(N_FRAMES):
        field = np.zeros((H, W), dtype=np.float32)
        for _ in range(4):
            kx = rng.uniform(0.5, 4.0); ky = rng.uniform(0.5, 4.0)
            phase = rng.uniform(0, 2 * np.pi); amp = rng.uniform(20, 80)
            field += amp * np.sin(2 * np.pi * (kx * xx / W + ky * yy / H) + phase)
        field += rng.standard_normal((H, W)) * 8.0
        for c in range(3):
            out[f, c] = field + rng.standard_normal((H, W)) * 4.0
    out = (out - out.min()) / (out.max() - out.min())
    arr = (out * 255).astype(np.uint8)
    return torch.from_numpy(arr).cuda().contiguous()


def time_compute_proxy() -> float:
    """30×4096² fp16 GEMM, matching poc/17. Returns time in seconds."""
    A = torch.randn(GEMM_DIM, GEMM_DIM, dtype=torch.float16, device="cuda")
    B = torch.randn(GEMM_DIM, GEMM_DIM, dtype=torch.float16, device="cuda")
    # Warm-up
    for _ in range(3):
        _ = A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = A @ B
    for _ in range(GEMM_REPS - 1):
        C = A @ C
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def time_pinned_offload(tensor: torch.Tensor) -> tuple[float, float]:
    """D->H to pinned memory, then H->D. Returns (d2h, h2d) seconds."""
    pinned = torch.empty_like(tensor, device="cpu").pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pinned.copy_(tensor, non_blocking=False)
    torch.cuda.synchronize()
    t_d2h = time.perf_counter() - t0
    t1 = time.perf_counter()
    back = torch.empty_like(tensor)
    back.copy_(pinned, non_blocking=False)
    torch.cuda.synchronize()
    t_h2d = time.perf_counter() - t1
    return t_d2h, t_h2d


def time_single_engine_codec(tensor: torch.Tensor, qp: int, lossless: bool
                                ) -> tuple[float, float, int]:
    backend = DirectBackend(height=H, width=W, qp=qp, lossless=lossless)
    try:
        # Warm-up
        _ = backend.encode_tensor_frames(tensor[:2])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        packets = backend.encode_tensor_frames(tensor)
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        _ = backend.decode_frames_cuda(packets, tensor.shape[0])
        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t1
        return t_enc, t_dec, sum(len(p) for p in packets)
    finally:
        backend.close()


def time_multi_engine_codec(tensor: torch.Tensor, qp: int, n_engines: int = 3
                              ) -> tuple[float, float, int]:
    """Split tensor across N engines. Decode is single-engine because
    MultiEngineDirectBackend's decode path doesn't parallelise yet."""
    multi = MultiEngineDirectBackend(height=H, width=W, qp=qp,
                                       n_engines=n_engines)
    # Split frames across engines via numpy batches
    frames_np = tensor.cpu().numpy()
    chunks = np.array_split(frames_np, n_engines)
    try:
        # Warm-up
        _ = multi.encode_frames_batch([chunks[0][:2] for _ in range(n_engines)])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = multi.encode_frames_batch(list(chunks))
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0

        # Decode each chunk (sequential — multi-engine decode is complex)
        decode_backend = DirectBackend(height=H, width=W, qp=qp)
        try:
            t1 = time.perf_counter()
            for chunk_i, packets in enumerate(results):
                _ = decode_backend.decode_frames(packets, chunks[chunk_i].shape[0])
            torch.cuda.synchronize()
            t_dec = time.perf_counter() - t1
        finally:
            decode_backend.close()

        encoded_total = sum(sum(len(p) for p in packs) for packs in results)
        return t_enc, t_dec, encoded_total
    finally:
        for b in multi.backends:
            b.close()


def time_pinned_offload_compressed(packets: list[bytes]) -> tuple[float, float]:
    """Time host<->device transfer of the COMPRESSED bitstream.
    Bitstream is bytes already on host; we measure pinned-buf->device round-trip
    as a proxy (the actual compressed transit on the wire is dominated by
    bytes-to-send / wire_speed)."""
    total_bytes = sum(len(p) for p in packets)
    # Build a single contiguous host pinned tensor of total_bytes bytes
    host = torch.empty(total_bytes, dtype=torch.uint8, device="cpu").pin_memory()
    offset = 0
    for p in packets:
        host[offset:offset + len(p)] = torch.frombuffer(p, dtype=torch.uint8)
        offset += len(p)
    device = torch.empty_like(host, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    device.copy_(host, non_blocking=False)
    torch.cuda.synchronize()
    t_h2d = time.perf_counter() - t0
    t1 = time.perf_counter()
    host2 = torch.empty_like(host).pin_memory()
    host2.copy_(device, non_blocking=False)
    torch.cuda.synchronize()
    t_d2h = time.perf_counter() - t1
    return t_d2h, t_h2d


def fmt_ms(s: float) -> str:
    return f"{s*1000:.2f} ms"


def fmt_bw(bytes_total: int, seconds: float) -> str:
    bps = bytes_total / max(seconds, 1e-9)
    if bps > 1e9:
        return f"{bps/1e9:.1f} GB/s"
    return f"{bps/1e6:.0f} MB/s"


# ---- Main ----------------------------------------------------------
def main() -> int:
    print("=" * 76)
    print("Pipelined codec + offload bench")
    print("=" * 76)
    print(f"  Activation: {N_FRAMES} × [3, {H}, {W}] uint8 = {RAW_BYTES/1e6:.1f} MB raw")
    print(f"  Compute proxy: {GEMM_REPS}×{GEMM_DIM}² fp16 GEMM (matches poc/17)")
    print(f"  Wire: VRAM ↔ pinned host RAM (PCIe Gen4 ×16, ~32 GB/s effective)")
    print()

    print("[setup] Synthesizing activation...")
    tensor = synth_activation()
    print()

    print("[1] Compute proxy (single iteration)")
    t_compute = time_compute_proxy()
    print(f"  GEMM proxy: {fmt_ms(t_compute)}")
    print()

    print("[2] Raw offload — pinned D↔H of full 25 MB")
    t_d2h_raw, t_h2d_raw = time_pinned_offload(tensor)
    t_offload_raw = t_d2h_raw + t_h2d_raw
    print(f"  D→H: {fmt_ms(t_d2h_raw)}   H→D: {fmt_ms(t_h2d_raw)}   total: {fmt_ms(t_offload_raw)}")
    print(f"  effective wire speed: {fmt_bw(2 * RAW_BYTES, t_offload_raw)}")
    print()

    print("[3] Single-engine codec — DirectBackend (lossless + QP=18)")
    t_enc_l, t_dec_l, bytes_l = time_single_engine_codec(tensor, qp=18, lossless=True)
    print(f"  Lossless: encode {fmt_ms(t_enc_l)}, decode {fmt_ms(t_dec_l)}, {bytes_l/1e6:.2f} MB encoded")
    t_enc_18, t_dec_18, bytes_18 = time_single_engine_codec(tensor, qp=18, lossless=False)
    print(f"  QP=18:    encode {fmt_ms(t_enc_18)}, decode {fmt_ms(t_dec_18)}, {bytes_18/1e6:.2f} MB encoded")
    print()

    print("[4] Multi-engine codec — MultiEngineDirectBackend, 3 NVENC engines")
    print("  (encoder-side parallelism only; decode goes through one engine)")
    try:
        t_enc_mu_l, t_dec_mu_l, bytes_mu_l = time_multi_engine_codec(tensor, qp=18)
        print(f"  Multi-engine encode (QP=18): {fmt_ms(t_enc_mu_l)}, "
              f"decode (single-engine): {fmt_ms(t_dec_mu_l)}, "
              f"{bytes_mu_l/1e6:.2f} MB encoded")
    except Exception as e:
        print(f"  Multi-engine codec failed: {type(e).__name__}: {e}")
        t_enc_mu_l = t_enc_18  # fall back to single-engine for the model
        t_dec_mu_l = t_dec_18
        bytes_mu_l = bytes_18
    print()

    print("[5] Pinned offload of COMPRESSED bytes (QP=18, ~3 MB)")
    backend = DirectBackend(height=H, width=W, qp=18, lossless=False)
    try:
        packets_18 = backend.encode_tensor_frames(tensor)
    finally:
        backend.close()
    t_d2h_c, t_h2d_c = time_pinned_offload_compressed(packets_18)
    t_offload_compressed = t_d2h_c + t_h2d_c
    print(f"  D→H: {fmt_ms(t_d2h_c)}   H→D: {fmt_ms(t_h2d_c)}   total: {fmt_ms(t_offload_compressed)}")
    print()

    # ---- Pipeline modelling ---------------------------------------
    print("=" * 76)
    print(f"Pipeline wall-clock model (N={N_ITERS} iterations)")
    print("=" * 76)

    # S0: Raw offload, compute concurrent with PCIe (independent units)
    # Per iter: max(compute, raw_offload) — best case overlap
    s0_per_iter = max(t_compute, t_offload_raw)
    s0_total = N_ITERS * s0_per_iter
    print(f"\nS0. Raw offload (compute || PCIe, no codec)")
    print(f"    per iter: max({fmt_ms(t_compute)}, {fmt_ms(t_offload_raw)}) = {fmt_ms(s0_per_iter)}")
    print(f"    {N_ITERS} iters: {fmt_ms(s0_total)}")
    print(f"    effective offload bandwidth: {fmt_bw(N_ITERS * 2 * RAW_BYTES, s0_total)}")

    # S1: Sequential codec, no overlap
    # Per iter: compute + encode + transit_compressed + decode
    s1_per_iter = t_compute + t_enc_18 + t_offload_compressed + t_dec_18
    s1_total = N_ITERS * s1_per_iter
    print(f"\nS1. Sequential codec (compute → encode → transit → decode, no overlap)")
    print(f"    per iter: {fmt_ms(t_compute)} + {fmt_ms(t_enc_18)} + {fmt_ms(t_offload_compressed)} + {fmt_ms(t_dec_18)} = {fmt_ms(s1_per_iter)}")
    print(f"    {N_ITERS} iters: {fmt_ms(s1_total)}")
    print(f"    effective offload bandwidth: {fmt_bw(N_ITERS * 2 * RAW_BYTES, s1_total)}")
    print(f"    speedup vs S0: {s0_total/s1_total:.2f}×")

    # S2: Overlapped codec on multi-engine
    # Per iter (steady-state): max(compute, encode_multi) + transit_compressed + max(compute, decode)
    # Decode is single-engine in our impl, so it's slower
    s2_per_iter = max(t_compute, t_enc_mu_l) + t_offload_compressed + max(t_compute, t_dec_mu_l)
    s2_total = N_ITERS * s2_per_iter
    print(f"\nS2. Overlapped multi-engine codec (encode || compute, decode || compute)")
    print(f"    per iter: max({fmt_ms(t_compute)}, {fmt_ms(t_enc_mu_l)}) + {fmt_ms(t_offload_compressed)} + max({fmt_ms(t_compute)}, {fmt_ms(t_dec_mu_l)}) = {fmt_ms(s2_per_iter)}")
    print(f"    {N_ITERS} iters: {fmt_ms(s2_total)}")
    print(f"    effective offload bandwidth: {fmt_bw(N_ITERS * 2 * RAW_BYTES, s2_total)}")
    print(f"    speedup vs S0: {s0_total/s2_total:.2f}×")

    # S2-best-case: assume codec encode/decode hide perfectly behind compute
    s2_best_per_iter = t_compute + t_offload_compressed
    s2_best_total = N_ITERS * s2_best_per_iter
    print(f"\nS2-best. If codec hides 100% behind compute (theoretical ceiling)")
    print(f"    per iter: {fmt_ms(t_compute)} + {fmt_ms(t_offload_compressed)} = {fmt_ms(s2_best_per_iter)}")
    print(f"    {N_ITERS} iters: {fmt_ms(s2_best_total)}")
    print(f"    effective offload bandwidth: {fmt_bw(N_ITERS * 2 * RAW_BYTES, s2_best_total)}")
    print(f"    speedup vs S0: {s0_total/s2_best_total:.2f}×")

    print()
    print("=" * 76)
    print("Verdict")
    print("=" * 76)
    print(f"  S0 (no codec, just compute || PCIe) effective bw: {fmt_bw(2*RAW_BYTES, s0_per_iter)}")
    print(f"  S1 (sequential codec) effective bw:                {fmt_bw(2*RAW_BYTES, s1_per_iter)}")
    print(f"  S2 (overlapped multi-engine codec) effective bw:   {fmt_bw(2*RAW_BYTES, s2_per_iter)}")
    print(f"  S2-best (codec fully hidden) effective bw:         {fmt_bw(2*RAW_BYTES, s2_best_per_iter)}")
    print()
    if s2_per_iter < s0_per_iter * 0.95:
        print(f"  RESULT: codec WINS — {s0_per_iter/s2_per_iter:.2f}× speedup over raw offload")
    elif s2_per_iter < s0_per_iter * 1.05:
        print(f"  RESULT: codec ROUGHLY MATCHES raw offload (within 5%)")
    else:
        print(f"  RESULT: codec LOSES — adds {(s2_per_iter/s0_per_iter - 1)*100:.0f}% wall-clock vs raw")
    print()
    print(f"  Compute time ({fmt_ms(t_compute)}) vs raw offload ({fmt_ms(t_offload_raw)})")
    print(f"    if compute >> raw offload, PCIe is already free — codec can't improve.")
    print(f"    if raw offload >> compute, PCIe is the bottleneck — codec can help by")
    print(f"    reducing wire bytes, but only if codec time can hide behind compute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
