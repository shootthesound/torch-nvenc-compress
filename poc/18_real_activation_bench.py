"""18 — Real-workload bench: DirectBackend vs PyAV CodecSession on FLUX activations.

Loads captured FLUX.2 Klein 9B mid-block activations from ring0/data/, runs
each through the full PCA + per-channel quantize + YUV444 packing pipeline,
then times the codec stage on three backends:

  - PyAV CodecSession (single engine, single output buffer)
  - PyAV MultiEngineCodecSession (3 engines × Python threads)
  - DirectBackend zero-copy (pure ctypes, 8-deep output ring)

This is the apples-to-apples comparison the project's headline depends on:
the same K=500 LOO-PCA-projected activation, the same 256x256 YUV444 frames,
the same QP=18, scored on wall-clock encode + decode time + reconstruction
quality.

Defaults are sized to run in ~10s — N=4 holdout activations, K=500.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from nvenc_compress.codec import pack_yuv_frames, unpack_yuv_frames, pad_to_min, crop_pad, MIN_FRAME_DIM
from nvenc_compress.pca import build_shared_basis
from nvenc_compress.quantize import per_channel_quantise, per_channel_dequantise
from nvenc_compress.direct.backend import DirectBackend
from nvenc_compress.direct.multi_backend import MultiEngineDirectBackend


DATA_DIR = Path("W:/Peter/Documents/Development/NVENC Activations/ring0/data")
N_CALIB = 16        # number of activations to build PCA basis from
N_HOLDOUT = 4       # number of activations to bench codec on
K = 500             # PCA rank
QP = 18


def load_activation(path: Path) -> torch.Tensor:
    """Load a saved FLUX activation, return as [T, D] float32 on CPU."""
    d = torch.load(str(path), weights_only=False, map_location="cpu")
    t = d["tensor"].float().squeeze(0)  # [4096, 4096]
    return t


def preprocess(tensor: torch.Tensor, basis) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, int, int, int]:
    """Run the full pre-codec pipeline. Returns frames + everything needed to
    invert it after decode."""
    R = basis.project(tensor)                  # [T, K]
    T = R.shape[0]
    side = int(np.sqrt(T))
    while side * side < T:
        side += 1
    pad_rows = side * side - T
    if pad_rows:
        R = torch.cat([R, torch.zeros(pad_rows, basis.K, dtype=R.dtype)], dim=0)
    R_chw = R.T.reshape(basis.K, side, side).contiguous().cpu()
    q, scale, offset = per_channel_quantise(R_chw)
    q_padded, padded_h, padded_w = pad_to_min(q, MIN_FRAME_DIM)
    frames, pad_channels = pack_yuv_frames(q_padded)
    return frames, padded_h, padded_w, scale, offset, pad_channels, side, T


def reconstruct(frames: np.ndarray, basis, scale, offset, pad_channels, side, T, padded_h, padded_w) -> torch.Tensor:
    """Run the post-decode pipeline. Returns reconstructed [T, D] float32."""
    q_padded_recon = unpack_yuv_frames(frames, basis.K, pad_channels)
    q_recon = crop_pad(q_padded_recon, side, side)
    R_chw_recon = per_channel_dequantise(q_recon, scale, offset)
    R_recon = R_chw_recon.reshape(basis.K, -1).T
    R_recon = R_recon[:T]
    return basis.invert(R_recon)


def cosine_similarity_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity between two [T, D] tensors."""
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + 1e-12
    return num / den


def bench_pyav_single(holdout_frames, padded_h, padded_w, n_frames_per):
    from nvenc_compress.session import CodecSession
    print("[pyav-single] CodecSession init...")
    t0 = time.perf_counter()
    sess = CodecSession(height=padded_h, width=padded_w, qp=QP)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")
    try:
        # Encode all
        t0 = time.perf_counter()
        all_packets = []
        for frames in holdout_frames:
            pkts = sess.encode_frames(frames)
            all_packets.append(pkts)
        enc_ms = (time.perf_counter() - t0) * 1000
        # Decode all
        t0 = time.perf_counter()
        all_decoded = []
        for pkts, n_f in zip(all_packets, n_frames_per):
            decoded = sess.decode_frames(pkts, n_f)
            all_decoded.append(decoded)
        dec_ms = (time.perf_counter() - t0) * 1000
        return enc_ms, dec_ms, all_packets, all_decoded
    finally:
        sess.close()


def bench_pyav_multi(holdout_frames, padded_h, padded_w, n_frames_per):
    from nvenc_compress.multi_session import MultiEngineCodecSession
    print("\n[pyav-multi] MultiEngineCodecSession init (3 engines)...")
    t0 = time.perf_counter()
    multi = MultiEngineCodecSession(height=padded_h, width=padded_w, qp=QP, n_engines=3)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")
    try:
        # Multi works one tensor at a time per call. Equivalent serial pattern:
        t0 = time.perf_counter()
        all_packets = []
        for frames in holdout_frames:
            # Use the underlying session's encode_frames directly for fairness
            pkts = multi.sessions[0].encode_frames(frames)
            all_packets.append(pkts)
        enc_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        all_decoded = []
        for pkts, n_f in zip(all_packets, n_frames_per):
            decoded = multi.sessions[0].decode_frames(pkts, n_f)
            all_decoded.append(decoded)
        dec_ms = (time.perf_counter() - t0) * 1000
        return enc_ms, dec_ms, all_packets, all_decoded
    finally:
        multi.close()


def bench_direct_multi(holdout_frames, padded_h, padded_w, n_frames_per, n_engines=3):
    print(f"\n[direct-multi] MultiEngineDirectBackend init ({n_engines} engines, pool=8)...")
    t0 = time.perf_counter()
    multi = MultiEngineDirectBackend(height=padded_h, width=padded_w, qp=QP,
                                       n_engines=n_engines, output_pool_size=8)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")
    try:
        # Stage all frames on GPU once
        cuda_frames_list = [
            torch.from_numpy(np.ascontiguousarray(f)).cuda() for f in holdout_frames
        ]
        torch.cuda.synchronize()

        # Warmup each engine
        for backend in multi.backends:
            _ = backend.encode_tensor_frames(cuda_frames_list[0][:1])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        all_packets = multi.encode_tensor_batch(cuda_frames_list)
        torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t0) * 1000

        # Decode timing — also measure the new zero-copy CUDA-tensor decode
        t0 = time.perf_counter()
        all_decoded = multi.decode_frames_batch(all_packets, n_frames_per)
        dec_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        decoded_cuda = multi.decode_frames_cuda_batch(all_packets, n_frames_per)
        torch.cuda.synchronize()
        dec_cuda_ms = (time.perf_counter() - t0) * 1000
        # Convert CUDA tensors to numpy for downstream reconstruction (one DtoH per
        # tensor; not part of the timed decode region above)
        all_decoded_via_cuda = [t.cpu().numpy() for t in decoded_cuda]

        print(f"    decode (numpy/DtoH): {dec_ms:.1f} ms ({dec_ms/sum(n_frames_per):.3f} ms/f)")
        print(f"    decode (torch/D2D):  {dec_cuda_ms:.1f} ms ({dec_cuda_ms/sum(n_frames_per):.3f} ms/f)")
        return enc_ms, dec_cuda_ms, all_packets, all_decoded_via_cuda
    finally:
        multi.close()


def bench_direct(holdout_frames, padded_h, padded_w, n_frames_per):
    print("\n[direct] DirectBackend init (zero-copy, pool=8)...")
    t0 = time.perf_counter()
    backend = DirectBackend(height=padded_h, width=padded_w, qp=QP)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")
    try:
        # Stage all frames on GPU once
        cuda_frames_list = [
            torch.from_numpy(np.ascontiguousarray(f)).cuda() for f in holdout_frames
        ]
        torch.cuda.synchronize()

        # Warmup with the first frame to pay one-time CUDA-buf register cost
        _ = backend.encode_tensor_frames(cuda_frames_list[0][:1])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        all_packets = []
        for cf in cuda_frames_list:
            pkts = backend.encode_tensor_frames(cf)
            all_packets.append(pkts)
        torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        all_decoded = []
        for pkts, n_f in zip(all_packets, n_frames_per):
            decoded = backend.decode_frames(pkts, n_f)
            all_decoded.append(decoded)
        dec_ms = (time.perf_counter() - t0) * 1000
        return enc_ms, dec_ms, all_packets, all_decoded
    finally:
        backend.close()


def main() -> int:
    print(f"Real-activation bench — {N_HOLDOUT} holdout × FLUX.2 Klein 9B activations,")
    print(f"PCA K={K}, QP={QP}, calibration N={N_CALIB}\n")

    paths = sorted(DATA_DIR.glob("activation_*.pt"))[: N_CALIB + N_HOLDOUT]
    if len(paths) < N_CALIB + N_HOLDOUT:
        print(f"!!! only {len(paths)} captures available; need {N_CALIB + N_HOLDOUT}")
        return 1

    print(f"Loading {len(paths)} activations from {DATA_DIR}...")
    tensors = [load_activation(p) for p in paths]
    calib_tensors = tensors[:N_CALIB]
    holdout_tensors = tensors[N_CALIB:]
    print(f"  calibration: {len(calib_tensors)} tensors, holdout: {len(holdout_tensors)}\n")

    print(f"Building shared PCA basis (K={K})...")
    t0 = time.perf_counter()
    basis = build_shared_basis(calib_tensors, K=K)
    print(f"  done in {(time.perf_counter() - t0)*1000:.0f} ms\n")

    print("Pre-processing holdout (PCA + quant + YUV pack)...")
    holdout_frames = []
    holdout_meta = []
    for i, t in enumerate(holdout_tensors):
        frames, padded_h, padded_w, scale, offset, pad_channels, side, T = preprocess(t, basis)
        holdout_frames.append(frames)
        holdout_meta.append((scale, offset, pad_channels, side, T, padded_h, padded_w))
        print(f"  [{i}] frames {frames.shape} dtype={frames.dtype}")
    n_frames_per = [f.shape[0] for f in holdout_frames]
    padded_h, padded_w = holdout_meta[0][5], holdout_meta[0][6]
    total_frames = sum(n_frames_per)
    print(f"  total frames across holdout: {total_frames}\n")

    # ---- benches ----
    enc_p, dec_p, pkts_p, dec_arrs_p = bench_pyav_single(holdout_frames, padded_h, padded_w, n_frames_per)
    enc_m, dec_m, pkts_m, dec_arrs_m = bench_pyav_multi(holdout_frames, padded_h, padded_w, n_frames_per)
    enc_d, dec_d, pkts_d, dec_arrs_d = bench_direct(holdout_frames, padded_h, padded_w, n_frames_per)
    enc_dm, dec_dm, pkts_dm, dec_arrs_dm = bench_direct_multi(holdout_frames, padded_h, padded_w, n_frames_per, n_engines=3)

    # ---- quality + size ----
    def reconstruct_all(arrs):
        return [
            reconstruct(arr, basis, *m[:5], padded_h, padded_w)
            for arr, m in zip(arrs, holdout_meta)
        ]

    recon_p = reconstruct_all(dec_arrs_p)
    recon_m = reconstruct_all(dec_arrs_m)
    recon_d = reconstruct_all(dec_arrs_d)
    recon_dm = reconstruct_all(dec_arrs_dm)

    def cos_stats(recons):
        all_cos = []
        for orig, rec in zip(holdout_tensors, recons):
            all_cos.append(cosine_similarity_per_row(orig, rec))
        merged = torch.cat(all_cos)
        return merged.mean().item(), merged.quantile(0.01).item()

    def total_bytes(pkts_list):
        return sum(sum(len(p) for p in pl) for pl in pkts_list)

    cos_p, p1_p = cos_stats(recon_p)
    cos_m, p1_m = cos_stats(recon_m)
    cos_d, p1_d = cos_stats(recon_d)
    cos_dm, p1_dm = cos_stats(recon_dm)

    bytes_p = total_bytes(pkts_p)
    bytes_m = total_bytes(pkts_m)
    bytes_d = total_bytes(pkts_d)
    bytes_dm = total_bytes(pkts_dm)

    raw_bytes = sum(t.numel() * 2 for t in holdout_tensors)  # bf16 = 2 bytes/elem

    print("\n--- summary ---")
    print(f"holdout: {N_HOLDOUT} activations, {total_frames} frames @ {padded_w}x{padded_h} YUV444 QP={QP}")
    print(f"raw input size:    {raw_bytes / 1e6:.2f} MB ({raw_bytes / N_HOLDOUT / 1e6:.2f} MB/activation)\n")

    print(f"{'backend':<22}  {'enc ms':>8}  {'dec ms':>8}  {'enc ms/f':>9}  {'dec ms/f':>9}  {'bytes':>10}  {'ratio':>6}  {'cos':>6}  {'p1cos':>6}")
    for name, enc, dec, b, c, p in [
        ("pyav-single",            enc_p,  dec_p,  bytes_p,  cos_p,  p1_p),
        ("pyav-multi (1eng)",      enc_m,  dec_m,  bytes_m,  cos_m,  p1_m),
        ("direct (1eng, pool=8)",  enc_d,  dec_d,  bytes_d,  cos_d,  p1_d),
        ("direct-multi (3eng x 8)",enc_dm, dec_dm, bytes_dm, cos_dm, p1_dm),
    ]:
        ratio = raw_bytes / b
        print(f"{name:<26}  {enc:>8.1f}  {dec:>8.1f}  {enc / total_frames:>9.3f}  {dec / total_frames:>9.3f}  {b:>10}  {ratio:>5.1f}x  {c:>6.4f}  {p:>6.4f}")

    print(f"\nencode speedup direct vs pyav-single:        {enc_p / enc_d:.2f}x")
    print(f"decode speedup direct vs pyav-single:        {dec_p / dec_d:.2f}x")
    print(f"end-to-end direct vs pyav:                   {(enc_p + dec_p) / (enc_d + dec_d):.2f}x")
    print(f"encode speedup direct-multi vs pyav-single:  {enc_p / enc_dm:.2f}x")
    print(f"decode speedup direct-multi vs pyav-single:  {dec_p / dec_dm:.2f}x")
    print(f"encode speedup direct-multi vs direct-1eng:  {enc_d / enc_dm:.2f}x")
    print(f"decode speedup direct-multi vs direct-1eng:  {dec_d / dec_dm:.2f}x")
    print(f"end-to-end direct-multi vs pyav:             {(enc_p + dec_p) / (enc_dm + dec_dm):.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
