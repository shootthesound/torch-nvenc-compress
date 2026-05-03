"""16 — DirectBackend smoke test + bench vs PyAV CodecSession.

Validates that the new DirectBackend matches CodecSession's encode_frames /
decode_frames signature, then times both backends on a small batch.

The bench is honest about what it measures:
  - DirectBackend: per-frame nvEncEncodePicture + lock_bitstream lifecycle
    on a host-memory input buffer (write_input_buffer copies through CPU
    every frame). Zero-copy via nvEncRegisterResource is session 6 work.
  - CodecSession (PyAV): same workload with the existing PyAV path.
  - Both run on the persistent encoder model — init is amortised.

The number that matters here isn't the absolute throughput (small synthetic
frames are dominated by call overhead) but the ratio between backends.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from nvenc_compress.direct.backend import DirectBackend


W, H = 256, 256
QP = 18
N_FRAMES = 16


def gradient_frames(n: int) -> np.ndarray:
    """N frames [N, 3, H, W] uint8 — simple gradient varying with frame index."""
    rr = np.arange(H, dtype=np.int32)[:, None]
    cc = np.arange(W, dtype=np.int32)[None, :]
    out = np.empty((n, 3, H, W), dtype=np.uint8)
    for i in range(n):
        out[i, 0] = ((rr + cc + i * 7) & 0xFF).astype(np.uint8)
    out[:, 1] = 128
    out[:, 2] = 128
    return out


def bench_direct(frames: np.ndarray) -> tuple[float, float, list[bytes], np.ndarray]:
    print("[direct] DirectBackend init...")
    t0 = time.perf_counter()
    backend = DirectBackend(height=H, width=W, qp=QP)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")

    try:
        # Encode
        t0 = time.perf_counter()
        packets = backend.encode_frames(frames)
        enc_ms = (time.perf_counter() - t0) * 1000
        total_bytes = sum(len(p) for p in packets)
        print(f"    encode {len(packets)} packets in {enc_ms:.1f} ms "
              f"({enc_ms / max(1, len(packets)):.2f} ms/frame, "
              f"{total_bytes} bytes total)")

        # Decode
        t0 = time.perf_counter()
        decoded = backend.decode_frames(packets, N_FRAMES)
        dec_ms = (time.perf_counter() - t0) * 1000
        print(f"    decode {N_FRAMES} frames in {dec_ms:.1f} ms "
              f"({dec_ms / N_FRAMES:.2f} ms/frame)")

        return enc_ms, dec_ms, packets, decoded
    finally:
        backend.close()


def bench_direct_zero_copy(frames: np.ndarray) -> tuple[float, list[bytes]]:
    """encode_tensor_frames path — input is a CUDA tensor; the only host
    work is the initial CPU->GPU copy of the test frames (which we do
    once outside the timed region)."""
    print("\n[direct/zerocopy] DirectBackend init...")
    t0 = time.perf_counter()
    backend = DirectBackend(height=H, width=W, qp=QP)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")
    try:
        # Stage frames on GPU once (NOT counted toward encode time — same way
        # PyAV / DirectBackend host paths assume the data is already in RAM)
        cuda_frames = torch.from_numpy(frames).cuda().contiguous()
        torch.cuda.synchronize()

        # Warmup encode (registers CUDA buffer + maps once) — pay one-time cost
        # outside the measurement
        _ = backend.encode_tensor_frames(cuda_frames[:1])

        # Real timed encode of the full batch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        packets = backend.encode_tensor_frames(cuda_frames)
        torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t0) * 1000
        total_bytes = sum(len(p) for p in packets)
        print(f"    encode {len(packets)} packets in {enc_ms:.1f} ms "
              f"({enc_ms / max(1, len(packets)):.2f} ms/frame, "
              f"{total_bytes} bytes total)")
        return enc_ms, packets
    finally:
        backend.close()


def bench_pyav(frames: np.ndarray) -> tuple[float, float, list[bytes], np.ndarray] | None:
    try:
        from nvenc_compress.session import CodecSession
    except Exception as e:
        print(f"[pyav] CodecSession unavailable ({e}) — skipping bench")
        return None

    print("\n[pyav] CodecSession init...")
    t0 = time.perf_counter()
    try:
        session = CodecSession(height=H, width=W, qp=QP)
    except Exception as e:
        print(f"    init failed ({e}) — skipping")
        return None
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    init: {init_ms:.1f} ms")

    try:
        t0 = time.perf_counter()
        packets = session.encode_frames(frames)
        enc_ms = (time.perf_counter() - t0) * 1000
        total_bytes = sum(len(p) for p in packets)
        print(f"    encode {len(packets)} packets in {enc_ms:.1f} ms "
              f"({enc_ms / max(1, len(packets)):.2f} ms/frame, "
              f"{total_bytes} bytes total)")

        t0 = time.perf_counter()
        decoded = session.decode_frames(packets, N_FRAMES)
        dec_ms = (time.perf_counter() - t0) * 1000
        print(f"    decode {N_FRAMES} frames in {dec_ms:.1f} ms "
              f"({dec_ms / N_FRAMES:.2f} ms/frame)")
        return enc_ms, dec_ms, packets, decoded
    finally:
        session.close()


def main() -> int:
    print(f"DirectBackend bench — {N_FRAMES} frames {W}x{H} YUV444 QP={QP}\n")

    print("Building synthetic gradient frames...")
    frames = gradient_frames(N_FRAMES)
    print(f"  shape={frames.shape} dtype={frames.dtype}\n")

    direct_result = bench_direct(frames)
    zero_result = bench_direct_zero_copy(frames)
    pyav_result = bench_pyav(frames)

    # Verify direct round-trip is correct
    enc_d, dec_d, pkts_d, decoded_d = direct_result
    diffs = np.abs(decoded_d.astype(np.int32) - frames.astype(np.int32))
    print(f"\n[direct] round-trip diff: max={diffs.max()} mean={diffs.mean():.3f}")

    enc_z, pkts_z = zero_result

    # Quick correctness check: decode the zero-copy bitstream and verify
    # it round-trips against the same input
    print("\n[direct/zerocopy] decode-back sanity check...")
    sanity_backend = DirectBackend(height=H, width=W, qp=QP)
    try:
        decoded_z = sanity_backend.decode_frames(pkts_z, N_FRAMES)
    finally:
        sanity_backend.close()
    diffs_z = np.abs(decoded_z.astype(np.int32) - frames.astype(np.int32))
    print(f"    zero-copy round-trip diff: max={diffs_z.max()} mean={diffs_z.mean():.3f}")

    print("\n--- summary ---")
    print(f"direct encode (host buf):    {enc_d:.1f} ms  ({enc_d / N_FRAMES:.2f} ms/frame)")
    print(f"direct encode (zero-copy):   {enc_z:.1f} ms  ({enc_z / N_FRAMES:.2f} ms/frame)")
    print(f"direct decode:               {dec_d:.1f} ms  ({dec_d / N_FRAMES:.2f} ms/frame)")
    if pyav_result is not None:
        enc_p, dec_p, _, decoded_p = pyav_result
        print(f"pyav encode:                 {enc_p:.1f} ms  ({enc_p / N_FRAMES:.2f} ms/frame)")
        print(f"pyav decode:                 {dec_p:.1f} ms  ({dec_p / N_FRAMES:.2f} ms/frame)")
        if enc_d > 0:
            print(f"\nencode speedup (pyav / direct host):     {enc_p / enc_d:.2f}x")
            print(f"encode speedup (pyav / direct zero-copy):{enc_p / enc_z:.2f}x")
            print(f"encode speedup (host / zero-copy):       {enc_d / enc_z:.2f}x")
            print(f"decode speedup (pyav / direct):          {dec_p / dec_d:.2f}x")
        # Sanity: both should produce visually-identical reconstructions
        diffs_p = np.abs(decoded_p.astype(np.int32) - frames.astype(np.int32))
        print(f"pyav round-trip diff:   max={diffs_p.max()} mean={diffs_p.mean():.3f}")
    print(f"zero-copy bitstream bytes total: {sum(len(p) for p in pkts_z)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
