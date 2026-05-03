"""n4 — PyNvVideoCodec persistent session: blocked by hardware pipeline delay.

Hypothesis: NVIDIA's official PyNvVideoCodec gives lower-level access to
NVENC than PyAV. With persistent encoder context + GPU-input zero-copy,
should give a meaningful speedup over PyAV CodecSession for batch workloads.

Reality: NVENC's hardware encoding pipeline holds 2 frames in flight at
all times. Even with bf=0, delay=0, rc-lookahead=0, low-latency presets
and tunes, every config leaves 2 frames buffered. EndEncode() only flushes
1 of them. So with a persistent encoder across multiple tensor encodes,
the last 2 frames of each tensor leak into the next tensor's packet
stream — making per-tensor packet boundaries non-deterministic.

PyAV doesn't have this problem because its FFmpeg layer handles the
pipeline flush internally on each encode call. PyAV CodecSession with bf=0
emits exactly N packets for N input frames.

This script reproduces the issue: encodes 10 frames, gets only 8 packets
back. Even calling EndEncode adds only 1 more (9/10).

We also fix the dlpack version mismatch with a monkey-patch — that part
works fine. So PyNvVideoCodec IS usable per-call (as a third backend
option), but doesn't unlock further speedup over PyAV CodecSession in
the persistent-context configuration.

For the multi-GPU NVLink-class wins, we'd need direct Video Codec SDK
access via cffi, where we have low-level control over the encoder
pipeline including explicit flush semantics.
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    try:
        import PyNvVideoCodec as nvc
    except ImportError:
        print("PyNvVideoCodec not installed. To reproduce this spike: pip install PyNvVideoCodec")
        return
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Monkey-patch fix for dlpack version mismatch
    _orig = torch.Tensor.__dlpack__
    def _patched(self, stream=None, *a, **kw):
        return _orig(self, stream=stream) if stream is not None else _orig(self)
    torch.Tensor.__dlpack__ = _patched

    H, W = 256, 256
    torch.cuda.manual_seed(0)

    enc = nvc.CreateEncoder(
        width=W, height=H, fmt='YUV444', usecpuinputbuffer=False,
        codec='hevc', preset='P4', rc='constqp', qp='18',
        bf=0, repeatspspps=1,
    )
    FORCE_IDR_FLAG = (
        int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR)
        | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
    )

    print("Setup: persistent NVENC encoder, GPU input, bf=0, repeatspspps=1\n")

    # Warmup
    warmup = torch.zeros(3 * H, W, dtype=torch.uint8, device='cuda')
    enc.Encode(warmup, FORCE_IDR_FLAG)

    # Submit 10 frames, count packets returned synchronously
    print("Submitting 10 frames (frame 0 forced IDR + SPS/PPS):")
    frames = torch.randint(0, 255, size=(10, 3 * H, W), dtype=torch.uint8, device='cuda').contiguous()
    pkts = []
    for i in range(10):
        flag = FORCE_IDR_FLAG if i == 0 else 0
        pkt = enc.Encode(frames[i].contiguous(), flag)
        pkts.append(len(bytes(pkt)) if pkt else 0)
    print(f"  packets per submitted frame: {pkts}")
    nonzero = sum(1 for x in pkts if x > 0)
    print(f"  non-empty: {nonzero}/10  (2-frame pipeline delay)")

    # Try EndEncode flush
    flush = enc.EndEncode()
    flush_size = len(bytes(flush)) if flush else 0
    print(f"\n  EndEncode flush returned: {flush_size} bytes")
    print(f"  Total accounted for: {nonzero}/10 + 1 flush packet = {nonzero+1}/10 frames")
    print(f"  Missing: {10 - (nonzero + 1)} frame's worth of packet data is")
    print(f"  permanently in the encoder's pipeline (no API to extract without close)")

    print("\nVerdict:")
    print("  - dlpack monkey-patch works; GPU input mode works")
    print("  - But the 2-frame pipeline delay makes per-tensor packet boundaries")
    print("    non-deterministic in a persistent-encoder configuration.")
    print("  - PyAV CodecSession (bf=0) handles the pipeline cleanly inside FFmpeg")
    print("    and is the right backend for batch workloads.")
    print("  - For finer-grained control of the NVENC pipeline (and the multi-GPU")
    print("    NVLink-class wins), the direct Video Codec SDK via cffi is the path.")


if __name__ == "__main__":
    main()
