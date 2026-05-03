"""19 — Diagnose the DirectBackend vs PyAV bitstream / quality divergence.

poc/18 surfaced an unexpected result: DirectBackend produces higher
reconstruction quality (cos 0.9881) than PyAV CodecSession (cos 0.9731)
on identical input frames at the same QP=18, with slightly smaller
bitstream. Same hevc_nvenc, same yuv444p, same QP. Something is
configured differently.

This PoC:
  1. Encodes ONE 256x256 YUV444 frame (synthetic gradient) via both backends.
  2. Dumps the raw bitstreams to disk.
  3. Runs ffprobe on each and prints the human-readable stream metadata.
  4. Locates HEVC NAL units (VPS / SPS / PPS / IDR slice) by scanning for
     Annex-B start codes; reports type + length + first bytes of each.
  5. Diffs the IDR slice bytes between the two encoders to confirm whether
     the divergence is in the parameter sets or the actual coded slice data.
  6. Decodes both back through DirectBackend's NVDEC and reports max-abs
     and mean-abs diff vs the original input — direct apples-to-apples
     quality comparison on a single frame.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from nvenc_compress.direct.backend import DirectBackend
from nvenc_compress.session import CodecSession


W, H = 256, 256
QP = 18


def make_frame() -> np.ndarray:
    """One 256x256x3 YUV444 frame, [3, H, W] uint8 — diagonal gradient."""
    f = np.zeros((3, H, W), dtype=np.uint8)
    rr = np.arange(H, dtype=np.int32)[:, None]
    cc = np.arange(W, dtype=np.int32)[None, :]
    f[0] = ((rr + cc) & 0xFF).astype(np.uint8)
    f[1] = 128
    f[2] = 128
    return f


def encode_via_direct(frame: np.ndarray) -> bytes:
    backend = DirectBackend(height=H, width=W, qp=QP)
    try:
        cuda_frame = torch.from_numpy(frame[None]).cuda().contiguous()
        torch.cuda.synchronize()
        # Warmup so the staging-buffer registration cost is paid first
        _ = backend.encode_tensor_frames(cuda_frame)
        # Real encode
        pkts = backend.encode_tensor_frames(cuda_frame)
        return b"".join(pkts)
    finally:
        backend.close()


def encode_via_pyav(frame: np.ndarray) -> bytes:
    sess = CodecSession(height=H, width=W, qp=QP)
    try:
        # CodecSession.encode_frames takes [N, 3, H, W]
        pkts = sess.encode_frames(frame[None])
        # PyAV's first packet may not include extradata inline since session
        # ships it separately. Concatenate extradata + packets to make the
        # bitstream self-decodable for ffprobe.
        return sess.extradata + b"".join(pkts)
    finally:
        sess.close()


def find_nal_units(bitstream: bytes) -> list[tuple[int, int, int]]:
    """Find Annex-B NAL units. Returns list of (offset, length, nal_type)."""
    out = []
    n = len(bitstream)
    starts = []
    i = 0
    while i < n - 3:
        if bitstream[i:i+4] == b"\x00\x00\x00\x01":
            starts.append((i, 4))
            i += 4
        elif bitstream[i:i+3] == b"\x00\x00\x01":
            starts.append((i, 3))
            i += 3
        else:
            i += 1
    starts.append((n, 0))
    for k in range(len(starts) - 1):
        off, sc_len = starts[k]
        nxt = starts[k + 1][0]
        nal_byte = bitstream[off + sc_len]
        # HEVC NAL type is bits 1-6 of the header byte (forbidden=bit0=0)
        nal_type = (nal_byte >> 1) & 0x3F
        out.append((off, nxt - off, nal_type))
    return out


HEVC_NAL_NAMES = {
    0: "TRAIL_N", 1: "TRAIL_R", 19: "IDR_W_RADL", 20: "IDR_N_LP",
    21: "CRA_NUT", 32: "VPS_NUT", 33: "SPS_NUT", 34: "PPS_NUT",
    35: "AUD_NUT", 36: "EOS_NUT", 37: "EOB_NUT", 38: "FD_NUT",
    39: "PREFIX_SEI_NUT", 40: "SUFFIX_SEI_NUT",
}


def describe_nals(bitstream: bytes, label: str) -> None:
    print(f"\n[{label}] bitstream size: {len(bitstream)} bytes, NAL units:")
    nals = find_nal_units(bitstream)
    for off, length, ntype in nals:
        name = HEVC_NAL_NAMES.get(ntype, f"type_{ntype}")
        first = bitstream[off:off + min(length, 24)].hex()
        print(f"  off={off:>6}  len={length:>5}  type={ntype:<3} ({name:<14}) first_bytes={first}")


def ffprobe_stream(path: Path) -> str:
    """Run ffprobe and return the human-readable stream summary."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format",
             "-of", "default=noprint_wrappers=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout + ("\n[STDERR]\n" + out.stderr if out.stderr else "")
    except Exception as e:
        return f"ffprobe failed: {e!r}"


def round_trip_via_direct(bitstream: bytes) -> np.ndarray:
    """Decode a single-frame HEVC bitstream via DirectBackend's NVDEC path."""
    backend = DirectBackend(height=H, width=W, qp=QP)
    try:
        return backend.decode_frames([bitstream], n_frames=1)[0]
    finally:
        backend.close()


def main() -> int:
    print(f"DirectBackend vs PyAV bitstream diagnostic — single {W}x{H} YUV444 frame at QP={QP}\n")

    frame = make_frame()
    print(f"input frame shape: {frame.shape} dtype={frame.dtype}")
    print(f"input checksum: {hash(frame.tobytes()) & 0xFFFFFFFF:08x}\n")

    # Encode through both backends
    print("encoding via DirectBackend...")
    bs_d = encode_via_direct(frame)
    print(f"  {len(bs_d)} bytes\n")

    print("encoding via PyAV CodecSession (with extradata prepended)...")
    bs_p = encode_via_pyav(frame)
    print(f"  {len(bs_p)} bytes\n")

    # Save and ffprobe
    tmpdir = Path(tempfile.gettempdir())
    p_direct = tmpdir / "direct_one_frame.hevc"
    p_pyav = tmpdir / "pyav_one_frame.hevc"
    p_direct.write_bytes(bs_d)
    p_pyav.write_bytes(bs_p)

    print(f"saved to {p_direct} and {p_pyav}\n")
    print("--- ffprobe DIRECT ---")
    print(ffprobe_stream(p_direct))
    print("--- ffprobe PYAV ---")
    print(ffprobe_stream(p_pyav))

    # NAL unit breakdown
    describe_nals(bs_d, "direct")
    describe_nals(bs_p, "pyav")

    # Quality round-trip via direct decoder for both
    print("\n--- round-trip quality (decoded via DirectBackend NVDEC) ---")
    try:
        recon_d = round_trip_via_direct(bs_d)[0]
        diff_d = np.abs(recon_d.astype(np.int32) - frame.astype(np.int32))
        print(f"DIRECT: max abs diff {diff_d.max()}, mean {diff_d.mean():.4f}")
    except Exception as e:
        print(f"DIRECT decode failed: {e!r}")

    try:
        recon_p = round_trip_via_direct(bs_p)[0]
        diff_p = np.abs(recon_p.astype(np.int32) - frame.astype(np.int32))
        print(f"PYAV:   max abs diff {diff_p.max()}, mean {diff_p.mean():.4f}")
    except Exception as e:
        print(f"PYAV decode failed: {e!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
