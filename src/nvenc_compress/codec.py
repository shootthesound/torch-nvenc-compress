"""FFmpeg subprocess wrapper for NVENC HEVC encode + NVDEC HEVC decode.

This is the slow path. Each call spawns an `ffmpeg` subprocess (50-100 ms
overhead each on Windows) and pipes raw YUV through stdin/stdout. Suitable
for offline analysis, NOT for production-grade offload pipelines. See
docs/parallel_path.md for the planned PyAV / Video Codec SDK fast path.

NVENC HEVC has a minimum frame size of ~144x144 on Blackwell. The pipeline
pads small per-channel frames up to MIN_FRAME_DIM (256x256) before encode;
the codec compresses the zero-padded region almost-for-free.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


FFMPEG = "ffmpeg"
MIN_FRAME_DIM = 256


def find_ffmpeg() -> str:
    path = shutil.which(FFMPEG)
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg` "
            "on Windows, `apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` "
            "on macOS) and ensure it's on your PATH."
        )
    return path


def check_nvenc_available() -> tuple[bool, str]:
    """Returns (ok, message). `ok` is True if hevc_nvenc + hevc_cuvid are both present."""
    try:
        ffmpeg = find_ffmpeg()
    except RuntimeError as e:
        return False, str(e)
    enc = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    dec = subprocess.run(
        [ffmpeg, "-hide_banner", "-decoders"], capture_output=True, text=True
    )
    if "hevc_nvenc" not in enc.stdout:
        return False, "hevc_nvenc encoder not found in ffmpeg build"
    if "hevc_cuvid" not in dec.stdout:
        return False, "hevc_cuvid decoder not found in ffmpeg build"
    return True, "hevc_nvenc + hevc_cuvid available"


def encode_hevc(
    frames: np.ndarray, height: int, width: int, qp: int, out_path: Path
) -> int:
    """Encode raw YUV 4:4:4 frames as HEVC via NVENC.

    Args:
        frames: numpy array shape [N, 3, H, W] uint8 — the YUV planes.
        height, width: spatial dimensions of each frame.
        qp: constant quantization parameter (0-51 for HEVC; 18-26 typical).
        out_path: where to write the .hevc bitstream.

    Returns:
        The byte size of the resulting bitstream (out_path.stat().st_size).
    """
    raw = frames.tobytes()
    cmd = [
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "yuv444p",
        "-s", f"{width}x{height}", "-framerate", "30",
        "-i", "-",
        "-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", str(qp),
        "-pix_fmt", "yuv444p",
        "-f", "hevc",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"hevc_nvenc encode failed (qp={qp})")
    return out_path.stat().st_size


def decode_hevc(
    in_path: Path, num_frames: int, height: int, width: int
) -> np.ndarray:
    """Decode an HEVC bitstream to raw YUV 4:4:4 frames via NVDEC.

    Args:
        in_path: path to a .hevc bitstream produced by `encode_hevc`.
        num_frames: number of frames expected (used for shape/size verification).
        height, width: expected spatial dimensions.

    Returns:
        numpy array shape [num_frames, 3, height, width] uint8.
    """
    cmd = [
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-c:v", "hevc_cuvid", "-i", str(in_path),
        "-f", "rawvideo", "-pix_fmt", "yuv444p",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError("hevc_cuvid decode failed")
    expected = num_frames * 3 * height * width
    if len(proc.stdout) != expected:
        raise RuntimeError(
            f"decoded byte count {len(proc.stdout)} != expected {expected} "
            f"({num_frames} frames * 3 * {height} * {width})"
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
        num_frames, 3, height, width
    ).copy()


def pad_to_min(q: np.ndarray, min_dim: int = MIN_FRAME_DIM) -> tuple[np.ndarray, int, int]:
    """Pad each [H, W] channel up to at least min_dim x min_dim with zeros bottom-right.

    Args:
        q: uint8 array shape [D, H, W].
        min_dim: minimum frame dimension required by NVENC HEVC (~144 on Blackwell;
                 we use 256 as a safe default).

    Returns:
        (padded_q, new_H, new_W).
    """
    d, h, w = q.shape
    target_h = max(h, min_dim)
    target_w = max(w, min_dim)
    if target_h == h and target_w == w:
        return q, target_h, target_w
    out = np.zeros((d, target_h, target_w), dtype=q.dtype)
    out[:, :h, :w] = q
    return out, target_h, target_w


def crop_pad(q: np.ndarray, native_h: int, native_w: int) -> np.ndarray:
    """Inverse of pad_to_min — crop back to native spatial dimensions."""
    return q[:, :native_h, :native_w]


def pack_yuv_frames(q: np.ndarray) -> tuple[np.ndarray, int]:
    """Group D channels into frames of 3 planes each (Y, U, V).

    Args:
        q: uint8 array shape [D, H, W].

    Returns:
        (frames [F, 3, H, W], pad) where pad is the number of zero channels
        appended so that D was divisible by 3.
    """
    d, h, w = q.shape
    pad = (-d) % 3
    if pad:
        q = np.concatenate([q, np.zeros((pad, h, w), dtype=q.dtype)], axis=0)
    return q.reshape(-1, 3, h, w), pad


def unpack_yuv_frames(frames: np.ndarray, original_d: int, pad: int) -> np.ndarray:
    """Inverse of pack_yuv_frames."""
    f, three, h, w = frames.shape
    assert three == 3
    q = frames.reshape(f * 3, h, w)
    if pad:
        q = q[:-pad]
    assert q.shape[0] == original_d
    return q
