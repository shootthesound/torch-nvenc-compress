"""End-to-end compress/decompress pipeline.

The full path:

    tensor [T, D]              # input (rows are samples, cols are channels)
        -> PCA project         # rotate into basis where signal is concentrated
        -> truncate to top K   # drop low-variance dimensions (lossy step)
        -> reshape to [K, H, W]  # arbitrary 2D reshape for codec consumption
        -> per-channel uint8   # quantise channels to 8 bits each
        -> pad each channel to MIN_FRAME_DIM  # NVENC HEVC needs >=144x144
        -> pack channels as YUV 4:4:4 frames (3 channels per frame)
        -> NVENC HEVC encode   # the dedicated GPU silicon does its thing
        -> bitstream (the wire format)

Inverse path is symmetric.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .codec import (
    pack_yuv_frames, unpack_yuv_frames,
    pad_to_min, crop_pad, MIN_FRAME_DIM,
)
from . import codec as _codec_subprocess
from .pca import Basis
from .quantize import per_channel_quantise, per_channel_dequantise


def _get_backend(name: str):
    """Returns (encode_hevc, decode_hevc) functions for the named backend."""
    if name == "subprocess":
        return _codec_subprocess.encode_hevc, _codec_subprocess.decode_hevc
    if name == "pyav":
        from . import codec_pyav as _codec_pyav
        return _codec_pyav.encode_hevc, _codec_pyav.decode_hevc
    raise ValueError(f"unknown backend {name!r}; expected 'subprocess' or 'pyav'")


@dataclass
class Recipe:
    """Per-tensor metadata needed for decompression. Sent alongside the bitstream.

    Tiny — a few hundred bytes per tensor.
    """
    seq_len: int
    K: int
    side_h: int
    side_w: int
    pad_channels: int
    padded_h: int
    padded_w: int
    n_frames: int
    scale: np.ndarray         # [K] float32
    offset: np.ndarray        # [K] float32

    def serialised_metadata_bytes(self) -> int:
        """Estimate the bytes overhead of shipping this recipe (scale + offset dominate)."""
        return self.scale.nbytes + self.offset.nbytes + 32  # +32 for the small ints


def _reshape_for_codec(R: torch.Tensor, K: int) -> tuple[torch.Tensor, int, int]:
    """Reshape [T, K] into [K, side_h, side_w] for codec ingest. Pads T up to a
    perfect square if needed."""
    T = R.shape[0]
    side = int(math.isqrt(T))
    while side * side < T:
        side += 1
    pad_rows = side * side - T
    if pad_rows:
        R = torch.cat([R, torch.zeros(pad_rows, K, device=R.device, dtype=R.dtype)], dim=0)
    return R.T.reshape(K, side, side).contiguous(), side, side


def compress(
    tensor: torch.Tensor,
    basis: Optional[Basis],
    qp: int = 18,
    work_dir: Optional[Path] = None,
    backend: str = "subprocess",
) -> tuple[bytes, Recipe]:
    """Compress one [T, D] tensor through the full pipeline.

    Args:
        tensor: shape [T, D] (rows are samples / spatial positions / token positions).
        basis: Basis to project through (PCA + rank truncation). Pass None to skip
               the PCA step entirely (full-rank codec only).
        qp: NVENC HEVC constant quantization parameter (10-26 typical).
        work_dir: where to write the temporary bitstream. Default: system temp.
        backend: codec backend — 'subprocess' (default, FFmpeg subprocess) or
                 'pyav' (in-process FFmpeg via PyAV; eliminates subprocess overhead).

    Returns:
        (bitstream_bytes, recipe). Send both to the receiver.
    """
    if tensor.dim() != 2:
        raise ValueError(f"expected [T, D] 2D tensor, got shape {tuple(tensor.shape)}")
    work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    work_dir.mkdir(parents=True, exist_ok=True)
    encode_hevc, _ = _get_backend(backend)

    if basis is not None:
        R = basis.project(tensor)            # [T, K]
        K = basis.K
    else:
        R = tensor                            # [T, D] used as-is
        K = tensor.shape[1]

    R_chw, side_h, side_w = _reshape_for_codec(R, K)
    R_chw_cpu = R_chw.cpu()

    q, scale, offset = per_channel_quantise(R_chw_cpu)
    q_padded, padded_h, padded_w = pad_to_min(q, MIN_FRAME_DIM)
    frames, pad_channels = pack_yuv_frames(q_padded)
    n_frames = frames.shape[0]

    bitstream_path = work_dir / f"_nvenc_compress_tmp_{id(tensor)}.hevc"
    encode_hevc(frames, padded_h, padded_w, qp, bitstream_path)
    data = bitstream_path.read_bytes()
    bitstream_path.unlink(missing_ok=True)

    recipe = Recipe(
        seq_len=tensor.shape[0],
        K=K,
        side_h=side_h,
        side_w=side_w,
        pad_channels=pad_channels,
        padded_h=padded_h,
        padded_w=padded_w,
        n_frames=n_frames,
        scale=scale,
        offset=offset,
    )
    return data, recipe


def decompress(
    bitstream: bytes,
    basis: Optional[Basis],
    recipe: Recipe,
    work_dir: Optional[Path] = None,
    backend: str = "subprocess",
) -> torch.Tensor:
    """Inverse of compress. Returns the reconstructed [T, D] tensor on basis.device
    (or CPU if basis is None).

    backend: 'subprocess' or 'pyav'. Should match what was used in compress().
    """
    work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    work_dir.mkdir(parents=True, exist_ok=True)
    _, decode_hevc = _get_backend(backend)
    bitstream_path = work_dir / f"_nvenc_decompress_tmp_{id(bitstream)}.hevc"
    bitstream_path.write_bytes(bitstream)
    decoded = decode_hevc(bitstream_path, recipe.n_frames, recipe.padded_h, recipe.padded_w)
    bitstream_path.unlink(missing_ok=True)

    q_padded_recon = unpack_yuv_frames(decoded, recipe.K, recipe.pad_channels)
    q_recon = crop_pad(q_padded_recon, recipe.side_h, recipe.side_w)
    R_chw_recon = per_channel_dequantise(q_recon, recipe.scale, recipe.offset)
    R_recon = R_chw_recon.reshape(recipe.K, -1).T          # [side_h*side_w, K]
    R_recon = R_recon[: recipe.seq_len]                    # crop padded rows

    if basis is not None:
        device = basis.V_K.device
        R_recon = R_recon.to(device)
        return basis.invert(R_recon)
    return R_recon
