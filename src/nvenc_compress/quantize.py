"""Per-channel uint8 quantisation.

Each channel is mapped to [0, 255] using its own min/max. This preserves
relative dynamic range across channels (some channels have std=0.1, others
std=10; treating them all on a global scale would destroy the small-std ones).
The cost is per-channel scale + offset metadata (8 bytes per channel — trivial).
"""

from __future__ import annotations

import numpy as np
import torch


def per_channel_quantise(X: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise [D, H, W] fp32 -> uint8 [D, H, W] with per-channel scale/offset.

    Args:
        X: torch.Tensor shape [D, H, W] in any float dtype on any device.

    Returns:
        (q [D, H, W] uint8 numpy array, scale [D] float32 numpy array,
         offset [D] float32 numpy array).
    """
    if X.ndim != 3:
        raise ValueError(f"expected [D, H, W], got shape {tuple(X.shape)}")
    d, h, w = X.shape
    flat = X.reshape(d, -1)
    mn = flat.min(dim=1).values
    mx = flat.max(dim=1).values
    rng = (mx - mn).clamp_min(1e-12)
    scale = rng / 255.0
    offset = mn
    q = ((X - offset[:, None, None]) / scale[:, None, None]).round().clamp(0, 255).to(torch.uint8)
    return q.cpu().numpy(), scale.cpu().numpy(), offset.cpu().numpy()


def per_channel_dequantise(
    q: np.ndarray, scale: np.ndarray, offset: np.ndarray
) -> torch.Tensor:
    """Inverse of per_channel_quantise.

    Args:
        q: uint8 numpy array shape [D, H, W].
        scale, offset: float32 numpy arrays shape [D].

    Returns:
        torch.Tensor shape [D, H, W] fp32 (on CPU).
    """
    return torch.from_numpy(
        q.astype(np.float32) * scale[:, None, None] + offset[:, None, None]
    )
