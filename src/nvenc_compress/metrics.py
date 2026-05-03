"""Reconstruction quality metrics.

Per-channel cosine similarity is the primary metric — diffusion and LLM
literature both use it as the standard "did this perturbation move the
representation in a meaningful direction" check.
"""

from __future__ import annotations

import torch


def compute(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute reconstruction metrics.

    Args:
        original, reconstructed: matching shape, dtype float. Reduces last
                                 dimensions per-channel, treating the first
                                 dim of the channel axis as channels.

                                 Shapes accepted: [B, D, ...] or [D, ...].
                                 Channel axis is dim 0 if 3D, dim 1 if 4D+.

    Returns:
        dict with keys:
            mse, mae, max_abs                    — global error magnitudes
            cos_mean, cos_min, cos_p1            — per-channel cosine similarity
            worst_max_abs_over_std               — outlier flag (max-abs error /
                                                   that channel's stddev)
            channels_below_0.95_cos              — count of bad channels
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {reconstructed.shape}")
    # Flatten to [D, X]
    if original.dim() >= 4:
        a = original.reshape(original.shape[0] * original.shape[1], -1) \
                if original.shape[0] != 1 else original[0].reshape(original.shape[1], -1)
        b = reconstructed.reshape(reconstructed.shape[0] * reconstructed.shape[1], -1) \
                if reconstructed.shape[0] != 1 else reconstructed[0].reshape(reconstructed.shape[1], -1)
    elif original.dim() == 3:
        a = original.reshape(original.shape[0], -1)
        b = reconstructed.reshape(reconstructed.shape[0], -1)
    else:
        raise ValueError(f"need 3D or 4D tensor, got {original.dim()}D")

    a = a.float()
    b = b.float()
    diff = a - b
    abs_diff = diff.abs()

    a_norm = a.norm(dim=1).clamp_min(1e-12)
    b_norm = b.norm(dim=1).clamp_min(1e-12)
    cos = (a * b).sum(dim=1) / (a_norm * b_norm)

    chan_std = a.std(dim=1).clamp_min(1e-12)
    chan_max_abs = abs_diff.max(dim=1).values

    return {
        "mse": (diff ** 2).mean().item(),
        "mae": abs_diff.mean().item(),
        "max_abs": abs_diff.max().item(),
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "cos_p1": torch.quantile(cos, 0.01).item(),
        "worst_max_abs_over_std": (chan_max_abs / chan_std).max().item(),
        "channels_below_0.95_cos": int((cos < 0.95).sum().item()),
    }
