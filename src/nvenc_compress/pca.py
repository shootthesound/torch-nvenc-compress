"""PCA basis: build a shared rotation V_K from many sample tensors.

Treats each spatial position as a sample of a D-dim distribution. The shared
basis is the top-K eigenvectors of the averaged channel covariance across
all samples. This is the LoRA-style "ship V with the model" pattern: V is
computed once offline and applied at runtime to every activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class Basis:
    """A PCA basis ready for project / invert.

    Attributes:
        mean: tensor shape [D] — channel-wise mean of the calibration set.
        V_K:  tensor shape [D, K] — top-K eigenvectors of the averaged covariance,
              sorted by descending eigenvalue. Columns are orthonormal.
    """

    mean: torch.Tensor
    V_K: torch.Tensor

    @property
    def D(self) -> int:
        return self.V_K.shape[0]

    @property
    def K(self) -> int:
        return self.V_K.shape[1]

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """[T, D] -> [T, K] in this basis."""
        return (X - self.mean) @ self.V_K

    def invert(self, R: torch.Tensor) -> torch.Tensor:
        """[T, K] -> [T, D] back in the original basis."""
        return R @ self.V_K.T + self.mean

    def to(self, device: torch.device | str) -> "Basis":
        return Basis(self.mean.to(device), self.V_K.to(device))


def build_shared_basis(samples: Iterable[torch.Tensor], K: int) -> Basis:
    """Compute a shared PCA basis from the average channel covariance over many samples.

    Args:
        samples: iterable of [T_i, D] tensors. Each tensor's rows are samples
                 (e.g., spatial positions of an activation, or token positions
                 of a KV cache). All tensors must share the same D.
        K: number of top eigenvectors to keep. Smaller K = lower bandwidth at
           inference, lower quality.

    Returns:
        Basis with mean and V_K on the device of the first sample.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("need at least one sample to build a basis")
    device = samples[0].device
    D = samples[0].shape[1]
    means = []
    covs = []
    for X in samples:
        if X.shape[1] != D:
            raise ValueError(f"sample dim mismatch: {X.shape[1]} vs {D}")
        T = X.shape[0]
        mn = X.mean(dim=0)
        Xc = X - mn
        cov = (Xc.T @ Xc) / max(T - 1, 1)
        means.append(mn)
        covs.append(cov)
    avg_mean = torch.stack(means).mean(dim=0)
    avg_cov = torch.stack(covs).mean(dim=0)
    eigvals, eigvecs = torch.linalg.eigh(avg_cov)
    order = torch.argsort(eigvals, descending=True)
    V_full = eigvecs[:, order]
    return Basis(mean=avg_mean, V_K=V_full[:, :K].contiguous())


class LeaveOneOutBasisBuilder:
    """Efficient leave-one-out basis builder.

    For each of N samples, we want a basis built from the OTHER N-1 samples.
    Naive: rebuild from scratch N times (O(N^2 * D^2)). Optimised: precompute
    per-sample (mean, cov), keep running totals, subtract held-out and
    eigendecompose. O(N * D^2).
    """

    def __init__(self, samples: Iterable[torch.Tensor]):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("need at least one sample")
        self.device = self.samples[0].device
        self.D = self.samples[0].shape[1]
        self._means: list[torch.Tensor] = []
        self._covs: list[torch.Tensor] = []
        for X in self.samples:
            T = X.shape[0]
            mn = X.mean(dim=0)
            Xc = X - mn
            cov = (Xc.T @ Xc) / max(T - 1, 1)
            self._means.append(mn)
            self._covs.append(cov)
        self._total_mean = torch.stack(self._means).sum(dim=0)
        self._total_cov = torch.stack(self._covs).sum(dim=0)
        self.N = len(self.samples)

    def basis_excluding(self, held_idx: int, K: int) -> Basis:
        """Returns the shared basis built from all samples EXCEPT held_idx."""
        if not 0 <= held_idx < self.N:
            raise IndexError(held_idx)
        train_mean = (self._total_mean - self._means[held_idx]) / (self.N - 1)
        train_cov = (self._total_cov - self._covs[held_idx]) / (self.N - 1)
        eigvals, eigvecs = torch.linalg.eigh(train_cov)
        order = torch.argsort(eigvals, descending=True)
        V_K = eigvecs[:, order][:, :K].contiguous()
        return Basis(mean=train_mean, V_K=V_K)
