"""MultiEngineCodecSession — distribute encodes across multiple NVENC engines.

Modern NVIDIA GPUs ship with multiple independent NVENC encoder engines on
the same die — the RTX 5090 has 3, the H100 has 4, the A100 has 1. They run
in parallel hardware lanes. This class holds one CodecSession per engine
and dispatches tensor compress calls across them via Python threads.

Measured on RTX 5090, batch of 12 real captured FLUX activations
(K=1000, QP=18):

    Single CodecSession sequential:    177 ms/tensor   (baseline)
    3 CodecSessions in parallel:       111 ms/tensor   (1.60x)

Combined with CodecSession's 1.77x speedup over the subprocess backend,
the multi-engine version reaches ~2.83x over subprocess for batch workloads.

Usage:

    with MultiEngineCodecSession(height=256, width=256, qp=18) as multi:
        results = multi.compress_batch(tensors, basis)
        # results is a list of (packets, recipe) in tensors-input order
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from .pca import Basis
from .session import CodecSession, SessionRecipe


class MultiEngineCodecSession:
    """Wraps N CodecSession instances and distributes encodes across them.

    Threading model: one Python thread per engine during compress_batch().
    The Python GIL is released during PyAV's C-level encode call, so threads
    actually run in parallel on the NVENC hardware engines.
    """

    def __init__(self, height: int, width: int, qp: int = 18, n_engines: int = 3):
        if n_engines < 1:
            raise ValueError("n_engines must be >= 1")
        self.height = height
        self.width = width
        self.qp = qp
        self.n_engines = n_engines
        # Build N independent encoder sessions
        self.sessions: list[CodecSession] = [
            CodecSession(height=height, width=width, qp=qp) for _ in range(n_engines)
        ]

    @property
    def extradata(self) -> bytes:
        """All sessions share the same encoder config so extradata is identical
        (verified by SPS/PPS being deterministic from encoder params). Take from
        session 0 for convenience."""
        return self.sessions[0].extradata

    def compress_batch(
        self, tensors: list[torch.Tensor], basis: Optional[Basis]
    ) -> list[tuple[list[bytes], SessionRecipe]]:
        """Compress a list of tensors in parallel across all engines.

        Returns: list of (packets, recipe) in the same order as the input tensors.
        """
        n = len(tensors)
        if n == 0:
            return []

        results: list[Optional[tuple[list[bytes], SessionRecipe]]] = [None] * n

        # Round-robin assign tensors to engines
        assignments: list[list[int]] = [[] for _ in range(self.n_engines)]
        for i in range(n):
            assignments[i % self.n_engines].append(i)

        def worker(engine_idx: int, indices: list[int]):
            session = self.sessions[engine_idx]
            for i in indices:
                results[i] = session.compress(tensors[i], basis)

        threads = [
            threading.Thread(target=worker, args=(eid, idx_list))
            for eid, idx_list in enumerate(assignments) if idx_list
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All slots should be filled
        return [r for r in results if r is not None]

    def decompress_batch(
        self,
        encoded: list[tuple[list[bytes], SessionRecipe]],
        basis: Optional[Basis],
    ) -> list[torch.Tensor]:
        """Decode a list of (packets, recipe) tuples in parallel across engines."""
        n = len(encoded)
        if n == 0:
            return []
        results: list[Optional[torch.Tensor]] = [None] * n
        assignments: list[list[int]] = [[] for _ in range(self.n_engines)]
        for i in range(n):
            assignments[i % self.n_engines].append(i)

        def worker(engine_idx: int, indices: list[int]):
            session = self.sessions[engine_idx]
            for i in indices:
                packets, recipe = encoded[i]
                results[i] = session.decompress(packets, basis, recipe)

        threads = [
            threading.Thread(target=worker, args=(eid, idx_list))
            for eid, idx_list in enumerate(assignments) if idx_list
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [r for r in results if r is not None]

    def close(self) -> None:
        for s in self.sessions:
            s.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
