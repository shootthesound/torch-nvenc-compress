"""nvenc_compress — compress neural network intermediate state with NVENC HEVC.

Public API:

    from nvenc_compress import (
        Basis,                  # PCA basis (mean + V_K)
        build_shared_basis,     # build a shared PCA basis from many sample tensors
        compress,               # compress a tensor through PCA + quant + NVENC HEVC
        decompress,             # inverse path: NVDEC + dequant + inverse PCA
        metrics,                # cos / mae / max_abs / ratio helpers
    )

Typical use:

    # offline: calibrate the basis from N>=30 representative samples
    basis = build_shared_basis(samples, K=1000)

    # runtime: per-tensor compress / decompress
    compressed_bytes, recipe = compress(activation, basis, qp=18)
    reconstructed = decompress(compressed_bytes, basis, recipe)
"""

from .pca import Basis, build_shared_basis
from .pipeline import compress, decompress
from . import metrics
from . import codec
from . import quantize

# Lazy import — session.py requires PyAV
def __getattr__(name):
    if name == "CodecSession":
        from .session import CodecSession
        return CodecSession
    if name == "SessionRecipe":
        from .session import SessionRecipe
        return SessionRecipe
    if name == "MultiEngineCodecSession":
        from .multi_session import MultiEngineCodecSession
        return MultiEngineCodecSession
    raise AttributeError(f"module 'nvenc_compress' has no attribute {name!r}")

__all__ = [
    "Basis",
    "build_shared_basis",
    "compress",
    "decompress",
    "CodecSession",
    "SessionRecipe",
    "MultiEngineCodecSession",
    "metrics",
    "codec",
    "quantize",
]

__version__ = "0.1.0"
