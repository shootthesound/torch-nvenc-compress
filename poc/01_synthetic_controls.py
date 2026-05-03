"""01 — Pipeline sanity check on synthetic data.

Pushes tensors of KNOWN compressibility through the full pipeline (no PCA,
just per-channel uint8 + NVENC HEVC) and prints the resulting ratios. Tells
you the toolchain works before you commit to multi-GB model downloads.

Reference numbers we expect (anything wildly different = something is wrong):

  zeros                  ~600x   cos ~1.0 (or 0/0 = NaN)
  one_image_replicated   ~600x   cos ~0.999
  smooth_per_channel      ~75x   cos ~0.999
  noise_plus_lowfreq       ~6x   cos ~0.99
  gaussian_noise           ~4x   cos ~0.99
"""

from __future__ import annotations

import numpy as np
import torch

from nvenc_compress import compress, decompress, metrics


# Mimic a Flux-class activation tensor shape: 4096 channels at 64x64
# laid out as [T=4096, D=4096] (each spatial position is a sample).
T = 4096
D = 4096
SIDE = 64                                 # T = SIDE*SIDE


def synth_zeros() -> torch.Tensor:
    return torch.zeros(T, D)


def synth_one_image_replicated(rng: np.random.Generator) -> torch.Tensor:
    """Same low-freq pattern replicated across all D channels => very compressible."""
    y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32) / max(SIDE, SIDE)
    base = (np.sin(x * 6.0) * np.cos(y * 4.0)
            + 0.5 * np.sin((x + y) * 3.0))
    base = (base - base.mean()) / (base.std() + 1e-12)
    arr = np.broadcast_to(base.reshape(-1), (D, T)).T.copy()      # [T, D]
    return torch.from_numpy(arr)


def synth_smooth_per_channel(rng: np.random.Generator) -> torch.Tensor:
    """Each channel a distinct low-frequency sinusoidal pattern."""
    y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32) / max(SIDE, SIDE)
    out = np.zeros((D, SIDE, SIDE), dtype=np.float32)
    for d in range(D):
        ax = rng.uniform(1, 6)
        ay = rng.uniform(1, 6)
        ph = rng.uniform(0, 2 * np.pi)
        out[d] = np.sin(x * ax + ph) * np.cos(y * ay)
    out = (out - out.mean(axis=(1, 2), keepdims=True)) / (
        out.std(axis=(1, 2), keepdims=True) + 1e-12
    )
    return torch.from_numpy(out.reshape(D, -1).T)                  # [T, D]


def synth_noise_plus_lowfreq(rng: np.random.Generator) -> torch.Tensor:
    smooth = synth_smooth_per_channel(rng)
    noise = torch.from_numpy(rng.standard_normal(smooth.shape, dtype=np.float32) * 0.3)
    return smooth + noise


def synth_gaussian_noise(rng: np.random.Generator) -> torch.Tensor:
    return torch.from_numpy(rng.standard_normal((T, D), dtype=np.float32))


def main() -> None:
    rng = np.random.default_rng(0)
    sources = [
        ("zeros",                  synth_zeros()),
        ("one_image_replicated",   synth_one_image_replicated(rng)),
        ("smooth_per_channel",     synth_smooth_per_channel(rng)),
        ("noise_plus_lowfreq",     synth_noise_plus_lowfreq(rng)),
        ("gaussian_noise",         synth_gaussian_noise(rng)),
    ]

    print(f"Pipeline sanity (no PCA), tensor shape [T={T}, D={D}], NVENC HEVC at QP=18\n")
    print(f"{'name':<25s}  {'ratio':>10s}  {'cos_mean':>10s}  {'cos_p1':>10s}  {'mae':>10s}")
    for name, tensor in sources:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_dev = tensor.to(device)
        bytes_orig_fp16 = tensor.numel() * 2
        try:
            data, recipe = compress(tensor_dev, basis=None, qp=18)
            recon = decompress(data, basis=None, recipe=recipe).cpu()
        except RuntimeError as e:
            print(f"{name:<25s}  FAILED: {e}")
            continue
        ratio = bytes_orig_fp16 / len(data)
        # metrics expects [B, D, ...] with channels as second axis. We have [T, D].
        # Treat each channel as a sequence of T values.
        m = metrics.compute(
            tensor.T.unsqueeze(0).unsqueeze(-1),
            recon.T.unsqueeze(0).unsqueeze(-1),
        )
        print(f"{name:<25s}  {ratio:>9.2f}x  {m['cos_mean']:>10.4f}  "
              f"{m['cos_p1']:>10.4f}  {m['mae']:>10.4f}")


if __name__ == "__main__":
    main()
