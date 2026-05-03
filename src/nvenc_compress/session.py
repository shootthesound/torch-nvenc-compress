"""Persistent codec session — keep one PyAV codec context open across many
encode calls, amortising the ~80-100 ms NVENC init cost over batch workloads.

For batches of N>=5 tensors at the same resolution, this is the best codec
backend we ship. For one-off single-tensor encodes, use the standard
compress() / decompress() functions instead — the session has more setup.

Usage:

    from nvenc_compress import CodecSession, build_shared_basis

    basis = build_shared_basis(samples, K=1000)
    side = 256                                # frame H = W after padding

    with CodecSession(height=side, width=side, qp=18) as session:
        encoded = []
        for tensor in batch_of_tensors:
            packets, recipe = session.compress(tensor, basis)
            encoded.append((packets, recipe))

        # ... transmission ...

        reconstructed = []
        for packets, recipe in encoded:
            t = session.decompress(packets, basis, recipe)
            reconstructed.append(t)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import numpy as np
import torch

from .codec import pack_yuv_frames, unpack_yuv_frames, pad_to_min, crop_pad, MIN_FRAME_DIM
from .pca import Basis
from .quantize import per_channel_quantise, per_channel_dequantise


def _import_av():
    try:
        import av
        return av
    except ImportError as e:
        raise RuntimeError("PyAV not installed. Install with: pip install av") from e


@dataclass
class SessionRecipe:
    """Metadata for one tensor encoded through a CodecSession."""
    seq_len: int
    K: int
    side_h: int
    side_w: int
    pad_channels: int
    padded_h: int
    padded_w: int
    n_frames: int
    n_packets: int
    scale: np.ndarray
    offset: np.ndarray


def _mkframe(av, planes_3hw: np.ndarray, width: int, height: int):
    frame = av.VideoFrame(width, height, 'yuv444p')
    for i in range(3):
        plane = frame.planes[i]
        ls = plane.line_size
        if ls == width:
            plane.update(planes_3hw[i].tobytes())
        else:
            padded = np.zeros((height, ls), dtype=np.uint8)
            padded[:, :width] = planes_3hw[i]
            plane.update(padded.tobytes())
    return frame


class CodecSession:
    """Persistent encode/decode session sharing one NVENC codec context across calls.

    The session is locked to a single (height, width, qp) — encoding tensors
    that produce frames of different dimensions in the same session is not
    supported; create separate sessions instead.
    """

    def __init__(self, height: int, width: int, qp: int = 18):
        if height < MIN_FRAME_DIM or width < MIN_FRAME_DIM:
            raise ValueError(
                f"frame {width}x{height} below NVENC minimum {MIN_FRAME_DIM}x{MIN_FRAME_DIM}; "
                f"pad each channel to >= {MIN_FRAME_DIM} before passing to the session"
            )
        self._av = _import_av()
        self.height = height
        self.width = width
        self.qp = qp
        self._enc_ctx = None
        self._dec_ctx = None
        self._extradata: Optional[bytes] = None
        self._build_encoder()

    def _build_encoder(self) -> None:
        av = self._av
        codec = av.codec.Codec('hevc_nvenc', 'w')
        ctx = av.codec.CodecContext.create(codec)
        ctx.width = self.width
        ctx.height = self.height
        ctx.pix_fmt = 'yuv444p'
        ctx.framerate = Fraction(30, 1)
        ctx.time_base = Fraction(1, 30)
        ctx.options = {
            'preset': 'p4',
            'rc': 'constqp',
            'qp': str(self.qp),
            'forced_idr': '1',
            'bf': '0',           # no B-frames -> deterministic per-frame output
            'delay': '0',        # no encoder delay
            'rc-lookahead': '0', # no lookahead
        }
        ctx.flags |= av.codec.context.Flags.global_header
        # Warmup encode triggers NVENC session init and populates extradata
        warmup = np.zeros((3, self.height, self.width), dtype=np.uint8)
        f = _mkframe(av, warmup, self.width, self.height)
        f.pict_type = av.video.frame.PictureType.I
        for _ in ctx.encode(f):
            pass
        if ctx.extradata is None:
            raise RuntimeError("NVENC encoder produced no extradata after warmup")
        self._extradata = bytes(ctx.extradata)
        self._enc_ctx = ctx

    def _build_decoder(self):
        av = self._av
        ctx = av.codec.CodecContext.create(av.codec.Codec('hevc_cuvid', 'r'))
        ctx.pix_fmt = 'yuv444p'
        ctx.extradata = self._extradata
        return ctx

    @property
    def extradata(self) -> bytes:
        """The HEVC SPS/PPS/VPS bytes. Receivers must have this to decode."""
        return self._extradata

    def encode_frames(self, frames: np.ndarray) -> list[bytes]:
        """Encode N frames [N, 3, H, W] uint8 -> list of packet bytes.

        Forces an IDR keyframe on the first frame so the resulting packet
        sequence is independently decodable (with the session's extradata).
        """
        av = self._av
        out: list[bytes] = []
        for i in range(frames.shape[0]):
            f = _mkframe(av, frames[i], self.width, self.height)
            if i == 0:
                f.pict_type = av.video.frame.PictureType.I
            for pkt in self._enc_ctx.encode(f):
                out.append(bytes(pkt))
        return out

    def decode_frames(self, packets: list[bytes], n_frames: int) -> np.ndarray:
        """Decode a list of packet bytes back to [N, 3, H, W] uint8."""
        av = self._av
        # We re-create the decoder per call because cuvid doesn't cleanly
        # reset between independent IDR-led streams. The cost is small:
        # decoder init is ~1-2 ms vs encoder init's ~80-100 ms.
        dec_ctx = self._build_decoder()
        out = np.empty((n_frames, 3, self.height, self.width), dtype=np.uint8)
        seen = 0
        for pkt_bytes in packets:
            pkt = av.Packet(pkt_bytes)
            try:
                frames = dec_ctx.decode(pkt)
            except av.error.EOFError:
                break
            for frame in frames:
                if seen >= n_frames:
                    break
                self._frame_to_planes(frame, out[seen])
                seen += 1
        if seen < n_frames:
            try:
                for frame in dec_ctx.decode(None):
                    if seen >= n_frames:
                        break
                    self._frame_to_planes(frame, out[seen])
                    seen += 1
            except av.error.EOFError:
                pass
        if seen != n_frames:
            raise RuntimeError(f"decoded {seen} frames, expected {n_frames}")
        return out

    def _frame_to_planes(self, frame, dst_3hw: np.ndarray) -> None:
        for i in range(3):
            plane = frame.planes[i]
            buf = bytes(plane)
            ls = plane.line_size
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, ls)[:, :self.width]
            dst_3hw[i] = arr

    # High-level API matching pipeline.compress / decompress

    def compress(
        self, tensor: torch.Tensor, basis: Optional[Basis]
    ) -> tuple[list[bytes], SessionRecipe]:
        """Compress one [T, D] tensor through PCA + quant + persistent NVENC encode.

        Returns (packet_bytes_list, recipe). The receiver needs the session's
        extradata plus these packets to decode.
        """
        if tensor.dim() != 2:
            raise ValueError(f"expected [T, D] 2D tensor, got shape {tuple(tensor.shape)}")

        if basis is not None:
            R = basis.project(tensor)
            K = basis.K
        else:
            R = tensor
            K = tensor.shape[1]

        T = R.shape[0]
        side = int(math.isqrt(T))
        while side * side < T:
            side += 1
        pad_rows = side * side - T
        if pad_rows:
            R = torch.cat([R, torch.zeros(pad_rows, K, device=R.device, dtype=R.dtype)], dim=0)
        R_chw = R.T.reshape(K, side, side).contiguous().cpu()

        q, scale, offset = per_channel_quantise(R_chw)
        q_padded, padded_h, padded_w = pad_to_min(q, MIN_FRAME_DIM)
        if padded_h != self.height or padded_w != self.width:
            raise ValueError(
                f"frame {padded_w}x{padded_h} doesn't match session "
                f"{self.width}x{self.height}; create a new session for this tensor shape"
            )
        frames, pad_channels = pack_yuv_frames(q_padded)
        n_frames = frames.shape[0]

        packets = self.encode_frames(frames)
        recipe = SessionRecipe(
            seq_len=tensor.shape[0],
            K=K,
            side_h=side,
            side_w=side,
            pad_channels=pad_channels,
            padded_h=padded_h,
            padded_w=padded_w,
            n_frames=n_frames,
            n_packets=len(packets),
            scale=scale,
            offset=offset,
        )
        return packets, recipe

    def decompress(
        self, packets: list[bytes], basis: Optional[Basis], recipe: SessionRecipe
    ) -> torch.Tensor:
        decoded = self.decode_frames(packets, recipe.n_frames)
        q_padded_recon = unpack_yuv_frames(decoded, recipe.K, recipe.pad_channels)
        q_recon = crop_pad(q_padded_recon, recipe.side_h, recipe.side_w)
        R_chw_recon = per_channel_dequantise(q_recon, recipe.scale, recipe.offset)
        R_recon = R_chw_recon.reshape(recipe.K, -1).T
        R_recon = R_recon[: recipe.seq_len]
        if basis is not None:
            R_recon = R_recon.to(basis.V_K.device)
            return basis.invert(R_recon)
        return R_recon

    def close(self) -> None:
        """Release the encoder. The session is unusable after this."""
        self._enc_ctx = None
        self._dec_ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
