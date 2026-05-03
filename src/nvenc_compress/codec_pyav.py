"""PyAV-based codec backend (in-process FFmpeg via the `av` package).

Drop-in replacement for codec.py's encode_hevc / decode_hevc — same args,
same return types, same on-disk bitstream format. Internally uses PyAV's
in-process FFmpeg API to avoid the subprocess startup penalty (~50-150 ms
per call on Windows).

Requires the optional `av` dependency: `pip install av`. The bundled FFmpeg
in modern PyAV wheels (>=14) includes hevc_nvenc and hevc_cuvid, so no
custom build is needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def _import_av():
    try:
        import av
        return av
    except ImportError as e:
        raise RuntimeError(
            "PyAV not installed. Install with: pip install av"
        ) from e


def check_available() -> tuple[bool, str]:
    """Returns (ok, message). True if PyAV is installed and hevc_nvenc + hevc_cuvid
    are available in its bundled FFmpeg."""
    try:
        av = _import_av()
    except RuntimeError as e:
        return False, str(e)
    encoders = []
    decoders = []
    for name in av.codecs_available:
        try:
            av.codec.Codec(name, 'w')
            if name in ('hevc_nvenc', 'h264_nvenc', 'av1_nvenc'):
                encoders.append(name)
        except Exception:
            pass
        try:
            av.codec.Codec(name, 'r')
            if name in ('hevc_cuvid', 'h264_cuvid', 'av1_cuvid'):
                decoders.append(name)
        except Exception:
            pass
    if 'hevc_nvenc' not in encoders:
        return False, f"hevc_nvenc not in PyAV's FFmpeg encoders (found: {encoders})"
    if 'hevc_cuvid' not in decoders:
        return False, f"hevc_cuvid not in PyAV's FFmpeg decoders (found: {decoders})"
    return True, f"PyAV {av.__version__} with FFmpeg {av.ffmpeg_version_info}, hevc_nvenc + hevc_cuvid available"


def _frame_from_planes(av_module, planes_3hw: np.ndarray, width: int, height: int):
    """Build a yuv444p VideoFrame from a [3, H, W] uint8 numpy array."""
    frame = av_module.VideoFrame(width, height, 'yuv444p')
    for plane_idx in range(3):
        plane = frame.planes[plane_idx]
        ls = plane.line_size
        if ls == width:
            plane.update(planes_3hw[plane_idx].tobytes())
        else:
            padded = np.zeros((height, ls), dtype=np.uint8)
            padded[:, :width] = planes_3hw[plane_idx]
            plane.update(padded.tobytes())
    return frame


def encode_hevc(
    frames: np.ndarray, height: int, width: int, qp: int, out_path: Path
) -> int:
    """Encode raw YUV 4:4:4 frames as HEVC via PyAV + hevc_nvenc.

    Same signature and contract as codec.py:encode_hevc.

    Args:
        frames: numpy array shape [N, 3, H, W] uint8.
        height, width: frame dimensions.
        qp: NVENC HEVC constant quantization parameter (0-51).
        out_path: where to write the .hevc bitstream.

    Returns:
        Byte size of the resulting bitstream.
    """
    av = _import_av()
    out_buf = io.BytesIO()
    container = av.open(out_buf, mode='w', format='hevc')
    stream = container.add_stream('hevc_nvenc', rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv444p'
    stream.options = {
        'preset': 'p4',
        'rc': 'constqp',
        'qp': str(qp),
    }
    for i in range(frames.shape[0]):
        frame = _frame_from_planes(av, frames[i], width, height)
        for packet in stream.encode(frame):
            container.mux(packet)
    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    data = out_buf.getvalue()
    out_path.write_bytes(data)
    return len(data)


def decode_hevc(
    in_path: Path, num_frames: int, height: int, width: int
) -> np.ndarray:
    """Decode an HEVC bitstream to raw YUV 4:4:4 frames via PyAV + hevc_cuvid.

    Same signature and contract as codec.py:decode_hevc.

    Returns:
        numpy array shape [num_frames, 3, height, width] uint8.
    """
    av = _import_av()
    data = in_path.read_bytes()
    container = av.open(io.BytesIO(data), mode='r', format='hevc')
    # Force the cuvid decoder
    stream = container.streams.video[0]
    stream.codec_context.options = {}
    # PyAV doesn't have a clean public API to force a specific decoder for an
    # already-opened stream, but for raw HEVC bitstreams the default 'hevc'
    # software decoder is what gets used. To force NVDEC we need to specify
    # the codec at open time via codec= kwarg.
    container.close()

    container = av.open(io.BytesIO(data), mode='r', format='hevc')
    # Replace the codec with the cuvid one
    stream = container.streams.video[0]
    new_codec = av.codec.Codec('hevc_cuvid', 'r')
    new_ctx = av.codec.CodecContext.create(new_codec)
    new_ctx.pix_fmt = 'yuv444p'

    out = np.empty((num_frames, 3, height, width), dtype=np.uint8)
    frames_seen = 0
    for packet in container.demux(stream):
        for frame in new_ctx.decode(packet):
            if frames_seen >= num_frames:
                break
            _frame_to_planes(frame, out[frames_seen], height, width)
            frames_seen += 1
    # Flush: PyAV 17+ raises EOFError on the None sentinel; older returns empty.
    if frames_seen < num_frames:
        try:
            for frame in new_ctx.decode(None):
                if frames_seen >= num_frames:
                    break
                _frame_to_planes(frame, out[frames_seen], height, width)
                frames_seen += 1
        except av.error.EOFError:
            pass
    container.close()

    if frames_seen != num_frames:
        raise RuntimeError(f"decoded {frames_seen} frames, expected {num_frames}")
    return out


def _frame_to_planes(frame, dst_3hw: np.ndarray, height: int, width: int) -> None:
    """Read an av.VideoFrame's yuv444p planes into a preallocated [3, H, W] uint8 array."""
    for plane_idx in range(3):
        plane = frame.planes[plane_idx]
        buf = bytes(plane)
        ls = plane.line_size
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, ls)[:, :width]
        dst_3hw[plane_idx] = arr
