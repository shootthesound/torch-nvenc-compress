"""Verify the toolchain is ready: torch + CUDA, FFmpeg with NVENC HEVC and NVDEC.

Run this BEFORE downloading multi-GB models — it tells you in 5 seconds whether
the rest of the repo will work on this machine.

    python scripts/check_environment.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def check_torch_cuda() -> tuple[bool, str]:
    try:
        import torch
    except ImportError as e:
        return False, f"torch not installed ({e})"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False — no GPU detected by torch"
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    return True, f"torch {torch.__version__} sees {name} (compute capability {cc[0]}.{cc[1]})"


def check_ffmpeg() -> tuple[bool, str]:
    path = shutil.which("ffmpeg")
    if not path:
        return False, "ffmpeg not found on PATH"
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        first_line = proc.stdout.splitlines()[0] if proc.stdout else "unknown"
    except Exception as e:
        return False, f"ffmpeg execution failed: {e}"
    return True, f"ffmpeg at {path} — {first_line}"


def check_nvenc() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return False, f"failed to query ffmpeg encoders: {e}"
    if "hevc_nvenc" in proc.stdout:
        return True, "hevc_nvenc encoder available"
    return False, (
        "hevc_nvenc NOT found in this ffmpeg build. You need an FFmpeg compiled "
        "with --enable-nvenc. On Windows, the gyan.dev essentials build includes it."
    )


def check_nvdec() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return False, f"failed to query ffmpeg decoders: {e}"
    if "hevc_cuvid" in proc.stdout:
        return True, "hevc_cuvid decoder available"
    return False, (
        "hevc_cuvid NOT found in this ffmpeg build. You need an FFmpeg compiled "
        "with --enable-cuda-nvcc."
    )


def check_diffusers() -> tuple[bool, str]:
    try:
        import diffusers
    except ImportError:
        return False, "diffusers not installed (optional — install with `pip install -e .[diffusion]`)"
    return True, f"diffusers {diffusers.__version__}"


def check_transformers() -> tuple[bool, str]:
    try:
        import transformers
    except ImportError:
        return False, "transformers not installed (optional — install with `pip install -e .[llm]`)"
    return True, f"transformers {transformers.__version__}"


def main() -> int:
    print("torch-nvenc-compress environment check\n")

    required = [
        ("torch + CUDA", check_torch_cuda()),
        ("FFmpeg",       check_ffmpeg()),
        ("hevc_nvenc",   check_nvenc()),
        ("hevc_cuvid",   check_nvdec()),
    ]
    optional = [
        ("diffusers",    check_diffusers()),
        ("transformers", check_transformers()),
    ]

    any_required_failed = False
    print("REQUIRED:")
    for label, (ok, msg) in required:
        marker = "  PASS" if ok else "  FAIL"
        print(f"{marker}  {label:<14s}  {msg}")
        if not ok:
            any_required_failed = True

    print("\nOPTIONAL (needed only for the model-download / capture scripts):")
    for label, (ok, msg) in optional:
        marker = "  ok  " if ok else "  --  "
        print(f"{marker}  {label:<14s}  {msg}")

    print()
    if any_required_failed:
        print("Some required checks FAILED. Fix the items above before running the PoCs.")
        return 1
    print("Environment OK. Try `python poc/01_synthetic_controls.py` next "
          "(no model download required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
