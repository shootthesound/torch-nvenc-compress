"""n2 — AV1 vs HEVC: HEVC wins for our use case.

Hypothesis: AV1 is a newer codec with better entropy coding. Should compress
our content better than HEVC at the same quality.

Reality on Blackwell (RTX 5090):

  - AV1 NVENC 4:4:4 returns "No capable devices found". The hardware doesn't
    support AV1 4:4:4 encoding even though FFmpeg lists yuv444p as a
    supported pix_fmt for av1_nvenc. The driver rejects it.

  - The 4:2:0-only workaround forces 1-channel-per-Y-plane packing (with
    chroma planes filled with neutral 128s). That's 3x more frames than
    HEVC 4:4:4 (which packs 3 channels per frame as Y/U/V), tripling the
    per-frame overhead.

  - In direct comparison at K=500, LOO across our captures:
        HEVC qp=18:  37.3x ratio at cos 0.943
        AV1  qp=24:  26.8x ratio at cos 0.952

    AV1 ratio is also relatively *insensitive* to QP across 10/24/36
    (22.2x / 26.8x / 29.9x at near-identical 0.951 cos). It's hitting a
    quality floor at this content type.

Conclusion: HEVC 4:4:4 is the right primary codec until NVIDIA enables AV1
4:4:4 on consumer NVENC.

This script just verifies that AV1 4:4:4 still fails on your machine. If
NVIDIA's driver eventually adds support, the ffmpeg call below will succeed
and the result is worth re-evaluating with a real PoC sweep.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH; cannot test")
        sys.exit(1)

    print("Testing AV1 NVENC support on this system...\n")

    # 1. Does the FFmpeg build claim AV1 NVENC at all?
    enc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    if "av1_nvenc" not in enc.stdout:
        print("This FFmpeg build does NOT include av1_nvenc at all.")
        print("(That's a build-time choice, not a hardware limit.)")
        return
    print("[PASS] FFmpeg build includes av1_nvenc")

    # 2. Does AV1 4:2:0 work? (the supported case)
    print("\n[2] Trying AV1 NVENC 4:2:0 encode on testsrc input...")
    proc = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=256x256:rate=30:duration=1",
        "-c:v", "av1_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "24",
        "-pix_fmt", "yuv420p", "-f", "ivf", "-",
    ], capture_output=True)
    if proc.returncode == 0:
        print("    [PASS] AV1 4:2:0 encode works on this hardware")
    else:
        print(f"    [FAIL] AV1 4:2:0 encode failed:\n        {proc.stderr.decode(errors='replace').strip().splitlines()[-1]}")

    # 3. Does AV1 4:4:4 work? (the broken case on Blackwell)
    print("\n[3] Trying AV1 NVENC 4:4:4 encode on testsrc input...")
    proc = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=256x256:rate=30:duration=1",
        "-c:v", "av1_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "24",
        "-pix_fmt", "yuv444p", "-f", "ivf", "-",
    ], capture_output=True)
    if proc.returncode == 0:
        print("    [PASS!] AV1 4:4:4 works on this hardware!")
        print("    NVIDIA may have enabled AV1 4:4:4 in a recent driver. Re-run the")
        print("    pareto sweep with av1_nvenc as a competing codec — it's worth")
        print("    re-measuring whether AV1 now beats HEVC for our content type.")
    else:
        last_err = proc.stderr.decode(errors='replace').strip().splitlines()[-1]
        print(f"    [FAIL as expected] AV1 4:4:4 not supported: {last_err}")
        print()
        print("This matches the result we got on RTX 5090 / Blackwell during research.")
        print("HEVC 4:4:4 remains the right primary codec.")
        print()
        print("Why this matters: AV1's main advantage is entropy coding efficiency.")
        print("With AV1 forced to 4:2:0, our per-channel packing has to use 1 channel")
        print("per Y plane (chroma planes filled with 128s), giving 3x more frames per")
        print("'video' than HEVC 4:4:4. The frame-count overhead negates AV1's coding")
        print("efficiency advantage and HEVC wins.")


if __name__ == "__main__":
    main()
