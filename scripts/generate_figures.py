"""Generate the PNG figures referenced by the project READMEs.

Numbers are from the most recent benchmark runs documented in the project
log; if you re-run poc/16, poc/17, poc/18, the actual measured numbers may
shift by ~5-10% due to GPU thermals + CPU scheduling variance. The figures
here are representative, not freshly re-measured each time.

Outputs to docs/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "subprocess":    "#cccccc",
    "pyav":          "#9aa5b1",
    "pyav-multi":    "#7a8693",
    "direct-1":      "#5b8def",
    "direct-multi":  "#1f4ed8",
    "nvlink-3090":   "#a8d5a8",
    "nvlink-target": "#5cba5c",
}


def fig_encode_decode_bench():
    """Bar chart of encode + decode ms/frame across backends.

    Numbers from poc/18 real-FLUX-activation bench, 668 frames @ 256x256
    YUV444 QP=18 on RTX 5090.
    """
    backends = [
        "PyAV CodecSession",
        "DirectBackend\n(1 engine, pool=8)",
        "MultiEngineDirectBackend\n(3 engines × pool=8)",
    ]
    enc = [0.469, 0.243, 0.180]
    dec = [0.887, 0.435, 0.262]
    colors = [COLORS["pyav"], COLORS["direct-1"], COLORS["direct-multi"]]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    x = np.arange(len(backends))
    w = 0.36
    bars1 = ax.bar(x - w/2, enc, w, label="encode", color=colors, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, dec, w, label="decode", color=colors, alpha=0.55,
                    edgecolor="black", linewidth=0.5, hatch="///")

    for bars, vals in [(bars1, enc), (bars2, dec)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(backends, fontsize=9)
    ax.set_ylabel("ms / frame")
    ax.set_title("Codec backend latency on real FLUX activations\n"
                 "(668 frames @ 256×256 YUV444, QP=18, RTX 5090)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "encode_decode_bench.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_speedup_vs_baselines():
    """End-to-end speedup of MultiEngineDirectBackend vs other backends."""
    labels = ["FFmpeg subprocess\n(original)", "PyAV per-call",
              "PyAV CodecSession", "MultiEngineCodecSession", "DirectBackend (1 eng)",
              "MultiEngineDirectBackend\n(3 eng)"]
    # Per-tensor latency in ms (from project log + poc/18)
    latencies = [577, 466, 180, 108, 122.9, 80.2]
    speedups = [577 / l for l in latencies]
    colors = [COLORS["subprocess"], COLORS["pyav"], COLORS["pyav-multi"],
              COLORS["pyav-multi"], COLORS["direct-1"], COLORS["direct-multi"]]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    bars = ax.barh(labels, speedups, color=colors, edgecolor="black", linewidth=0.5)
    for b, s in zip(bars, speedups):
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height()/2,
                f"{s:.2f}×", va="center", fontsize=9)
    ax.set_xlabel("Speedup vs FFmpeg subprocess baseline")
    ax.set_title("End-to-end codec speedup, smaller-is-better workload\n"
                 "(per-tensor latency, K=500 PCA + HEVC YUV444 QP=18 on real FLUX)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(speedups) * 1.18)

    fig.tight_layout()
    out = OUT_DIR / "speedup_vs_baselines.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_parallel_path_overlap():
    """Parallel-path overlap visualisation: timeline of GEMM vs encode."""
    # From poc/17 measurements
    gemm_only = 20.9
    encode_only = 19.9
    serialized = 40.1
    parallel = 26.0

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)

    # Bar 1: serialized (GEMM then encode end-to-end)
    ax.barh(0, gemm_only, height=0.5, color=COLORS["direct-1"],
             edgecolor="black", label="GEMM (compute)")
    ax.barh(0, encode_only, left=gemm_only, height=0.5,
             color=COLORS["direct-multi"], edgecolor="black", label="NVENC encode")
    ax.text(serialized + 0.5, 0, f"  {serialized:.1f} ms (sum)",
             va="center", fontsize=10, fontweight="bold")

    # Bar 2: parallel (overlap)
    ax.barh(1, gemm_only, height=0.5, color=COLORS["direct-1"], edgecolor="black")
    ax.barh(1, encode_only, height=0.5, color=COLORS["direct-multi"],
             alpha=0.7, edgecolor="black", hatch="///")
    ax.text(parallel + 0.5, 1, f"  {parallel:.1f} ms (parallel — 1.34× speedup)",
             va="center", fontsize=10, fontweight="bold", color="#1a5e1a")

    # Theoretical floor marker
    floor = max(gemm_only, encode_only)
    ax.axvline(floor, ymin=0.15, ymax=0.85, linestyle=":", color="#5cba5c",
                label=f"theoretical max overlap floor ({floor:.1f} ms)")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["serialized\n(no overlap)", "parallel\n(streams A + B)"])
    ax.set_xlabel("wall-clock (ms)")
    ax.set_title("Parallel-path overlap: NVENC encode runs concurrently with SM compute\n"
                 "(stream A: 30×4096² fp16 GEMM, stream B: 64-frame encode bound via nvEncSetIOCudaStreams)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, serialized * 1.55)
    ax.set_ylim(-0.5, 1.7)

    fig.tight_layout()
    out = OUT_DIR / "parallel_path_overlap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_nvlink_status():
    """Status pie / progress bar of the four NVLink-replacement building blocks."""
    blocks = [
        "Compression ratio\n(6× lossless on diffusion)",
        "Codec latency\n(0.180 ms/f encode)",
        "Parallel-path overlap\n(67% measured)",
        "Cross-GPU PCIe P2P\n(blocked on 2nd GPU)",
    ]
    pct = [100, 100, 100, 0]
    colors = [COLORS["nvlink-3090"], COLORS["nvlink-3090"],
              COLORS["nvlink-3090"], "#e9b3b3"]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    bars = ax.barh(blocks, pct, color=colors, edgecolor="black", linewidth=0.5)
    for b, p, label in zip(bars, pct, ["DONE", "DONE", "DONE", "BLOCKED"]):
        ax.text(min(p + 2, 102), b.get_y() + b.get_height()/2,
                f"{p}%  {label}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("validation % per building block")
    ax.set_title("NVLink-replacement claim status — ~75% validated\n"
                 "(three of four building blocks fully measured; the fourth is hardware-blocked)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "nvlink_status.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_bandwidth_amplification():
    """Effective wire bandwidth, with vs without NVENC compression."""
    wires = ["RTX 5090\nPCIe P2P", "10 Gbit\nethernet", "1 Gbit\nethernet",
             "100 Mbps\nresidential", "NVMe Gen4\n(GPUDirect Storage)"]
    raw_gbps = [30, 1.25, 0.125, 0.0125, 7]
    amplified_gbps = [r * 6 for r in raw_gbps]

    x = np.arange(len(wires))
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.bar(x - w/2, raw_gbps, w, label="without compression",
            color=COLORS["pyav"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, amplified_gbps, w, label="with NVENC at 6× lossless",
            color=COLORS["direct-multi"], edgecolor="black", linewidth=0.5)

    # Annotate
    for i, (raw, amp) in enumerate(zip(raw_gbps, amplified_gbps)):
        ax.text(i - w/2, raw, f"{raw} GB/s" if raw >= 1 else f"{int(raw*1000)} MB/s",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, amp, f"{amp:.0f} GB/s" if amp >= 1 else f"{int(amp*1000)} MB/s",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#1a3a8e")

    # NVLink-3 reference line
    ax.axhline(56, linestyle="--", color="#5cba5c", alpha=0.7,
                label="NVLink 3 (RTX 3090) per-direction = 56 GB/s")

    ax.set_xticks(x)
    ax.set_xticklabels(wires, fontsize=9)
    ax.set_ylabel("effective bandwidth (GB/s, log scale)")
    ax.set_yscale("log")
    ax.set_title("NVENC compression multiplies effective wire bandwidth by ~6×\n"
                 "(diffusion activations, lossless 6× compression)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "bandwidth_amplification.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    fig_encode_decode_bench()
    fig_speedup_vs_baselines()
    fig_parallel_path_overlap()
    fig_nvlink_status()
    fig_bandwidth_amplification()
    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
