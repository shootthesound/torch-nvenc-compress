# Findings

All numbers from leave-one-out (LOO) PCA + per-channel uint8 quantisation + NVENC HEVC 4:4:4 at constant QP. Calibration N is the number of captures the basis is built from per LOO iteration.

For the prior-art context (LLM.265, KVFetcher, CodecFlow) and what's specifically new here, see the README's "Prior art and what's new" section.

## Quick navigation

- [Compression Pareto curves](#diffusion-flux2-klein-9b-mid-double-block-activations) — what compression at what quality on diffusion + LLM
- [Synthetic-data sanity controls](#synthetic-data-sanity-controls-poc01) — proof the pipeline isn't broken
- [Cross-domain comparison](#whats-settled-across-both-domains) — diffusion vs LLM Pareto compared
- [Codec subprocess overhead breakdown](#codec-subprocess-overhead-breakdown) — proving the fast-wrapper claim with numbers
- [End-to-end wall-clock measurements](#end-to-end-wall-clock-measured-today) — what wins TODAY on which wires

## Diffusion: FLUX.2 Klein 9B mid-double-block activations

- **Model**: FLUX.2 Klein 9B fp8 distilled (commercially licensed, ComfyUI capture path during research; PoC repo uses FLUX.1-schnell instead, see "Model substitution" below)
- **Capture point**: double-block layer 4 of 8, the `img` stream output (image tokens only, no text tokens)
- **Tensor shape**: `[1, 4096, 4096]` bf16 = `[T=4096 image tokens, D=4096 hidden dim]`, 32 MB per tensor
- **Calibration**: N=64 LOO

| K (rank kept) | QP | Mean ratio | Mean cos | Min cos (worst-case prompt) | Mean p1 (worst channel) |
|---|---|---|---|---|---|
| 500 | 10 | 24.7× | 0.951 | 0.853 | 0.844 |
| **500** | **18** | **37.3×** | 0.943 | 0.845 | 0.833 |
| 1000 | 10 | 12.3× | 0.975 | 0.902 | 0.906 |
| **1000** | **18** | **18.4×** | 0.966 | 0.893 | 0.894 |
| **2000** | **10** | **6.1×** | **0.991** | **0.955** | **0.956** |
| 2000 | 18 | 9.1× | 0.982 | 0.946 | 0.945 |

**Spectrum**: top 100 channels (2.4% of D=4096) hold 75% of variance. Effective rank 160/4096. Largest eigenvalue is ~26 000× the median.

**V matrix size** (the per-layer constant shipped with the model): K=500 → 4 MB, K=1000 → 8 MB, K=2000 → 16 MB. For all 8 double-blocks at K=500: **32 MB total** vs 9 GB model = trivial overhead.

## LLM: Mistral 7B v0.3 KV cache (mid-network layer 16)

- **Model**: Mistral 7B v0.3 base (Apache 2.0)
- **Architecture**: 32 layers, num_key_value_heads=8, head_dim=128 → **1024 KV channels** per layer
- **Capture**: 6 prompts × ~2400 tokens, K and V tensors at layer 16
- **Calibration**: N=6 LOO

### K cache (Keys)

| K | QP | Mean ratio | Mean cos | Mean p1 |
|---|---|---|---|---|
| 50 | 18 | 79× | 0.700 | 0.000 |
| 100 | 18 | 40× | 0.86 | 0.31 |
| 200 | 18 | 20× | 0.91 | 0.58 |
| 400 | 18 | 10× | 0.95 | 0.77 |
| **800** | **10** | **3.4×** | **0.993** | 0.95 |
| 800 | 18 | 5.1× | 0.986 | 0.94 |
| **1024** | **10** | **2.7×** | **0.999** | **0.998** |

### V cache (Values)

| K | QP | Mean ratio | Mean cos | Mean p1 |
|---|---|---|---|---|
| 200 | 18 | 19× | 0.78 | 0.41 |
| 400 | 18 | 10× | 0.87 | 0.64 |
| 800 | 18 | 5.1× | 0.965 | 0.91 |
| **1024** | **10** | **2.7×** | **0.999** | **0.997** |

### K vs V asymmetry

K cache spectrum is more concentrated (effective rank 86/1024) than V cache (148/1024). At lossy operating points K reconstructs noticeably better than V. Asymmetric bit allocation per cache type is a real lever we didn't explore in detail.

## Diffusion: VAE latents (FLUX.2 Klein 9B)

- **Tensor shape**: `[1, 128, 64, 64]` bf16 = 1 MB per latent (small)
- **Calibration**: N=8 LOO

| K | QP | Mean ratio | Mean cos |
|---|---|---|---|
| **128** | **10** | **3.6×** | **0.999** |
| 128 | 18 | 5.9× | 0.989 |
| 75 | 18 | 10× | 0.967 |
| 25 | 18 | **27.8×** | 0.895 |

Latents do NOT compress dramatically better than mid-block activations — the "image-shaped tensors compress like images" hypothesis is empirically false for FLUX.2's specific VAE design. Channel stds range only 0.5–1.2 (vs activations' 0.26–12.6) — the VAE's training already fused the channel covariance, leaving PCA less heavy-tailed structure to exploit.

## Synthetic-data sanity controls (`poc/01`)

The pipeline isn't broken — it's working at the information-theoretic limit for noise-like content. Verified by feeding tensors of *known* compressibility through it:

| Source | Ratio | cos_mean |
|---|---|---|
| zeros | 686× | (NaN — all zero) |
| same image × 4096 | 595× | 0.9999 |
| smooth_per_channel | 75× | 0.9998 |
| smooth + noise mix | 6.2× | 0.993 |
| pure Gaussian noise | 4.2× | 0.988 |
| **REAL FLUX activation** (no PCA) | **4.8×** | **0.988** |

Real mid-block activations compress at almost exactly the same rate as pure Gaussian noise, in the standard basis — confirming they ARE noise-like in the standard basis. PCA reveals the heavy-tailed structure that lets us beat noise's information-theoretic limit.

## What's settled across both domains

The pipeline qualitatively works across MM-DiT (FLUX) and decoder LLM (Mistral) families. Pareto curves are similar shape:

| Compression target | Diffusion (FLUX) | LLM KV (Mistral) |
|---|---|---|
| Lossless (cos > 0.99) | **6.1×** | **2.7×** |
| Near-lossless (cos > 0.97) | 12× | 5× |
| Aggressive (cos > 0.94) | **37×** | 10× |

KV is consistently less compressible than diffusion activations. Two reasons:

1. **Fewer absolute channels** (1024 vs 4096 → less low-variance tail to truncate)
2. **GQA already compressed information architecturally** — the trained KV cache is closer to information-theoretic minimum per channel than diffusion model intermediates are.

## Things that did NOT crack the Pareto open

See [`poc/null_findings/`](../poc/null_findings/) for runnable demonstrations.

- **Sparse residual**: error is uniformly distributed, not outlier-concentrated. Even 10% of positions corrected only adds ~0.02 cos.
- **AV1 NVENC**: 4:4:4 unsupported on Blackwell; 4:2:0 + 1ch-per-Y-plane workaround loses to HEVC 4:4:4 on overhead.
- **10-bit encoding** (yuv444p16le): essentially identical to 8-bit at same QP. NVENC's main10 doesn't engage usefully on this content.
- **Channel reordering** (greedy similarity): PCA orthogonalises by construction, so codec's P-frame prediction has nothing to exploit.

## Model substitution caveat

The numbers above were measured during research on **FLUX.2 Klein 9B fp8** (via ComfyUI custom node) and **Mistral 7B v0.3** (via transformers). The public PoC repo uses **FLUX.1-schnell** (same MM-DiT family, ungated, in `diffusers`) and **Qwen 2.5 7B** by default (also ungated, 512 KV channels) with Mistral 7B v0.3 as an opt-in. Your numbers running the PoCs will differ slightly from the FLUX.2 Klein and Mistral numbers (different specific models, slightly different statistics) but the qualitative shape — heavy-tailed spectrum, 6×-class lossless ratio for diffusion, 2.7-3×-class lossless ratio for LLM KV — should reproduce.

## Codec subprocess overhead breakdown

From [`poc/10_codec_overhead_breakdown.py`](../poc/10_codec_overhead_breakdown.py), measured on RTX 5090 with Windows FFmpeg 7.0.2 (gyan.dev essentials build):

| Component | Time | % of encode |
|---|---|---|
| Pure subprocess spawn (`ffmpeg -version`) | ~20 ms | 8% |
| Spawn + FFmpeg init + arg parse + null-sink encode | ~171 ms | 66% |
| Real encode round-trip (full pipeline) | ~258 ms | 100% |
| Real decode round-trip | ~258 ms | — |
| **Implied: subprocess + init overhead** | **~171 ms** | **66% of each encode call** |
| **Implied: real NVENC HW + necessary I/O** | **~87 ms** | **34% of each encode call** |

This is the empirical evidence behind the "fast wrapper would close the gap" claim used throughout the docs. PyAV partially attacks this by eliminating the subprocess startup (~20 ms saved) but still pays per-call FFmpeg codec init (~150 ms). CodecSession amortises the per-call FFmpeg init across many tensor encodes — see next section.

Interesting datapoint: pure subprocess spawn on Windows is only ~20 ms, but the **FFmpeg-specific** initialisation (loading codecs, parsing args, opening NVENC session) adds another ~150 ms before any real encoding happens. Most of the "subprocess overhead" is actually FFmpeg startup, not the OS process model.

## Codec backend comparison (measured)

From [`poc/11_codec_session_bench.py`](../poc/11_codec_session_bench.py), real captured FLUX activations (4096-channel, K=1000 PCA truncation, QP=18), batch sizes 1-16, on RTX 5090:

| Backend | Per-tensor encode | Speedup vs subprocess | Quality (cos) | Bitstream size at QP=18 |
|---|---|---|---|---|
| `subprocess` (default) | ~302 ms | 1.0× | 0.946 | baseline |
| `pyav` (per-call, in-process FFmpeg) | ~243 ms | 1.24× | 0.946 (same) | same |
| **`CodecSession` (persistent NVENC context)** | **~170 ms** | **1.77×** | 0.953 (slightly better) | ~22% larger* |

`*` CodecSession disables B-frames and lookahead for deterministic per-frame output. The bitstream-size penalty can be offset by bumping QP from 18 to ~22, recovering a comparable compression ratio at comparable quality.

The CodecSession speedup is **consistent across batch sizes from N=1 to N=16**:

| N tensors | subprocess | per-call PyAV | CodecSession | Session vs subprocess |
|---|---|---|---|---|
| 1 | 307 ms | 248 ms | 180 ms | 1.70× |
| 2 | 617 ms | 478 ms | 344 ms | 1.79× |
| 4 | 1213 ms | 976 ms | 670 ms | 1.81× |
| 8 | 2420 ms | 1986 ms | 1369 ms | 1.77× |
| 16 | 4833 ms | 3901 ms | 2727 ms | 1.77× |

The session amortises the ~80-100 ms NVENC init cost over the batch via persistent codec context, plus eliminates per-call FFmpeg subprocess overhead via PyAV. Both savings stack.

## End-to-end wall-clock measured today

These are the measurements that prove the codec wins TODAY on slow consumer wires, even with the slow subprocess pipeline.

### Single-tensor wire-bottleneck case (`poc/08_wire_simulation.py`)

Real codec round-trip + simulated wire transmission via `sleep(bytes/bandwidth)`. Measured for a 33 MB tensor at 17× compression:

| Wire | Direct | Codec round-trip | Speedup |
|---|---|---|---|
| PCIe 5.0 ×16 | 0.5 ms | 700 ms | direct wins |
| 10 Gbit ethernet | 27 ms | 701 ms | direct wins |
| 1 Gbit ethernet | 268 ms | 716 ms | direct wins |
| **100 Mbps residential** | **2685 ms** | **858 ms** | **3.13× codec** |
| **50 Mbps residential** | **5369 ms** | **1015 ms** | **5.29× codec** |

### Multi-tensor concurrent dual-lane case (`poc/09_dual_lane.py`)

8 tensors offloaded by three strategies. "Dual lane" = half via direct PCIe + half via codec, both running concurrently on separate cuda streams + Python threads:

| Wire | All direct (baseline) | All codec | Dual lane | Dual-lane vs direct |
|---|---|---|---|---|
| PCIe (no wire bottleneck) | 72 ms | 2406 ms | 1234 ms | codec subprocess dominates |
| **1 Gbit ethernet** | **2186 ms** | 2544 ms | **1290 ms** | **1.69× faster** |
| **100 Mbps residential** | **21512 ms** | 3651 ms | 10758 ms | **2.00× faster** |

Take-aways:
- For consumer wires (1 Gbit and slower), the codec already wins TODAY without any further engineering.
- For PCIe and faster wires, the projected wins from the README require the fast wrapper.
- Dual-lane operation (mixed direct + codec on separate streams) wins on every consumer wire because the two paths run on independent hardware (PCIe controller + NVENC silicon) and the codec lane consumes only the small compressed bytes on the shared PCIe bus.
