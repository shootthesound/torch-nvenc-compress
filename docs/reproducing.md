# Reproducing the results

End-to-end from clone to numbers, with rough time estimates.

## 1. Prerequisites

- An NVIDIA GPU (Turing or newer for HEVC NVENC; tested on RTX 5090 / Blackwell)
- A PyTorch CUDA build matching your GPU
- An FFmpeg build that includes `hevc_nvenc` (encoder) and `hevc_cuvid` (decoder)

On Windows: `winget install Gyan.FFmpeg` gets you a build with both.
On Debian/Ubuntu: install `ffmpeg` from your distro, then verify (older distro packages may not include NVENC; `apt install ffmpeg` from recent Ubuntu does).
On macOS: NVENC is NVIDIA-only; this repo is Linux/Windows.

## 2. Install

```bash
git clone https://github.com/shootthesound/torch-nvenc-compress
cd torch-nvenc-compress

# Fresh venv (Python 3.10–3.13)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

pip install -e ".[all]"             # core + diffusion + llm extras
```

If `pip install` of `torch` doesn't pick up CUDA, replace the torch install line with the right index URL for your CUDA version, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(Then `pip install -e .` for the package itself.)

## 3. Verify the toolchain

```bash
python scripts/check_environment.py
```

Expect all four required checks (torch+CUDA, ffmpeg, hevc_nvenc, hevc_cuvid) to PASS. If any fails, fix that before continuing.

## 4. Synthetic-data sanity (no model download)

```bash
python poc/01_synthetic_controls.py
```

Takes about 2 minutes. Expect numbers close to:
```
zeros                  ~600x   cos ~0  (NaN cosine on all-zero is normal)
one_image_replicated   ~600x   cos ~0.999
smooth_per_channel      ~75x   cos ~0.999
noise_plus_lowfreq       ~6x   cos ~0.99
gaussian_noise           ~4x   cos ~0.99
```

If this works, the rest of the repo will work. If it doesn't, something is wrong with the codec wrapper or the pipeline — open an issue with the failing output.

## 5. Diffusion path (FLUX.1-schnell, ~24 GB download)

```bash
# Download (slowest step — ~30 min on 100 Mbps)
python scripts/download_flux_diffusers.py

# Capture activations from 32 diverse prompts (~5 min on a 5090)
python scripts/capture_diffusion.py --num-prompts 32 --layer 9 --steps 4

# Run the analyses
python poc/02_spectrum_diffusion.py     # ~10 sec
python poc/03_diffusion_pareto.py       # ~5-10 min depending on N and K sweep
```

Expect spectrum to show "top 100 channels hold ~75% of variance" and the Pareto sweep to show ~6× lossless / 12× near-lossless / 30× aggressive ratios. Numbers will differ from FLUX.2 Klein (which the research used) but qualitative shape should match.

## 6. LLM KV path (Qwen 2.5 7B, ungated, ~15 GB download)

```bash
# Download (~16 min on 100 Mbps). Apache 2.0, no auth needed.
python scripts/download_qwen_7b.py

# Capture KV from 8 long prompts (~2 min on a 5090)
python scripts/capture_llm_kv.py --num-prompts 8 --layer 14

# Run the analyses
python poc/04_spectrum_llm_kv.py        # ~5 sec
python poc/05_llm_kv_pareto.py          # ~3-5 min
```

Expect spectrum to show K cache more concentrated than V cache. Pareto sweep shows ~2.7-3× lossless for both K and V at K_keep = D.

### Alternative LLMs

| Script | Model | Gated? | KV channels per layer | Download |
|---|---|---|---|---|
| `download_qwen.py` | Qwen 2.5 1.5B Instruct | No | 256 | ~3 GB |
| `download_qwen_7b.py` | Qwen 2.5 7B | No | 512 | ~15 GB |
| `download_mistral.py` | Mistral 7B v0.3 | **Yes** | 1024 | ~14 GB |

**The Mistral path requires HuggingFace authentication.** Specifically:
1. Create a HF account if you don't have one
2. Visit https://huggingface.co/mistralai/Mistral-7B-v0.3 and click "Agree and access repository" at the top of the page
3. `huggingface-cli login` and paste a token from https://huggingface.co/settings/tokens
4. THEN `python scripts/download_mistral.py` will work

Mistral has the widest KV channel count (1024) and gives the best PCA spectrum concentration we measured. But you can validate the qualitative findings on Qwen 2.5 7B without any account setup.

After downloading any of them, point the capture script at it:

```bash
python scripts/capture_llm_kv.py --model mistralai/Mistral-7B-v0.3 --layer 16
# or
python scripts/capture_llm_kv.py --model Qwen/Qwen2.5-1.5B-Instruct --layer 14
```

## 7. The bandwidth argument

```bash
python poc/06_pcie_microbench.py             # ~30 sec — codec round-trip vs PCIe
python poc/07_parallel_path_demo.py          # ~1-2 min — pipelined cuda streams
python poc/08_wire_simulation.py             # ~1 min — end-to-end across simulated wires
python poc/09_dual_lane.py                   # ~3-5 min — dual-lane parallel offload
python poc/10_codec_overhead_breakdown.py    # ~30 sec — proves subprocess overhead is the bottleneck
python poc/11_codec_session_bench.py         # ~5 min — CodecSession (persistent NVENC) vs per-call backends, multiple batch sizes
python poc/12_multi_engine_bench.py          # ~5 min — MultiEngineCodecSession parallel across the GPU's NVENC engines
```

What to expect:

- **`06_pcie_microbench.py`** — codec round-trip ~600-700 ms vs PCIe ~5 ms. In isolation, codec loses on fast wires.
- **`07_parallel_path_demo.py`** — synthesises N tensors and demonstrates pipelined compression. Honest about subprocess overhead masking the wins.
- **`08_wire_simulation.py`** — measures REAL codec time + simulates wire transmission. Shows codec **winning today** on residential broadband (~3-5×) even with the slow pipeline. Loses on PCIe / 10 Gbit / 1 Gbit.
- **`09_dual_lane.py`** — runs direct PCIe and codec lanes concurrently on cuda streams. Shows ~1.7× wall-clock speedup on simulated 1 Gbit ethernet, ~2× on 100 Mbps, even with the slow pipeline.
- **`10_codec_overhead_breakdown.py`** — decomposes the 258 ms encode round-trip: ~171 ms (66%) is subprocess overhead, ~87 ms (34%) is real codec + I/O. Proves that an in-process wrapper (PyAV) would close the bulk of the gap without algorithmic changes.
- **`11_codec_session_bench.py`** — measured comparison of subprocess / per-call PyAV / CodecSession across batch sizes 1-16 on real PCA-rotated activations. CodecSession gives 1.77× speedup over subprocess and 1.45× over per-call PyAV by holding the NVENC encoder context open across multiple `compress()` calls. Quality slightly better, bitstream ~22% larger at same QP (offset by bumping QP a few notches).

Together these PoCs let you reproduce both the wins-today claims (slow wires, dual-lane) and the wins-with-fast-wrapper projections (PCIe, multi-GPU NVLink-replacement). See [`docs/parallel_path.md`](parallel_path.md) for the full math.

## 8. Null findings (~5 min total, optional)

```bash
python poc/null_findings/n1_sparse_residual.py
python poc/null_findings/n2_av1_vs_hevc.py
python poc/null_findings/n3_channel_reorder.py
```

Expect each to print honest "this didn't help, here's why" output. Short reads in [`poc/null_findings/README.md`](../poc/null_findings/README.md).

## Total time / disk budget

- Disk: ~40 GB downloaded models + ~2 GB captured tensors
- Time: ~1.5 hours wall-clock on a 100 Mbps connection (mostly downloads); ~30 minutes of actual compute

## Troubleshooting

- **`hevc_nvenc not found`** in `check_environment.py`: your FFmpeg build doesn't include NVENC. Reinstall from a build that does (Gyan.dev essentials on Windows; check `--enable-nvenc` in distro package).
- **CUDA out of memory** on FLUX capture: reduce `--width 512 --height 512` (smaller activations).
- **HuggingFace 401 unauthorized** on download: usually means you're trying Mistral 7B without first visiting the model page and clicking "Agree and access repository", OR without having run `huggingface-cli login`. See the section on Mistral above. FLUX.1-schnell, Qwen 2.5 1.5B, and Qwen 2.5 7B are NOT gated and should never need auth.
- **Numbers wildly different from the reference**: open an issue with `python scripts/check_environment.py` output and the offending PoC's full output.
