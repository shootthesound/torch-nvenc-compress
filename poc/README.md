# PoCs — proofs of concept

Numbered for the recommended reading / running order.

| # | Script | Demonstrates | Needs model? |
|---|---|---|---|
| 01 | [`01_synthetic_controls.py`](01_synthetic_controls.py) | Pipeline sanity: zeros 600×, smooth 70×, Gaussian noise ~4× | No |
| 02 | [`02_spectrum_diffusion.py`](02_spectrum_diffusion.py) | Diffusion activation channel-covariance spectrum is heavy-tailed | Yes (FLUX) |
| 03 | [`03_diffusion_pareto.py`](03_diffusion_pareto.py) | Full LOO PCA + truncation + codec sweep on diffusion activations | Yes (FLUX) |
| 04 | [`04_spectrum_llm_kv.py`](04_spectrum_llm_kv.py) | LLM KV cache spectrum (K and V separately) | Yes (LLM) |
| 05 | [`05_llm_kv_pareto.py`](05_llm_kv_pareto.py) | LOO PCA + truncation + codec sweep on KV cache | Yes (LLM) |
| 06 | [`06_pcie_microbench.py`](06_pcie_microbench.py) | Codec round-trip vs PCIe latency, decomposed by stage | No (just one captured tensor) |
| 07 | [`07_parallel_path_demo.py`](07_parallel_path_demo.py) | Pipelined codec + PCIe using cuda streams; the wall-clock argument | No |
| 08 | [`08_wire_simulation.py`](08_wire_simulation.py) | End-to-end wall-clock across simulated wires (PCIe / 1 Gbit / 100 Mbps); shows codec winning on slow wires today | Optional |
| 09 | [`09_dual_lane.py`](09_dual_lane.py) | Dual-lane offload: half tensors via direct PCIe, half via codec, concurrent on cuda streams | No |
| 10 | [`10_codec_overhead_breakdown.py`](10_codec_overhead_breakdown.py) | Decomposes the 258 ms encode round-trip into subprocess overhead vs real codec work; proves PyAV would close 66% of the gap | No |
| 11 | [`11_codec_session_bench.py`](11_codec_session_bench.py) | CodecSession (persistent NVENC context) vs subprocess vs per-call PyAV across batch sizes; measures the 1.77x batch-workload speedup | Yes (real captures) |
| 12 | [`12_multi_engine_bench.py`](12_multi_engine_bench.py) | MultiEngineCodecSession — distributes encodes across the GPU's multiple NVENC engines (5090: 3, H100: 4) via parallel threads; measures 2.81x speedup over subprocess on 5090 | Yes (real captures) |
| 13 | [`13_direct_nvenc_scaffold.py`](13_direct_nvenc_scaffold.py) | Direct NVENC ctypes scaffold — loads driver-shipped nvEncodeAPI64.dll, calls NvEncodeAPICreateInstance, verifies the API table is populated and 39 function pointers map. | No |
| 14 | [`14_direct_nvenc_first_frame.py`](14_direct_nvenc_first_frame.py) | First end-to-end NVENC encode through the direct ctypes path: init encoder, register input/output buffers, write a synthetic YUV444 frame, call nvEncEncodePicture, read out the HEVC bitstream. | No |
| 15 | [`15_direct_nvdec_round_trip.py`](15_direct_nvdec_round_trip.py) | Full encode + decode round-trip through pure ctypes (NVENC + nvcuvid). Verifies PSNR > 30 dB after the round-trip — confirms the NVDEC binding is correct. | No |
| 16 | [`16_direct_backend_bench.py`](16_direct_backend_bench.py) | DirectBackend bench vs PyAV CodecSession on synthetic frames. Measures the host-buffer path, the zero-copy CUDA-tensor path, and the round-trip diff. **Encode 0.22 ms/frame zero-copy vs 0.45 ms/frame PyAV (2.0×); decode 1.55 ms/frame vs 5.42 (3.4×).** | No |
| 17 | [`17_parallel_path_demo.py`](17_parallel_path_demo.py) | The killer demo: GEMM on stream A + DirectBackend encode on stream B with `nvEncSetIOCudaStreams`. Measures **67% of theoretical-max overlap realized** — confirms NVENC silicon is independent of SM compute. | No |
| 18 | [`18_real_activation_bench.py`](18_real_activation_bench.py) | Real-workload bench on captured FLUX.2 Klein 9B activations through the full PCA + quant + YUV pack pipeline. Compares pyav-single, pyav-multi, DirectBackend, and MultiEngineDirectBackend (3 engines). **End-to-end: 2.83× over PyAV CodecSession** at equal-or-better cos-sim quality. | Yes (FLUX captures from ring0/data/) |
| 19 | [`19_direct_vs_pyav_diff.py`](19_direct_vs_pyav_diff.py) | Diagnoses the bitstream/quality divergence between DirectBackend and PyAV. ffprobes both bitstreams, walks Annex-B NAL units, finds the root cause: PyAV's `pict_type=I` doesn't propagate to NVENC's FORCEIDR flag. | No |

## Recommended order

1. Run **01** first — confirms the full toolchain works in 2 minutes with no downloads.
2. Run **02 + 03** for the diffusion side. Requires capturing activations from FLUX.1-schnell (see `scripts/capture_diffusion.py`).
3. Run **04 + 05** for the LLM side. Requires capturing KV from Mistral 7B v0.3 (see `scripts/capture_llm_kv.py`).
4. Run **06 + 07** for the bandwidth argument that justifies why this matters at all.
5. Run **11 + 12** to see PyAV's CodecSession + MultiEngineCodecSession results — these were the fast path before the direct-SDK work landed.
6. Run **13 → 19** to walk the direct Video Codec SDK build-up: scaffold → first frame → round-trip → bench → parallel-path demo → real-workload bench → diagnostic. This is the work that brings us to ~75% of the NVLink-replacement claim.

## Null findings

[`null_findings/`](null_findings/) contains three runnable PoCs for things we tried that did NOT crack the Pareto open. We ship them so you don't have to re-run the same dead-ends.

| # | Script | Why it doesn't help |
|---|---|---|
| n1 | [`null_findings/n1_sparse_residual.py`](null_findings/n1_sparse_residual.py) | Reconstruction error is uniformly distributed across positions, not outlier-concentrated. Even 10% of positions corrected only adds ~0.02 cos. |
| n2 | [`null_findings/n2_av1_vs_hevc.py`](null_findings/n2_av1_vs_hevc.py) | AV1 NVENC 4:4:4 is unsupported on Blackwell. The 4:2:0 + 1-channel-per-Y-plane workaround loses to HEVC 4:4:4 due to 3× frame-count overhead. |
| n3 | [`null_findings/n3_channel_reorder.py`](null_findings/n3_channel_reorder.py) | PCA orthogonalises by construction, so PCA-rotated channels have zero linear cross-channel correlation. P-frame prediction has nothing to exploit. |
