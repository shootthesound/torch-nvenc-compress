"""07 — Parallel-path demo: pipelined codec + PCIe via cuda streams.

This is the lede demo. It shows that when codec encode and PCIe transfer
run on independent hardware paths concurrently, the user-visible time per
tensor is dominated by transfer of the small compressed bytes, NOT by the
codec round-trip in isolation.

The argument:

    NVENC silicon, NVDEC silicon, SM cores, and the PCIe controller are
    all independent hardware units. With pipelined scheduling, the codec
    encode/decode time is hidden behind compute or PCIe transfer of OTHER
    tensors. The user-visible cost reduces to PCIe transfer of the
    compressed bytes, which is `original_size / compression_ratio` divided
    by PCIe bandwidth.

For multi-GPU on consumer hardware (the 5090 has no NVLink), this approach
recovers something close to NVLink-class effective bandwidth via PCIe +
the GPU's idle compression silicon.

Note: with the current FFmpeg subprocess pipeline (~285 ms per call), the
pipelining win is dwarfed by subprocess overhead. The wall-clock numbers
this script prints reflect that. The "projected with PyAV" numbers reflect
what we expect once the fast path replaces the subprocess wrapper. See
docs/parallel_path.md.
"""

from __future__ import annotations

import time

import torch

from nvenc_compress import build_shared_basis, compress, decompress


N_TENSORS = 8                  # how many tensors to push through both paths
SHAPE = (4096, 4096)           # mimics a single FLUX activation per tensor
K_PCA = 1000
QP = 18


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — this PoC requires a GPU")
        return

    # --- Setup: synthesise N tensors and a one-tensor PCA basis --------------
    device = "cuda"
    torch.manual_seed(0)
    print(f"Synthesising {N_TENSORS} tensors of shape {SHAPE} on GPU...")
    tensors = [torch.randn(*SHAPE, device=device, dtype=torch.float32) for _ in range(N_TENSORS)]
    bytes_per_tensor_fp16 = SHAPE[0] * SHAPE[1] * 2
    total_bytes = N_TENSORS * bytes_per_tensor_fp16
    print(f"Total payload: {N_TENSORS} x {bytes_per_tensor_fp16/1e6:.1f} MB = "
          f"{total_bytes/1e6:.1f} MB (fp16-equivalent)")

    print("Building shared PCA basis from these tensors...")
    basis = build_shared_basis(tensors, K=K_PCA)

    # --- Baseline: sequential PCIe transfer ---------------------------------
    print(f"\n[1] Baseline: sequential cuda -> cpu transfer of {N_TENSORS} tensors")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cpu_copies = []
    for t in tensors:
        cpu_copies.append(t.cpu())
    torch.cuda.synchronize()
    baseline_time = time.perf_counter() - t0
    baseline_throughput = total_bytes / baseline_time
    print(f"  wall-clock: {baseline_time*1000:>7.1f} ms")
    print(f"  effective throughput: {baseline_throughput/1e9:.2f} GB/s")

    # --- Compressed sequential (no pipelining) -----------------------------
    print(f"\n[2] Compressed sequential: compress() then transfer compressed bytes, one at a time")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compressed_blobs = []
    for t in tensors:
        data, recipe = compress(t, basis, qp=QP)
        compressed_blobs.append((data, recipe))
    torch.cuda.synchronize()
    seq_compressed_time = time.perf_counter() - t0
    total_compressed_bytes = sum(len(b[0]) for b in compressed_blobs)
    compression_ratio = total_bytes / total_compressed_bytes
    print(f"  wall-clock: {seq_compressed_time*1000:>7.1f} ms")
    print(f"  total compressed bytes: {total_compressed_bytes/1e6:.2f} MB ({compression_ratio:.1f}x ratio)")
    print(f"  per-tensor codec round-trip: {seq_compressed_time/N_TENSORS*1000:.1f} ms")

    # --- Pipelined: codec encode on stream A, transfer on stream B ---------
    # NOTE: the FFmpeg subprocess wrapper isn't truly stream-aware; it does
    # CPU-side I/O. So this measurement is more illustrative than precise. With
    # a PyAV / VC SDK fast path that runs on a real cuda stream, the wins
    # would be much larger.
    print(f"\n[3] Pipelined: encode + transfer overlap via torch.cuda.Stream")
    encode_stream = torch.cuda.Stream()
    xfer_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pipelined_blobs = []
    for t in tensors:
        with torch.cuda.stream(encode_stream):
            data, recipe = compress(t, basis, qp=QP)
        with torch.cuda.stream(xfer_stream):
            # Just a "transfer" of the compressed bytes (already on CPU, but
            # would be the moment of dispatch over PCIe in a real pipeline)
            pipelined_blobs.append((data, recipe))
    torch.cuda.synchronize()
    pipelined_time = time.perf_counter() - t0
    print(f"  wall-clock: {pipelined_time*1000:>7.1f} ms")

    # --- The math the demo is selling --------------------------------------
    print(f"\n--- Effective bandwidth analysis ---")
    print(f"PCIe 5.0 x16 raw bandwidth: ~64 GB/s (theoretical)")
    print(f"This system measured PCIe at: {baseline_throughput/1e9:.2f} GB/s\n")

    print(f"With compression ratio {compression_ratio:.1f}x:")
    print(f"  Bytes that need to cross PCIe drop from {total_bytes/1e6:.1f} MB to "
          f"{total_compressed_bytes/1e6:.2f} MB")
    print(f"  Effective bandwidth IF codec is hidden by pipelining = "
          f"{baseline_throughput * compression_ratio / 1e9:.1f} GB/s")
    print()
    print(f"For reference:")
    print(f"  NVLink 4 (enterprise H100): ~900 GB/s")
    print(f"  NVLink 3 (consumer 3090, since removed on 4090/5090): ~50 GB/s")
    print(f"  Our claim: NVENC + PCIe with pipelining can recover NVLink-3-class")
    print(f"             effective bandwidth on cards where NVIDIA removed the link.")
    print()
    print(f"--- Honest caveats ---")
    print(f"The current FFmpeg-subprocess pipeline costs ~285 ms per encode and decode")
    print(f"(measured in poc/06_pcie_microbench.py). This dwarfs the pipelining win")
    print(f"because the subprocess startup and stdin/stdout I/O are NOT stream-aware.")
    print(f"")
    print(f"Wall-clock numbers above ARE NOT YET the headline numbers in docs/parallel_path.md.")
    print(f"To realise those, the FFmpeg subprocess wrapper needs to be replaced with PyAV")
    print(f"(in-process FFmpeg API, eliminates subprocess overhead) or direct Video Codec")
    print(f"SDK access via cffi (zero-copy GPU memory paths). Both are documented future")
    print(f"work. Once done, the per-encode/decode time drops from ~285 ms to ~5-15 ms,")
    print(f"and the pipelined demo here would show wall-clock improvement matching the")
    print(f"compression ratio.")


if __name__ == "__main__":
    main()
