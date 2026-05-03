# Null findings

These are things we tried that did **not** crack the Pareto open. We ship them as runnable PoCs so anyone considering the same path can verify our results in 5 minutes instead of weeks.

| # | Script | Expectation going in | What we measured |
|---|---|---|---|
| n1 | [`n1_sparse_residual.py`](n1_sparse_residual.py) | Send the codec output, then send only the worst-error positions as sparse corrections — should close most of the gap to lossless cheaply. | Reconstruction error is uniformly distributed across positions, NOT outlier-concentrated. Even 10% of positions (10 MB extra) only adds ~0.02 cos. The CLAUDE.md spec's hypothesis that residual would be sparse turns out to be wrong for this data. |
| n2 | [`n2_av1_vs_hevc.py`](n2_av1_vs_hevc.py) | AV1's better entropy coding should beat HEVC's at the same quality. | Blackwell's AV1 NVENC doesn't support 4:4:4 ("No capable devices found"). The 4:2:0 + 1-channel-per-Y-plane workaround has 3× more frames per "video", which negates AV1's coding-efficiency advantage. HEVC 4:4:4 wins. |
| n3 | [`n3_channel_reorder.py`](n3_channel_reorder.py) | HEVC uses inter-frame P-frame prediction. If we order PCA-rotated channels by similarity (so adjacent frames contain similar channels), the codec's temporal prediction should compress much better. | PCA orthogonalises by construction. PCA-rotated channels have zero linear cross-channel correlation. Greedy similarity-ordering finds residual non-linear correlations, but they're too weak for the codec to exploit. Result: 37.04× vs baseline 37.25× — within noise. |
| n4 | [`n4_pynvvideocodec.py`](n4_pynvvideocodec.py) | NVIDIA's PyNvVideoCodec gives lower-level NVENC access than PyAV. Persistent encoder + GPU-input zero-copy should beat PyAV CodecSession for batch workloads. | The NVENC hardware pipeline holds 2 frames in flight at all times. EndEncode() only flushes 1. So per-tensor packet boundaries are non-deterministic with persistent context. PyAV (bf=0) handles this internally; PyNvVideoCodec doesn't expose the necessary control. Direct Video Codec SDK is the only path to finer pipeline control. |

## Why these are still useful

- **Saves duplicated effort** for anyone else investigating the same problem space.
- **Documents the empirical limits** of the (PCA + uniform per-channel quant + standard video codec) approach. To go further you genuinely need joint spatio-channel transforms (3D DCT, tensor decomposition), variable bit-depth quantisation, or learned compression — each a real R&D project, not a 30-minute experiment.
- **Helps reviewers**: when this work gets discussed in research/community forums, "did you try sparse residual / AV1 / channel reordering?" will be asked. Having the runnable answer ready beats having to re-justify in conversation.
