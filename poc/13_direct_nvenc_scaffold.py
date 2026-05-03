"""13 — Direct NVENC ctypes scaffold (Option D, sessions 1+2 deliverables).

Demonstrates the working foundation of the direct Video Codec SDK backend,
through pure ctypes + cuda-python with NO PyAV / PyNvVideoCodec dependency
in the codec path:

  1. Load nvEncodeAPI64.dll (Windows) / libnvidia-encode.so (Linux) directly
  2. Call NvEncodeAPICreateInstance, get the 39-function API table
  3. Initialise CUDA driver, get current CUDA context handle
  4. Call nvEncOpenEncodeSessionEx with the CUDA context — get encoder handle
  5. Enumerate supported codecs (nvEncGetEncodeGUIDs)
  6. Enumerate presets for HEVC (nvEncGetEncodePresetGUIDs)
  7. Query preset config defaults (nvEncGetEncodePresetConfigEx)
  8. Destroy encoder cleanly via nvEncDestroyEncoder

This is the open/query/destroy lifecycle. The full first-frame encode
(nvEncInitializeEncoder + buffer creation + EncodePicture + LockBitstream)
is exercised in `14_direct_nvenc_first_frame.py`.

Code lives in `src/nvenc_compress/direct/` — see that module's docstring
for the full session-by-session roadmap.
"""

from __future__ import annotations

import sys

import torch
from cuda.bindings import driver as cuda

from nvenc_compress.direct import (
    create_instance, open_encode_session_cuda, destroy_encoder,
    get_encode_guids, get_preset_guids,
    NV_ENC_CODEC_HEVC_GUID, NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_AV1_GUID,
    NV_ENC_PRESET_P1_GUID, NV_ENC_PRESET_P4_GUID, NV_ENC_PRESET_P7_GUID,
    NVENCAPI_VERSION,
)
from nvenc_compress.direct.structs import (
    get_preset_config_ex,
)


def main() -> int:
    print("Direct NVENC scaffold demo (sessions 1+2 of D)\n")

    # 1: Initialize CUDA via cuda-python
    print("[1] Initialise CUDA driver...")
    err, = cuda.cuInit(0)
    if int(err) != 0:
        print(f"    cuInit failed: {err}")
        return 1
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    err, ctx = cuda.cuCtxGetCurrent()
    if int(ctx) == 0:
        print(f"    cuCtxGetCurrent returned NULL context")
        return 1
    print(f"    cuCtxGetCurrent: OK, ctx handle 0x{int(ctx):x}\n")

    # 2: Create NVENC API instance
    print("[2] NvEncodeAPICreateInstance via ctypes...")
    table = create_instance()
    print(f"    table version: 0x{table.version:08x}")
    print(f"    NVENCAPI version: 0x{NVENCAPI_VERSION:08x}\n")

    # 3: Open encoder session for the CUDA context
    print("[3] nvEncOpenEncodeSessionEx with our CUDA context...")
    encoder = open_encode_session_cuda(table, int(ctx))
    print(f"    encoder handle: 0x{encoder.value:x}\n")

    # 4: Enumerate codecs
    print("[4] nvEncGetEncodeGUIDs — enumerate supported codecs...")
    codecs = get_encode_guids(table, encoder)
    known = {
        NV_ENC_CODEC_H264_GUID: "H.264",
        NV_ENC_CODEC_HEVC_GUID: "HEVC",
        NV_ENC_CODEC_AV1_GUID: "AV1",
    }
    for g in codecs:
        print(f"    {known.get(g, '???'):<10s} {g}")
    print(f"    HEVC supported: {NV_ENC_CODEC_HEVC_GUID in codecs}\n")

    # 5: Enumerate HEVC presets
    print("[5] nvEncGetEncodePresetGUIDs — presets for HEVC...")
    presets = get_preset_guids(table, encoder, NV_ENC_CODEC_HEVC_GUID)
    preset_known = {
        NV_ENC_PRESET_P1_GUID: "P1 (fastest)",
        NV_ENC_PRESET_P4_GUID: "P4 (default)",
        NV_ENC_PRESET_P7_GUID: "P7 (best)",
    }
    for g in presets:
        print(f"    {preset_known.get(g, '?'):<15s} {g}")
    print()

    # 6: Query preset config defaults
    print("[6] nvEncGetEncodePresetConfigEx — query P4 default config...")
    pc = get_preset_config_ex(table, encoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P4_GUID)
    cfg = pc.presetCfg
    print(f"    gopLength={cfg.gopLength}, frameIntervalP={cfg.frameIntervalP}")
    print(f"    rateControlMode={cfg.rcParams.rateControlMode}, qpInterP={cfg.rcParams.constQP.qpInterP}\n")

    # 7: Destroy
    print("[7] nvEncDestroyEncoder...")
    destroy_encoder(table, encoder)
    print(f"    OK — clean lifecycle\n")

    print("Sessions 1+2 deliverables: ctypes loading, full encoder lifecycle,")
    print("GUID enumeration, preset config queries — all working through pure")
    print("ctypes + cuda-python with NO PyAV / PyNvVideoCodec in the codec path.")
    print()
    print("Next milestone (poc/14): nvEncInitializeEncoder + nvEncEncodePicture")
    print("for the full first-frame encode path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
