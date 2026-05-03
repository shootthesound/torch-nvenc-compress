"""Download FLUX.1-schnell via huggingface_hub.

FLUX.1-schnell is Apache 2.0 licensed (commercially permissive), 12B parameters,
distilled to 4 inference steps. This is the diffusion model the PoCs will hook
for activation capture. Approximate download size: ~24 GB.

If you already have FLUX.1-schnell cached, this script confirms the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -e '.[diffusion]'", file=sys.stderr)
        return 1

    print(f"Downloading {MODEL_ID} (~24 GB)...")
    print("This is cached by HuggingFace — running again is a no-op if already downloaded.")
    print()
    path = snapshot_download(
        repo_id=MODEL_ID,
        # Skip the original .safetensors single-file checkpoint to halve download time;
        # diffusers wants the split-by-component layout.
        ignore_patterns=["flux1-schnell.safetensors"],
    )
    print(f"\nReady at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
