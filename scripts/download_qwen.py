"""Download Qwen 2.5 1.5B Instruct via huggingface_hub.

Small (~3 GB), fast download, Apache 2.0 licensed. Useful as a quick alternative
to Mistral 7B for KV cache PoCs when you don't want to wait for a 14 GB download.

The KV cache lossless ratio is similar (~2.7-2.8x) regardless of model size,
so the qualitative findings reproduce on the small model. Mistral gives more
heavy-tailed PCA spectrum (1024 channels vs 256) so its lossy curve is better.
"""

from __future__ import annotations

import sys

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -e '.[llm]'", file=sys.stderr)
        return 1

    print(f"Downloading {MODEL_ID} (~3 GB)...")
    print()
    path = snapshot_download(repo_id=MODEL_ID)
    print(f"\nReady at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
