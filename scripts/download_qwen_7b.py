"""Download Qwen 2.5 7B (base) via huggingface_hub.

Apache 2.0, ungated, ~15 GB. Use this when you want a wide-channel KV cache
test without the Mistral gating dance.

Architecture: 28 layers, num_key_value_heads=4, head_dim=128 => 512 KV channels
per layer. Less wide than Mistral 7B v0.3's 1024, but still wide enough to
get a meaningful PCA spectrum and a different operating point than the
Qwen 1.5B (256 channels) baseline.
"""

from __future__ import annotations

import sys

MODEL_ID = "Qwen/Qwen2.5-7B"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -e '.[llm]'", file=sys.stderr)
        return 1

    print(f"Downloading {MODEL_ID} (~15 GB)...")
    print()
    path = snapshot_download(
        repo_id=MODEL_ID,
        ignore_patterns=["*.bin", "*.pt"],     # prefer safetensors
    )
    print(f"\nReady at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
