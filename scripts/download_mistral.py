"""Download Mistral 7B v0.3 (base) via huggingface_hub.

Mistral 7B base, Apache 2.0 licensed, ~14 GB. Used for LLM KV cache PoCs.
Has 8 KV heads x 128 head_dim = 1024 KV channels per layer — wide enough
for the PCA spectrum analysis to be informative.

GATING NOTE: although Apache 2.0, this model is gated on HuggingFace —
you must (a) have a HF account, (b) visit https://huggingface.co/mistralai/Mistral-7B-v0.3
and click "Agree and access repository", and (c) authenticate via
`huggingface-cli login`. If you don't want to do that, use one of the
ungated alternatives:

    python scripts/download_qwen_7b.py     # Qwen 2.5 7B, 512 KV channels, ~15 GB
    python scripts/download_qwen.py        # Qwen 2.5 1.5B, 256 KV channels, ~3 GB
"""

from __future__ import annotations

import sys

MODEL_ID = "mistralai/Mistral-7B-v0.3"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -e '.[llm]'", file=sys.stderr)
        return 1

    print(f"Downloading {MODEL_ID} (~14 GB)...\n")
    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            ignore_patterns=["consolidated.safetensors", "*.bin", "*.pt"],
        )
    except GatedRepoError:
        print("\n" + "=" * 70)
        print("GATED MODEL — manual access acceptance required.\n")
        print("To use Mistral 7B v0.3:")
        print(f"  1. Visit https://huggingface.co/{MODEL_ID}")
        print("     and click 'Agree and access repository' (top of page).")
        print("  2. Run `huggingface-cli login` and paste a token from")
        print("     https://huggingface.co/settings/tokens (read access is enough).")
        print("  3. Re-run this script.")
        print()
        print("Or use a fully ungated alternative:")
        print("  python scripts/download_qwen_7b.py    # 7B, 512 KV channels")
        print("  python scripts/download_qwen.py       # 1.5B, 256 KV channels")
        print("=" * 70)
        return 2
    except RepositoryNotFoundError as e:
        print(f"\nFailed to find {MODEL_ID}: {e}")
        print("Either the model ID changed, or you need `huggingface-cli login`.")
        return 2

    print(f"\nReady at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
