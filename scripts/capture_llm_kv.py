"""Capture LLM KV cache from a HuggingFace causal LM.

Loads a transformer model, runs a set of long prompts with `use_cache=True`,
saves the K and V tensors of one mid-network layer per prompt. The PoC scripts
load these and run spectrum / PCA / Pareto analyses.

Default model is Qwen 2.5 7B (512 KV channels, ungated). Pass `--model` to
use a different HF causal LM. Two recommended alternatives:

    --model mistralai/Mistral-7B-v0.3   # 1024 KV channels (wider, slightly
                                        #   better PCA spectrum) but GATED —
                                        #   needs `huggingface-cli login`
                                        #   and accepting terms on HF first.
    --model Qwen/Qwen2.5-1.5B-Instruct  # 256 KV channels, fast 3 GB download

Usage:
    python scripts/capture_llm_kv.py --num-prompts 8 --layer 14

Output files: data/kv/kv_NNN_layer{L}_K.pt and ..._V.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# Long-ish prompts spanning diverse domains so KV statistics are varied.
# Each is intentionally padded to ~2-3K tokens by repetition.
DEFAULT_PROMPTS = [
    "Write a Python implementation of a B-tree with insertion, deletion, and range queries. "
    "Include detailed docstrings, type hints, and a comprehensive test suite. Discuss the "
    "tradeoffs vs other data structures. Now extend it to support concurrent access with a "
    "reader-writer lock, and explain how this changes the time complexity analysis. " * 30,

    "Tell me a long story set in a Victorian-era detective agency where the protagonist "
    "investigates a series of strange occurrences in a foggy port town. Include rich "
    "descriptions of the setting, character development, atmospheric tension, and a twist "
    "ending. The story should weave together multiple subplots involving smuggling, family "
    "secrets, and an old maritime curse. " * 30,

    "Explain in depth how the human auditory system processes sound, from the outer ear "
    "through the cochlea to the auditory cortex. Cover the mechanical, electrical, and "
    "neural aspects, the role of hair cells, frequency mapping, sound localization, and "
    "how the brain reconstructs spatial audio. Include relevant equations and clinical "
    "implications of damage at each stage. " * 30,

    "Analyze the major schools of philosophical ethics — virtue ethics, deontology, "
    "consequentialism, contractualism, care ethics — and discuss how each would approach "
    "the trolley problem and its variants. Then critique the trolley problem itself as a "
    "tool for moral reasoning. " * 30,

    "Hi! I'm planning a two-week trip through northern Spain with my partner. We love "
    "hiking, food, and small towns away from tourist crowds. Can you suggest a detailed "
    "itinerary covering Asturias, Cantabria, the Basque Country, and Galicia, with "
    "specific restaurant recommendations, hike difficulty levels, accommodation tips, "
    "and how to get between regions without a car? " * 30,

    "Prove that the sum of the first n cubes equals the square of the sum of the first n "
    "natural numbers. Then generalise: for which other powers k does sum_{i=1}^{n} i^k "
    "have a nice closed form? Discuss Bernoulli numbers, the Faulhaber formula, and the "
    "connection to the Riemann zeta function. " * 30,

    "Discuss the development of jazz music from the early 1900s through bebop, cool jazz, "
    "modal jazz, free jazz, and fusion. Highlight key musicians, recordings, and how each "
    "style was a response to or rejection of what came before. " * 30,

    "Compare and contrast the rise of digital photography against the simultaneous "
    "persistence of film photography. Discuss the technical, aesthetic, and cultural "
    "factors driving both. Cover sensor technology, color science, dynamic range, "
    "the rise of mobile photography, and the recent revival of medium-format film. " * 30,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="HuggingFace causal LM model ID (default Qwen 2.5 7B; "
                             "ungated Apache 2.0)")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--layer", type=int, default=14,
                        help="which transformer layer's KV to capture (mid-network "
                             "for the default Qwen 2.5 7B's 28 layers)")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="max tokens per prompt (truncated)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/kv"))
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed. Run: pip install -e '.[llm]'", flush=True)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    if args.layer >= n_layers:
        print(f"--layer {args.layer} out of range (model has {n_layers} layers)")
        return 1
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"hidden_size={model.config.hidden_size} layers={n_layers} "
          f"num_kv_heads={model.config.num_key_value_heads} head_dim={head_dim} "
          f"=> {model.config.num_key_value_heads * head_dim} KV channels per layer")

    n = min(args.num_prompts, len(DEFAULT_PROMPTS))
    for i in range(n):
        prompt = DEFAULT_PROMPTS[i]
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=args.max_length,
        ).to("cuda")
        seq_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True)

        past = outputs.past_key_values
        if hasattr(past, "layers"):                # transformers >=4.40 DynamicCache
            k = past.layers[args.layer].keys
            v = past.layers[args.layer].values
        elif hasattr(past, "key_cache"):           # alternate API
            k = past.key_cache[args.layer]
            v = past.value_cache[args.layer]
        else:                                      # legacy tuple
            k, v = past[args.layer]

        for kind, t in (("K", k), ("V", v)):
            out_path = args.output_dir / f"kv_{i:03d}_layer{args.layer}_{kind}.pt"
            torch.save({
                "tensor": t.detach().to(device="cpu", dtype=torch.float32).clone(),
                "shape": tuple(t.shape),
                "dtype_orig": str(t.dtype),
                "layer": args.layer,
                "kind": kind,
                "seq_len": seq_len,
                "model": args.model,
                "num_kv_heads": model.config.num_key_value_heads,
                "head_dim": head_dim,
            }, out_path)
            print(f"  [{i}] {kind}: {out_path.name}  shape={tuple(t.shape)}  "
                  f"size={out_path.stat().st_size/1e6:.2f} MB")

    print(f"\nDone. {n*2} files written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
