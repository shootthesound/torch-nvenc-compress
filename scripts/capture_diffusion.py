"""Capture diffusion-model activations via the diffusers FluxPipeline.

Loads FLUX.1-schnell (Apache 2.0, 12B, 4-step distilled), registers a forward
hook on one mid-network MM-DiT block, runs a set of diverse prompts, saves
each block-output tensor to disk. The PoC scripts then load these tensors
and run the spectrum / PCA / Pareto analyses.

This replaces the ComfyUI custom node we used during research — pure-Python,
no external app required.

Usage:
    python scripts/capture_diffusion.py --num-prompts 32 --layer 9 --steps 4

Output files: data/diffusion/activation_NNN.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# Diverse prompts so the captured tensors span varied content statistics.
DEFAULT_PROMPTS = [
    "a fluffy cat sitting on a windowsill, soft afternoon light",
    "a busy city street at night with neon signs and crowds",
    "an aerial photo of a snow-covered mountain range at dawn",
    "a close-up portrait of an elderly man with a long white beard, sharp focus",
    "abstract geometric pattern, bauhaus style, primary colors",
    "a still life of fruit on a wooden table, soft window light",
    "a long exposure of a waterfall in dense forest, ethereal mist",
    "a cyberpunk street scene with rain and reflections",
    "a watercolor painting of a Tuscan villa at golden hour",
    "macro photograph of a butterfly wing, iridescent scales",
    "an astronaut floating in space with Earth in the background",
    "a 1920s art deco poster, geometric, gold and black",
    "a dense rainforest canopy from above with shafts of sunlight",
    "a pencil sketch of a violin on sheet music",
    "a ramen bowl with steam rising, top-down view",
    "a Victorian library with leather-bound books and a brass globe",
    "a herd of wild horses running across a beach at sunset",
    "neon underwater coral reef teeming with tropical fish",
    "a vintage steam locomotive in a snowy alpine valley",
    "a bustling Tokyo crossing at night with bright signage",
    "an oil painting of sunflowers in a vase, impasto texture",
    "a futuristic glass skyscraper reflecting a sunset",
    "a hand-drawn map of an imaginary fantasy continent",
    "two children playing with a paper boat in a puddle",
    "an industrial factory interior with massive machinery, dramatic lighting",
    "a meadow of wildflowers at peak bloom, soft focus background",
    "a Japanese zen garden with raked sand and stones",
    "a boxer mid-punch, motion blur, intense expression",
    "a desert canyon with red rock formations under blue sky",
    "a market stall in Marrakech with colorful spices",
    "a polar bear walking across drift ice",
    "a vintage typewriter on a wooden desk with a half-typed letter",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=32,
                        help="how many prompts to capture (max %d)" % len(DEFAULT_PROMPTS))
    parser.add_argument("--layer", type=int, default=9,
                        help="which MM-DiT double-stream block to hook (0-indexed)")
    parser.add_argument("--steps", type=int, default=4,
                        help="diffusion inference steps (FLUX.1-schnell is distilled to 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="generator seed (kept fixed across prompts so only prompt varies)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/diffusion"),
                        help="where to save activation_NNN.pt files")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell",
                        help="HuggingFace model ID")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    try:
        from diffusers import FluxPipeline
    except ImportError:
        print("diffusers not installed. Run: pip install -e '.[diffusion]'", flush=True)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    pipe = FluxPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    n_blocks = len(pipe.transformer.transformer_blocks)
    if args.layer >= n_blocks:
        print(f"--layer {args.layer} out of range (model has {n_blocks} double-stream blocks)")
        return 1
    print(f"transformer has {n_blocks} double-stream blocks; hooking block {args.layer}")

    captured: dict = {"tensor": None}

    def hook(_module, _inputs, output):
        # diffusers FLUX double-stream block returns (encoder_hidden_states, hidden_states)
        # i.e. (txt_stream, img_stream). We want the img stream — it's the spatial activation.
        if isinstance(output, tuple) and len(output) == 2:
            _txt, img = output
        else:
            img = output
        captured["tensor"] = img.detach().to(device="cpu").clone()

    handle = pipe.transformer.transformer_blocks[args.layer].register_forward_hook(hook)

    n = min(args.num_prompts, len(DEFAULT_PROMPTS))
    try:
        for i in range(n):
            prompt = DEFAULT_PROMPTS[i]
            captured["tensor"] = None
            generator = torch.Generator("cuda").manual_seed(args.seed)
            _ = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=0.0,            # FLUX.1-schnell is distilled, no real CFG
                width=args.width,
                height=args.height,
                generator=generator,
                output_type="latent",          # we don't need the decoded image
            )
            if captured["tensor"] is None:
                print(f"  [{i}] WARNING: no tensor captured — block {args.layer} not invoked?")
                continue
            t = captured["tensor"]
            out_path = args.output_dir / f"activation_{i:03d}.pt"
            torch.save({
                "tensor": t,
                "block_index": args.layer,
                "model": args.model,
                "prompt": prompt,
                "shape": tuple(t.shape),
                "dtype_orig": str(t.dtype),
            }, out_path)
            print(f"  [{i}] {prompt[:50]}... -> {out_path.name}  shape={tuple(t.shape)}  "
                  f"size={out_path.stat().st_size/1e6:.1f} MB")
    finally:
        handle.remove()

    print(f"\nDone. Captured {n} activation tensors to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
