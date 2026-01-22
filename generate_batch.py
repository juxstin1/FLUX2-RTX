"""
Flux2 Klein Local - Batch Generation Script
Generate multiple images from prompts.txt

Usage:
    python generate_batch.py [options]

Options:
    --steps N      Inference steps (default: 28)
    --upscale      Enable 4x upscaling
"""

import argparse
import os
from datetime import datetime

import torch
from PIL import Image

from generate import load_pipeline, generate_image, load_upscaler


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate images from prompts.txt")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps (default: 28)")
    parser.add_argument("--upscale", action="store_true", help="Enable 4x upscaling")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    return parser.parse_args()


def load_prompts(filepath="prompts.txt"):
    """Load prompts from file."""
    if not os.path.exists(filepath):
        # Create default prompts file
        default_prompts = [
            "a cyberpunk samurai standing in neon-lit rain on a Tokyo street at night, cinematic lighting, hyperrealistic",
            "an ancient library floating in the clouds, golden hour sunlight streaming through grand arched windows, photorealistic",
            "a bioluminescent alien jellyfish creature in a deep underwater cave, ethereal cyan and magenta glow",
            "a massive steampunk airship docking at a Victorian floating sky city at golden sunset, epic scale",
            "a majestic nine-tailed kitsune fox spirit in an enchanted cherry blossom forest, magical particles"
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            for prompt in default_prompts:
                f.write(prompt + "\n")

        print(f"Created {filepath} with sample prompts")
        return default_prompts

    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return prompts


def main():
    args = parse_args()

    print()
    print("=" * 50)
    print("  FLUX.2 [klein] 4B - Batch Generation")
    print("=" * 50)
    print()

    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from prompts.txt")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"batch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline()

    # Load upscaler if needed
    upscaler = None
    if args.upscale:
        print()
        upscaler = load_upscaler()

    # Generate images
    print()
    print(f"Generating {len(prompts)} images...")
    print(f"Settings: {args.width}x{args.height}, {args.steps} steps, upscale={args.upscale}")
    print()

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt[:60]}...")

        # Generate
        image = generate_image(
            pipe,
            prompt,
            args.width,
            args.height,
            args.steps,
            seed=42 + i  # Reproducible seeds
        )

        # Upscale
        if upscaler:
            print("  Upscaling 4x...")
            torch.cuda.empty_cache()
            image = upscaler.upscale(image)

        # Save
        output_path = os.path.join(output_dir, f"image_{i:02d}.png")
        image.save(output_path, quality=95)
        print(f"  Saved: {output_path} ({image.size[0]}x{image.size[1]})")

    print()
    print("=" * 50)
    print(f"  Batch complete! {len(prompts)} images generated")
    print(f"  Output: {output_dir}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
