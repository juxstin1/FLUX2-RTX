"""
Flux2 Klein Local - Image Generation Script
Generate images with FLUX.2 [klein] 4B and optional 4x upscaling.

Usage:
    python generate.py "your prompt here" [options]

Examples:
    python generate.py "a sunset over mountains"
    python generate.py "cyberpunk city at night" --steps 28
    python generate.py "fantasy castle" --steps 28 --upscale
"""

import argparse
import os
import sys
from datetime import datetime

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with FLUX.2 [klein] 4B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py "a beautiful sunset"
  python generate.py "cyberpunk city" --steps 28
  python generate.py "fantasy art" --steps 28 --upscale
  python generate.py "portrait" --width 768 --height 1024 --seed 42
        """
    )
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps: 4=fast, 28=quality (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--upscale", action="store_true", help="Enable 4x Real-ESRGAN upscaling")
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: auto)")
    parser.add_argument("--no-offload", action="store_true", help="Disable CPU offloading (uses more VRAM)")
    return parser.parse_args()


def load_pipeline(use_offload=True):
    """Load the Flux2 Klein pipeline."""
    from diffusers import Flux2KleinPipeline

    print("Loading FLUX.2 [klein] 4B pipeline...")
    print("(First run downloads ~8GB of models, please wait)")

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16
    )

    if use_offload:
        pipe.enable_model_cpu_offload()
        print("CPU offloading enabled (saves VRAM)")
    else:
        pipe.to("cuda")
        print("Full GPU mode (faster but uses more VRAM)")

    return pipe


def load_upscaler():
    """Load Real-ESRGAN 4x upscaler."""
    from upscaler import Upscaler
    return Upscaler()


def generate_image(pipe, prompt, width, height, steps, seed=None):
    """Generate a single image."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        print(f"Using seed: {seed}")

    print(f"Generating {width}x{height} with {steps} steps...")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=steps,
        generator=generator
    )

    return result.images[0]


def get_output_path(output_arg):
    """Generate output path."""
    os.makedirs("output", exist_ok=True)

    if output_arg:
        if not output_arg.endswith(".png"):
            output_arg += ".png"
        return os.path.join("output", output_arg)

    # Auto-generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing = len([f for f in os.listdir("output") if f.endswith(".png")])
    return os.path.join("output", f"flux_{existing:04d}_{timestamp}.png")


def main():
    args = parse_args()

    print()
    print("=" * 50)
    print("  FLUX.2 [klein] 4B - Local Image Generation")
    print("=" * 50)
    print()

    # Load pipeline
    pipe = load_pipeline(use_offload=not args.no_offload)

    # Generate base image
    image = generate_image(
        pipe,
        args.prompt,
        args.width,
        args.height,
        args.steps,
        args.seed
    )

    # Upscale if requested
    if args.upscale:
        print()
        print("Loading Real-ESRGAN 4x upscaler...")
        upscaler = load_upscaler()

        print(f"Upscaling {args.width}x{args.height} -> {args.width*4}x{args.height*4}...")
        torch.cuda.empty_cache()
        image = upscaler.upscale(image)
        print("Upscaling complete!")

    # Save
    output_path = get_output_path(args.output)
    image.save(output_path, quality=95)

    print()
    print("=" * 50)
    print(f"  Saved: {output_path}")
    print(f"  Size: {image.size[0]}x{image.size[1]}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
