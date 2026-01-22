"""
Real-ESRGAN 4x Upscaler Module
Uses spandrel for model loading and inference.
"""

import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from spandrel import ModelLoader


class Upscaler:
    """Real-ESRGAN 4x image upscaler."""

    def __init__(self, tile_size=512):
        """
        Initialize the upscaler.

        Args:
            tile_size: Size of tiles for processing (reduces VRAM usage)
        """
        self.tile_size = tile_size
        self.model = None
        self.scale = 4
        self._load_model()

    def _load_model(self):
        """Download and load the Real-ESRGAN model."""
        print("Downloading Real-ESRGAN 4x model...")
        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4.pth"
        )

        print("Loading upscaler...")
        self.model = ModelLoader().load_from_file(model_path)
        self.model.cuda().eval()
        self.scale = self.model.scale
        print(f"Upscaler ready (scale: {self.scale}x)")

    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Upscale an image 4x using tiled inference.

        Args:
            image: PIL Image to upscale

        Returns:
            Upscaled PIL Image
        """
        # Convert PIL to tensor
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

        with torch.no_grad():
            _, _, h, w = img_tensor.shape

            # Tiled inference to save VRAM
            tile = self.tile_size
            overlap = 32

            output = torch.zeros(
                (1, 3, h * self.scale, w * self.scale),
                device='cuda'
            )

            for y in range(0, h, tile - overlap):
                for x in range(0, w, tile - overlap):
                    # Extract tile
                    y_end = min(y + tile, h)
                    x_end = min(x + tile, w)
                    tile_in = img_tensor[:, :, y:y_end, x:x_end]

                    # Upscale tile
                    tile_out = self.model(tile_in)

                    # Place tile
                    y_out = y * self.scale
                    x_out = x * self.scale
                    y_out_end = y_end * self.scale
                    x_out_end = x_end * self.scale

                    output[:, :, y_out:y_out_end, x_out:x_out_end] = tile_out

            output = output.clamp(0, 1)

        # Convert back to PIL
        out_np = (output[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(out_np)


# Test if run directly
if __name__ == "__main__":
    print("Testing upscaler...")

    # Create a small test image
    test_img = Image.new('RGB', (256, 256), color='blue')

    upscaler = Upscaler()
    result = upscaler.upscale(test_img)

    print(f"Input: 256x256")
    print(f"Output: {result.size[0]}x{result.size[1]}")
    print("Upscaler test passed!")
