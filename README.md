# Flux2 Klein Local

Run **FLUX.2 [klein] 4B** locally on your RTX GPU with optional **4x Real-ESRGAN upscaling**.

Generate stunning AI images in seconds, upscale to 4096x4096 for print-quality output.

![Flux2 Klein](https://img.shields.io/badge/FLUX.2-Klein%204B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![CUDA](https://img.shields.io/badge/CUDA-12.x-brightgreen)

## Sample Outputs

| Cyberpunk Samurai | Floating Library | Bioluminescent Jellyfish |
|-------------------|------------------|--------------------------|
| Steampunk Airship | Kitsune Spirit   | *Your creation here*     |

---

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090, 4070, 4080, 4090 or better (minimum 12GB VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: ~15GB for models (downloaded automatically)

### Software
- **Windows 10/11** (64-bit)
- **Python 3.10 or 3.11** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **NVIDIA Driver 525+** ([Download](https://www.nvidia.com/Download/index.aspx))

---

## Quick Install

### Option 1: One-Click Install (Recommended)

1. Clone this repo:
   ```bash
   git clone https://github.com/juxstin1/FLUX2-RTX.git
   cd flux2-klein-local
   ```

2. Run the installer:
   ```bash
   install.bat
   ```

3. Wait for installation to complete (~5-10 minutes depending on internet speed)

4. Models will download automatically on first run (~8GB)

### Option 2: Manual Install

```bash
# Clone repo
git clone https://github.com/juxstin1/FLUX2-RTX.git
cd flux2-klein-local

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install diffusers from source (required for Flux2Klein)
pip install git+https://github.com/huggingface/diffusers.git
```

---

## Usage

### Activate Environment First
```bash
venv\Scripts\activate
```

### Quick Generation (Fast Mode - 4 steps)
```bash
python generate.py "a cyberpunk city at night"
```

### High Quality Generation (28 steps)
```bash
python generate.py "a cyberpunk city at night" --steps 28
```

### With 4x Upscaling (1024 -> 4096)
```bash
python generate.py "a cyberpunk city at night" --steps 28 --upscale
```

### All Options
```bash
python generate.py "your prompt here" [OPTIONS]

Options:
  --width       Image width (default: 1024)
  --height      Image height (default: 1024)
  --steps       Inference steps, 4=fast, 28=quality (default: 4)
  --seed        Random seed for reproducibility (default: random)
  --upscale     Enable 4x Real-ESRGAN upscaling
  --output      Output filename (default: auto-generated)
```

### Examples
```bash
# Fast preview
python generate.py "sunset over mountains"

# High quality
python generate.py "portrait of a warrior, dramatic lighting" --steps 28

# Print quality (4096x4096)
python generate.py "fantasy castle in clouds" --steps 28 --upscale

# Specific size
python generate.py "wide landscape" --width 1920 --height 1080 --steps 28

# Reproducible result
python generate.py "cute robot" --seed 42
```

### Batch Generation
```bash
python generate_batch.py
```
Edit `prompts.txt` to customize batch prompts.

---

## Model Information

### FLUX.2 [klein] 4B
- **Parameters**: 4 billion
- **License**: Apache 2.0 (fully open source, commercial use allowed)
- **VRAM**: ~13GB with CPU offloading
- **Speed**: ~1-2 seconds (4 steps), ~10 seconds (28 steps)
- **Source**: [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

### Real-ESRGAN 4x (Upscaler)
- **Scale**: 4x (1024 -> 4096)
- **License**: BSD-3-Clause
- **Source**: [GitHub](https://github.com/xinntao/Real-ESRGAN)

Models are downloaded automatically to:
- Windows: `C:\Users\YOUR_NAME\.cache\huggingface\`

---

## Troubleshooting

### "CUDA not available"
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

### "Out of memory"
- Close other GPU applications
- The script uses CPU offloading by default
- Try smaller resolution: `--width 768 --height 768`

### "Module not found"
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Slow download
Models are ~8GB total. First run takes time. Subsequent runs use cached models.

---

## File Structure

```
flux2-klein-local/
├── generate.py          # Main generation script
├── generate_batch.py    # Batch generation script
├── upscaler.py          # Real-ESRGAN upscaling module
├── prompts.txt          # Batch prompts (edit this)
├── requirements.txt     # Python dependencies
├── install.bat          # One-click installer
├── run.bat              # Quick launcher
├── output/              # Generated images (created automatically)
└── README.md            # This file
```

---

## Credits

- **FLUX.2 [klein]** by [Black Forest Labs](https://blackforestlabs.ai/)
- **Real-ESRGAN** by [xinntao](https://github.com/xinntao/Real-ESRGAN)
- **Diffusers** by [Hugging Face](https://github.com/huggingface/diffusers)

---

## License

This project is for personal and educational use.

- FLUX.2 [klein] 4B: Apache 2.0
- Real-ESRGAN: BSD-3-Clause
- This repo: MIT

---

## Links

- [FLUX.2 Klein on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Black Forest Labs Blog](https://blackforestlabs.ai/announcing-flux-2-klein/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
