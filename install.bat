@echo off
title Flux2 Klein Local - Installer
color 0A

echo.
echo  ============================================
echo   FLUX.2 Klein Local - One-Click Installer
echo  ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python found
python --version

:: Check for Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Git not found - some features may not work
) else (
    echo [OK] Git found
)

:: Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] NVIDIA GPU not detected!
    echo Please ensure you have:
    echo   1. An NVIDIA RTX GPU (3090, 4070, 4080, 4090, etc.)
    echo   2. Latest NVIDIA drivers installed
    echo.
    pause
    exit /b 1
)
echo [OK] NVIDIA GPU detected
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo.
echo ============================================
echo  Step 1: Creating virtual environment...
echo ============================================

if exist venv (
    echo [INFO] Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo.
echo ============================================
echo  Step 2: Activating environment...
echo ============================================

call venv\Scripts\activate.bat
echo [OK] Environment activated

echo.
echo ============================================
echo  Step 3: Upgrading pip...
echo ============================================

python -m pip install --upgrade pip
echo [OK] Pip upgraded

echo.
echo ============================================
echo  Step 4: Installing PyTorch with CUDA 12.4...
echo ============================================
echo This may take a few minutes...

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)
echo [OK] PyTorch installed with CUDA support

echo.
echo ============================================
echo  Step 5: Installing dependencies...
echo ============================================

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.
echo ============================================
echo  Step 6: Installing Diffusers (from source)...
echo ============================================
echo This is required for Flux2Klein pipeline...

pip install git+https://github.com/huggingface/diffusers.git
if errorlevel 1 (
    echo [ERROR] Failed to install Diffusers
    pause
    exit /b 1
)
echo [OK] Diffusers installed

echo.
echo ============================================
echo  Step 7: Verifying CUDA...
echo ============================================

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo [WARNING] Could not verify CUDA - please check manually
)

echo.
echo ============================================
echo  Creating output directory...
echo ============================================

if not exist output mkdir output
echo [OK] Output directory ready

echo.
echo ============================================
echo.
echo  [SUCCESS] Installation Complete!
echo.
echo ============================================
echo.
echo  Next steps:
echo.
echo  1. Run: run.bat
echo     OR
echo  2. Activate environment: venv\Scripts\activate
echo     Then run: python generate.py "your prompt"
echo.
echo  NOTE: First run will download models (~8GB)
echo        This only happens once!
echo.
echo ============================================
echo.

pause
