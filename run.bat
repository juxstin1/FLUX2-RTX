@echo off
title Flux2 Klein Local
color 0B

echo.
echo  FLUX.2 Klein Local
echo  ==================
echo.

:: Activate environment
call venv\Scripts\activate.bat

:: Check if first argument provided
if "%~1"=="" (
    echo Usage: run.bat "your prompt here" [options]
    echo.
    echo Examples:
    echo   run.bat "a sunset over mountains"
    echo   run.bat "cyberpunk city" --steps 28
    echo   run.bat "fantasy castle" --steps 28 --upscale
    echo.
    echo Options:
    echo   --steps N     Inference steps (4=fast, 28=quality)
    echo   --upscale     Enable 4x upscaling (1024 to 4096)
    echo   --seed N      Set random seed
    echo   --width N     Image width (default: 1024)
    echo   --height N    Image height (default: 1024)
    echo.
    pause
    exit /b 0
)

:: Run generation with all arguments
python generate.py %*

echo.
pause
