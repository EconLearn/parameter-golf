@echo off
echo ============================================
echo  Parameter Golf V4 - Windows RTX 4060 Setup
echo ============================================
echo.

:: Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Check CUDA
nvidia-smi 2>nul
if errorlevel 1 (
    echo ERROR: nvidia-smi not found. Install NVIDIA drivers.
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo.
echo [2/5] Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

echo.
echo [3/5] Installing dependencies...
pip install sentencepiece zstandard numpy huggingface-hub datasets tqdm -q

echo.
echo [4/5] Downloading dataset (1 training shard + full validation)...
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

echo.
echo [5/5] Verifying GPU...
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB'); print(f'CUDA: {torch.version.cuda}')"

echo.
echo ============================================
echo  Setup complete! Now run:
echo    run_v4.bat
echo ============================================
pause
