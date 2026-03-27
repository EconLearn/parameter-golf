@echo off
echo ============================================
echo  Parameter Golf V4 - Training on RTX 4060
echo ============================================
echo.

call .venv\Scripts\activate.bat

:: RTX 4060 has 8GB VRAM - use small batch size
:: 100 iterations with small batch = quick smoke test (~10-15 min)
:: Increase ITERATIONS for longer training

set ITERATIONS=100
set TRAIN_BATCH_TOKENS=8192
set VAL_BATCH_SIZE=8192
set MAX_WALLCLOCK_SECONDS=600
set WARMDOWN_ITERS=30
set VAL_LOSS_EVERY=50
set TRAIN_LOG_EVERY=10
set SEED=1337
set EVAL_STRIDE=512
set WARMUP_STEPS=5
set TRAIN_SEQ_LEN=1024
set EVAL_SEQ_LEN=1024
set PYTHONUNBUFFERED=1

echo Config: 100 iters, batch=8192 tokens, seq_len=1024 (8GB VRAM safe)
echo.

python -u records/track_10min_16mb/2026-03-27_V4_FullGPTQ_SelectivePrune_LeakyReLU2_XSAall/train_gpt.py

echo.
echo ============================================
echo  Training complete! Check logs/ folder.
echo ============================================
pause
