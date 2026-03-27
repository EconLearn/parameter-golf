@echo off
echo ============================================
echo  Parameter Golf V4 - FULL Training Run
echo  RTX 4060 (8GB) - ~60 min estimated
echo ============================================
echo.

call .venv\Scripts\activate.bat

:: Full 5-minute wallclock training (matching competition but single GPU)
:: Then full quantization + sliding window eval

set ITERATIONS=20000
set TRAIN_BATCH_TOKENS=8192
set VAL_BATCH_SIZE=8192
set MAX_WALLCLOCK_SECONDS=300
set WARMDOWN_ITERS=3500
set VAL_LOSS_EVERY=2000
set TRAIN_LOG_EVERY=100
set SEED=1337
set EVAL_STRIDE=64
set WARMUP_STEPS=20
set TRAIN_SEQ_LEN=1024
set EVAL_SEQ_LEN=1024
set PYTHONUNBUFFERED=1

echo Config: Full run, 300s wallclock, seq_len=1024, batch=8192
echo Estimated time: ~60 min (training + eval + quantization)
echo.

python -u records/track_10min_16mb/2026-03-27_V4_FullGPTQ_SelectivePrune_LeakyReLU2_XSAall/train_gpt.py

echo.
echo ============================================
echo  DONE! Look for final_int8_zlib_roundtrip_exact
echo  in the output above for your BPB score.
echo ============================================
pause
