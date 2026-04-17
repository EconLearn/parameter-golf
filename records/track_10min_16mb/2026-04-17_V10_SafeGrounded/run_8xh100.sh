#!/bin/bash
# === PARAMETER GOLF V10: Safe Grounded ===
#
# Design: start from the PROVEN 1.1228 BPB SOTA byte-for-byte. Add only
# eval-time post-training features that can reduce the reported BPB but
# cannot corrupt training. No SP8192, no hparam changes, no new optimizers,
# no architectural edits.
#
# The two additions:
#   1) OGD bias eval  — per-vocab bias adapted online from already-scored tokens
#   2) N-gram oracle  — bigram+trigram table mixed with neural via entropy gating
#
# Training is IDENTICAL to the 1.1228 record. V10 will reproduce ~1.1228 on
# the sliding_window eval (same pipeline, same hparams, same data), then
# layer the two new evals on top. Expected final BPB: ~1.115-1.119.
#
# Usage:
#   bash run_8xh100.sh           # Single seed validation (~15 min, ~$4)
#   bash run_8xh100.sh all       # All 3 seeds (~50 min, ~$12)
set -e
MODE="${1:-single}"
echo "=========================================="
echo "PARAMETER GOLF V10 - SAFE GROUNDED"
echo "Mode: $MODE"
echo "=========================================="
START=$(date +%s)
# --- Setup ---
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone -b v4-submission --depth 1 https://github.com/EconLearn/parameter-golf.git
fi
cd parameter-golf && git pull
# --- Install dependencies ---
# PyTorch and CUDA are pre-installed on RunPod PyTorch templates. We just need
# sentencepiece + huggingface_hub for data/tokenizer loading, and zstandard for
# compression (already used by SOTA; fallback to zlib if absent).
echo ">>> Installing dependencies..."
pip install sentencepiece huggingface_hub zstandard 2>&1 | tail -3
# --- Install flash_attn v3 (hopper kernels, critical for H100 perf) ---
# Without this, the fallback SDPA path adds ~70ms/step overhead. That lost V9
# almost half its training steps and broke the reported BPB. V10 has a CLEAN
# SDPA fallback (uses enable_gqa=True), so even if the v3 build fails the run
# still finishes — just at ~100ms/step instead of ~86ms.
if ! python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo ">>> Building flash_attn v3 hopper kernels (5-8 min — abort-safe)..."
    pip install ninja packaging 2>&1 | tail -3
    TMP=$(mktemp -d)
    (
      cd "$TMP" \
      && git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git \
      && cd flash-attention/hopper \
      && MAX_JOBS=4 python3 setup.py install
    ) 2>&1 | tail -10 || echo ">>> flash_attn v3 build failed — falling back to SDPA (expect ~100ms/step)"
fi
# Import sanity check BEFORE spending on data
python3 -c "
import sentencepiece, huggingface_hub, torch
print(f'>>> torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
try: import zstandard; print('>>> zstandard OK')
except ImportError: print('>>> WARN: zstandard missing — zlib fallback')
try:
    from flash_attn_interface import flash_attn_func
    print('>>> flash_attn v3 available — FAST PATH')
except ImportError:
    print('>>> flash_attn v3 NOT available — SDPA fallback (runs but ~15% slower)')
"
# --- Download SP1024 data (the ONLY data the 1.1228 SOTA was ever trained on) ---
# SP1024 is the official data shipped with openai/parameter-golf. No community
# mirror needed. This was the critical mistake in V9 — trying SP8192 without a
# proven training recipe for it.
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading SP1024 data (official openai/parameter-golf)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo ">>> SP1024 data already present"
fi
# Verify before burning compute
[ -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ] || { echo "ERROR: SP1024 val data missing"; exit 1; }
[ -f "data/tokenizers/fineweb_1024_bpe.model" ] || { echo "ERROR: SP1024 tokenizer missing"; exit 1; }
echo ">>> Data verified OK"
echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
SCRIPT="records/track_10min_16mb/2026-04-17_V10_SafeGrounded/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-04-17_V10_SafeGrounded/logs"
mkdir -p "$LOGDIR"
# --- V10 Config: EXACTLY the 1.1228 SOTA hyperparameters + two eval-time knobs ---
# All 1.1228 SOTA knobs are the defaults inside train_gpt.py. We do NOT re-export
# them here — any explicit export is a source of drift. Only the V10 additions
# are exposed below for reproducibility.
#
# (tokenizer/data default to SP1024 paths inside train_gpt.py; no override)
# (all training hparams are the 1.1228 defaults inside train_gpt.py; no override)
# V10 eval-time additions
export OGD_BIAS_ENABLED=1
export OGD_BIAS_LR=0.1
export NGRAM_EVAL_ENABLED=1
export NGRAM_EVAL_WEIGHT=0.10
if [ "$MODE" = "all" ]; then
    SEEDS="1337 42 2024"
else
    SEEDS="1337"
fi
for SEED in $SEEDS; do
    echo ""
    echo "=========================================="
    echo ">>> SEED $SEED - Starting at $(date)"
    echo "=========================================="
    SEED=$SEED torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOGDIR/train_seed${SEED}.log"
    [ -f "final_model.int6.ptz" ] && cp final_model.int6.ptz "$LOGDIR/model_seed${SEED}.int6.ptz"
    echo ">>> SEED $SEED - Finished at $(date)"
    echo "--- Best eval scores for seed $SEED ---"
    grep -E "final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int6_sliding_window_s64_exact|final_ogd_exact|final_ngram_oracle_exact|final_best_eval_exact|step_avg" "$LOGDIR/train_seed${SEED}.log" | tail -15
done
END=$(date +%s); ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "RUN COMPLETE - ${ELAPSED}s ($((ELAPSED/60))m)"
echo "=== SUMMARY ==="
for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"
    grep -E "final_best_eval_exact|final_int6_sliding_window_exact" "$LOGDIR/train_seed${SEED}.log" | tail -2
done
echo ""
echo "Logs: $LOGDIR/"
