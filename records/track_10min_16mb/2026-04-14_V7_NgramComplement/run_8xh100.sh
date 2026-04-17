#!/bin/bash
# === PARAMETER GOLF V8: SOTA Attempt ===
# Combines proven SOTA techniques with novel N-gram complement system.
# Usage:
#   bash run_8xh100.sh           # Single seed (validation, ~20 min, ~$5)
#   bash run_8xh100.sh all       # All 3 seeds (full submission, ~60 min, ~$15)
set -e
MODE="${1:-single}"

echo "=========================================="
echo "PARAMETER GOLF V8 - SOTA ATTEMPT"
echo "Mode: $MODE"
echo "=========================================="
START=$(date +%s)

# --- Setup ---
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone -b v4-submission --depth 1 https://github.com/EconLearn/parameter-golf.git
fi
cd parameter-golf && git pull

# --- Install ALL dependencies ---
echo ">>> Installing dependencies..."
pip install sentencepiece huggingface_hub zstandard brotli 2>&1 | tail -5
# Verify critical imports BEFORE spending time/money on data download
python3 -c "
import sentencepiece
import huggingface_hub
import torch
import brotli
print(f'>>> Dependencies OK: sentencepiece, huggingface_hub, brotli, torch={torch.__version__}')
print(f'>>> CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
"

# --- Download SP8192 data from community repo ---
# SP8192 is hosted on kevclark/parameter-golf (community), not the official repo.
# All top submissions (1.08 BPB and below) use SP8192.
# If SP8192 fails, automatically fall back to SP1024 with adjusted architecture.
SP_VARIANT="sp8192"
if [ ! -f "data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading SP8192 data from kevclark/parameter-golf..."
    if MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192; then
        echo ">>> SP8192 download successful"
    else
        echo ">>> WARNING: SP8192 download failed. Falling back to SP1024."
        SP_VARIANT="sp1024"
        python3 data/cached_challenge_fineweb.py --variant sp1024
        # Override architecture for SP1024 (wider model, smaller embedding)
        export VOCAB_SIZE=1024
        export DATA_PATH=./data/datasets/fineweb10B_sp1024
        export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
        export MODEL_DIM=640
        export NUM_HEADS=10
        export NUM_KV_HEADS=5
    fi
else
    echo ">>> SP8192 data already present"
fi

# Verify data and tokenizer exist before proceeding
if [ "$SP_VARIANT" = "sp8192" ]; then
    [ -f "data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin" ] || { echo "ERROR: SP8192 validation data not found"; exit 1; }
    [ -f "data/tokenizers/fineweb_8192_bpe.model" ] || { echo "ERROR: SP8192 tokenizer not found"; exit 1; }
else
    [ -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ] || { echo "ERROR: SP1024 validation data not found"; exit 1; }
    [ -f "data/tokenizers/fineweb_1024_bpe.model" ] || { echo "ERROR: SP1024 tokenizer not found"; exit 1; }
fi
echo ">>> Data verified OK (variant: $SP_VARIANT)"

echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

SCRIPT="records/track_10min_16mb/2026-04-14_V7_NgramComplement/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-04-14_V7_NgramComplement/logs"
mkdir -p "$LOGDIR"

# --- V8 Config: SP8192, dim=512, 7x2 recursive, SOTA hyperparams + novel N-gram ---
# Tokenizer & data
export VOCAB_SIZE=8192
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model

# Architecture: 7 base layers x 2 loops = 14 virtual layers with LoRA
export NUM_BASE_LAYERS=7
export LOOP_COUNT=2
export LORA_RANK=8
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export XSA_LAST_N=4
export VE_LAYERS="12,13"

# SOTA hyperparameters (matched to bigbag's 1.0810 submission)
export QK_GAIN_INIT=5.25
export MUON_MOMENTUM=0.97
export MATRIX_LR=0.03
export MUON_WD=0.095
export ADAM_WD=0.095
export EMA_DECAY=0.9965
export WARMDOWN_ITERS=5000

# Quantization: SDClip (k-based, better compression than percentile)
export SDCLIP_K_INT6=12.85
export SDCLIP_K_INT8=20.0

# Entropy regularization + late QAT
export ENTROPY_REG_LAMBDA=0.03
export LATE_QAT_THRESHOLD=0.25

# Novel techniques (our unique contribution)
export NGRAM_COMPLEMENT=1
export NGRAM_DISCOUNT=0.5
export NGRAM_EVAL_ENABLED=1
export NGRAM_EVAL_WEIGHT=0.3
export SELF_DISTILL_WEIGHT=0.1

# OGD bias at eval time (with Nacrith-style beta decay + recency weighting)
export OGD_BIAS_ENABLED=1
export OGD_BIAS_PER_TOKEN=1
export OGD_BIAS_BETA=0.995

# Legal Score-First TTT with entropy gating (V9 novel technique)
# SGD on matrix params only where neural entropy is in top 50%
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_MOMENTUM=0.9
export TTT_EPOCHS=3
export TTT_ENTROPY_GATE=1
export TTT_ENTROPY_QUANTILE=0.5

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
    [ -f "final_model.huff" ] && cp final_model.huff "$LOGDIR/model_seed${SEED}.huff"
    [ -f "final_model.int6.ptz" ] && cp final_model.int6.ptz "$LOGDIR/model_seed${SEED}.int6.ptz"
    [ -f "final_model.brotli" ] && cp final_model.brotli "$LOGDIR/model_seed${SEED}.brotli"
    echo ">>> SEED $SEED - Finished at $(date)"
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "Score not found"
    grep "final_ngram_oracle_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "N-gram score not found"
    grep "final_ttt_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "TTT score not found"
done

END=$(date +%s); ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "RUN COMPLETE - ${ELAPSED}s ($((ELAPSED/60))m)"
echo "=== RESULTS ==="
for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"
    grep "final_int6_sliding_window_exact\|final_ogd\|final_ngram_oracle_exact\|final_ttt_exact\|final_int8_zlib_roundtrip_exact\|WINNER\|Final submission size" "$LOGDIR/train_seed${SEED}.log" | tail -8
done
echo ""
echo "Logs: $LOGDIR/"
