#!/bin/bash
# === PARAMETER GOLF V7: Complementary Neural-Ngram System ===
# Usage:
#   bash run_8xh100.sh           # Single seed (validation, ~20 min, ~$5)
#   bash run_8xh100.sh all       # All 3 seeds (full submission, ~60 min, ~$15)
set -e
MODE="${1:-single}"

echo "=========================================="
echo "PARAMETER GOLF V7 - NGRAM COMPLEMENT"
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
# Only install what this script actually needs (targeted, not the full requirements.txt
# which includes packages like 'kernels' that may fail to compile).
echo ">>> Installing dependencies..."
pip install sentencepiece huggingface_hub zstandard 2>&1 | tail -5
# Verify critical imports before spending time on data download
python3 -c "import sentencepiece; import huggingface_hub; import torch; print('>>> Dependencies OK: sentencepiece, huggingface_hub, torch=' + torch.__version__)"

# --- Download data ---
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading SP1024 data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo ">>> SP1024 data already present"
fi

# Verify data and tokenizer exist
[ -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ] || { echo "ERROR: Validation data not found"; exit 1; }
[ -f "data/tokenizers/fineweb_1024_bpe.model" ] || { echo "ERROR: Tokenizer not found"; exit 1; }

echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

SCRIPT="records/track_10min_16mb/2026-04-14_V7_NgramComplement/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-04-14_V7_NgramComplement/logs"
mkdir -p "$LOGDIR"

# --- V7 config: SP1024, dim=640, 5x2 recursive, n-gram complement ---
export VOCAB_SIZE=1024
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export NUM_BASE_LAYERS=5
export LOOP_COUNT=2
export LORA_RANK=8
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export XSA_LAST_N=4
export OGD_BIAS_ENABLED=1
export ENTROPY_REG_LAMBDA=0.03
export NGRAM_COMPLEMENT=1
export NGRAM_DISCOUNT=0.5
export NGRAM_EVAL_ENABLED=1
export NGRAM_EVAL_WEIGHT=0.3
export SELF_DISTILL_WEIGHT=0.1

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
    echo ">>> SEED $SEED - Finished at $(date)"
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "Score not found"
    grep "final_ngram_oracle_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "N-gram score not found"
done

END=$(date +%s); ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "RUN COMPLETE - ${ELAPSED}s ($((ELAPSED/60))m)"
echo "=== RESULTS ==="
for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"
    grep "final_int6_sliding_window_exact\|final_ogd\|final_ngram_oracle_exact\|final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -4
done
echo ""
echo "Logs: $LOGDIR/"
