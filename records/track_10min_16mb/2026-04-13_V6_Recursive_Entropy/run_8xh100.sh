#!/bin/bash
# === PARAMETER GOLF V6: RunPod 8xH100 SXM ===
# Usage:
#   bash run_8xh100.sh           # Single seed 1337 (validation run, ~15 min, ~$4)
#   bash run_8xh100.sh all       # All 3 seeds (full submission, ~45 min, ~$12)
set -e
MODE="${1:-single}"

echo "=========================================="
echo "PARAMETER GOLF V6 - RECURSIVE DEPTH + HUFFMAN"
echo "Mode: $MODE"
echo "=========================================="
START=$(date +%s)

cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone -b v4-submission --depth 1 https://github.com/EconLearn/parameter-golf.git
fi
cd parameter-golf && git pull
pip install zstandard

if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo ">>> Data already present"
fi

echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SCRIPT="records/track_10min_16mb/2026-04-13_V6_Recursive_Entropy/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-04-13_V6_Recursive_Entropy/logs"
mkdir -p "$LOGDIR"

# V6: 5 base layers x 2 loops = 10 effective, dim=640
export NUM_BASE_LAYERS=5
export LOOP_COUNT=2
export LORA_RANK=8
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export XSA_LAST_N=4
export OGD_BIAS_ENABLED=1
export ENTROPY_REG_LAMBDA=0.01

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
done

END=$(date +%s); ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "RUN COMPLETE - ${ELAPSED}s ($((ELAPSED/60))m)"
echo "=== RESULTS ==="
for SEED in $SEEDS; do
    echo -n "Seed $SEED: "
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "NOT FOUND"
done
echo ""
echo "=== ARTIFACT SIZE ==="
for SEED in $SEEDS; do
    for ext in huff int6.ptz; do
        F="$LOGDIR/model_seed${SEED}.${ext}"
        [ -f "$F" ] && echo "Seed $SEED ($ext): $(stat -c%s "$F" 2>/dev/null || stat -f%z "$F") bytes"
    done
done
echo ""
echo "Logs: $LOGDIR/"
echo "To stop pod: runpodctl stop pod \$RUNPOD_POD_ID"
