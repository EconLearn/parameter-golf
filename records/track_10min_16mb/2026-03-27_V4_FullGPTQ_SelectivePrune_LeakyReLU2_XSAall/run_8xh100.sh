#!/bin/bash
# === PARAMETER GOLF V4 SUBMISSION: AUTOMATED 3-SEED RUN ===
# Run this on a RunPod 8xH100 SXM pod with the Parameter Golf template.
# Total time: ~25 minutes. Total cost: ~$9 at $21.52/hr.
# Usage: bash run_8xh100.sh
set -e

echo "=========================================="
echo "PARAMETER GOLF V4 - 3 SEED AUTOMATED RUN"
echo "=========================================="
START=$(date +%s)

# Setup
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone -b v4-submission --depth 1 https://github.com/EconLearn/parameter-golf.git
fi
cd parameter-golf

# Download data (if not already present)
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo ">>> Data already present, skipping download"
fi

# Verify GPU setup
echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

SCRIPT="records/track_10min_16mb/2026-03-27_V4_FullGPTQ_SelectivePrune_LeakyReLU2_XSAall/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-03-27_V4_FullGPTQ_SelectivePrune_LeakyReLU2_XSAall/logs"
mkdir -p "$LOGDIR"

# Run 3 seeds
for SEED in 1337 42 2024; do
    echo ""
    echo "=========================================="
    echo ">>> SEED $SEED - Starting at $(date)"
    echo "=========================================="

    SEED=$SEED torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOGDIR/train_seed${SEED}.log"

    # Copy artifacts
    if [ -f "final_model.int6.ptz" ]; then
        cp final_model.int6.ptz "$LOGDIR/model_seed${SEED}.int6.ptz"
    fi

    echo ">>> SEED $SEED - Finished at $(date)"

    # Extract BPB from log
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1
done

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "ALL 3 SEEDS COMPLETE"
echo "Total time: ${ELAPSED}s ($((ELAPSED/60))m $((ELAPSED%60))s)"
echo "=========================================="
echo ""
echo "=== RESULTS ==="
for SEED in 1337 42 2024; do
    echo -n "Seed $SEED: "
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "NOT FOUND"
done
echo ""
echo "Logs saved to: $LOGDIR/"
echo "TO SUBMIT: create PR from EconLearn/parameter-golf v4-submission branch"
