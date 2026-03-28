#!/bin/bash
# === PARAMETER GOLF V5: AUTOMATED 3-SEED RUN ===
# Run on RunPod 8xH100 SXM pod. Total: ~60 min (training + TTT eval). Cost: ~$14 at $14/hr spot.
# Usage: bash run_8xh100.sh
set -e
echo "=========================================="
echo "PARAMETER GOLF V5 - 3 SEED AUTOMATED RUN"
echo "=========================================="
START=$(date +%s)
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone -b v4-submission --depth 1 https://github.com/EconLearn/parameter-golf.git
fi
cd parameter-golf && git pull
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo ">>> Data already present"
fi
echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SCRIPT="records/track_10min_16mb/2026-03-27_V5_LeakyReLU2_GPTQ_TTT/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-03-27_V5_LeakyReLU2_GPTQ_TTT/logs"
mkdir -p "$LOGDIR"
for SEED in 1337 42 2024; do
    echo ""
    echo "=========================================="
    echo ">>> SEED $SEED - Starting at $(date)"
    echo "=========================================="
    SEED=$SEED torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOGDIR/train_seed${SEED}.log"
    [ -f "final_model.int6.ptz" ] && cp final_model.int6.ptz "$LOGDIR/model_seed${SEED}.int6.ptz"
    echo ">>> SEED $SEED - Finished at $(date)"
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1
done
END=$(date +%s); ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "ALL 3 SEEDS COMPLETE - ${ELAPSED}s ($((ELAPSED/60))m)"
echo "=== RESULTS ==="
for SEED in 1337 42 2024; do
    echo -n "Seed $SEED: "
    grep "final_int8_zlib_roundtrip_exact" "$LOGDIR/train_seed${SEED}.log" | tail -1 || echo "NOT FOUND"
done
echo "Logs: $LOGDIR/"
