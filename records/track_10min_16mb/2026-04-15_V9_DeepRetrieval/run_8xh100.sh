#!/bin/bash
# === PARAMETER GOLF V9: Deep Retrieval ===
#
# V9 targets ~1.08 BPB by combining:
#   1) the PROVEN 11-layer dense architecture from 1.1228 SOTA (stable, fast)
#   2) the SP8192 tokenizer used by every top-5 recent leaderboard entry
#   3) bigbag 1.0810's exact hyperparameter block (SDClip=12.85, QK=5.25,
#      Muon mom=0.97 lr=0.03 WD=0.095, EMA=0.9965, warmdown=5000, late QAT=0.25)
#   4) a stronger compression pipeline: full-Hessian GPTQ + selective pruning +
#      auto-pick smallest of {Huffman, zstd-22, Brotli-11}
#   5) eval-time boosts: sliding window + OGD vocab bias + n-gram oracle mixing
#
# V7 FAILURE POST-MORTEM (why we're not using it):
#   - flash_attn_3 not installed in the previous pod → SDPA fallback = 3-5x slower
#   - 7 base layers × 2 loops + LoRA + self-distillation = ~7x V7's step time
#   - result: 1017 steps in 600s, val_bpb 1.5307 — far worse than SOTA
# V9 fixes #1 by installing flash_attn_3 explicitly, and avoids #2 by using
# the 11-layer dense architecture proven at 1.1228 BPB.
#
# Usage:
#   bash run_8xh100.sh           # Single seed (validation, ~15-20 min, ~$5)
#   bash run_8xh100.sh all       # All 3 seeds (full submission, ~60 min, ~$15)
set -e
MODE="${1:-single}"

echo "=========================================="
echo "PARAMETER GOLF V9 - DEEP RETRIEVAL"
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
echo ">>> Installing dependencies..."
pip install sentencepiece huggingface_hub zstandard brotli 2>&1 | tail -5

# --- Install flash_attn_3 (CRITICAL) ---
# The V7 run crashed to 590ms/step because flash_attn_3 was missing and the
# fallback uses F.scaled_dot_product_attention (3-5x slower on H100s).
# On an H100 pod with the standard PyTorch 2.4 wheel, this builds in ~3 min.
if ! python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo ">>> Installing flash_attn_3 (needed for fast attention on H100)..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -5 || echo ">>> flash-attn pip install failed, will use SDPA fallback"
fi

# Verify critical imports BEFORE spending time/money on data download
python3 -c "
import sentencepiece, huggingface_hub, torch
print(f'>>> Dependencies OK: torch={torch.__version__}')
print(f'>>> CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
try:
    import zstandard; print('>>> zstandard available')
except ImportError: print('>>> WARNING: zstandard missing — will fall back to zlib')
try:
    import brotli; print('>>> brotli available')
except ImportError: print('>>> WARNING: brotli missing — Huffman/zstd only')
try:
    from flash_attn_interface import flash_attn_func; print('>>> flash_attn_3 available (fast path)')
except ImportError: print('>>> WARNING: flash_attn_3 missing — SDPA fallback (expect 3-5x slower steps)')
"

# --- Download SP8192 data (community fork: kevclark/parameter-golf) ---
# Every recent top submission (incl. the 1.081 frontier) uses SP8192.
# The official openai/parameter-golf repo only ships SP1024, so we pull SP8192
# from kevclark's mirror which has both variants.
if [ ! -f "data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading SP8192 data from kevclark/parameter-golf..."
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
else
    echo ">>> SP8192 data already present"
fi

# Verify data + tokenizer exist before burning compute
[ -f "data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin" ] || { echo "ERROR: SP8192 validation data not found"; exit 1; }
[ -f "data/tokenizers/fineweb_8192_bpe.model" ] || { echo "ERROR: SP8192 tokenizer not found"; exit 1; }
echo ">>> Data verified OK (variant: sp8192)"

echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

SCRIPT="records/track_10min_16mb/2026-04-15_V9_DeepRetrieval/train_gpt.py"
LOGDIR="records/track_10min_16mb/2026-04-15_V9_DeepRetrieval/logs"
mkdir -p "$LOGDIR"

# --- V9 Config: 11L DENSE + SP8192 + bigbag 1.0810 hyperparams ---
# These are the exposed knobs; defaults inside train_gpt.py already match this,
# but setting them explicitly here makes the run reproducible if train_gpt.py
# defaults drift later.
# Tokenizer & data
export VOCAB_SIZE=8192
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model

# Architecture: 11 DENSE layers (NOT recursive — that's what killed V7)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0
export XSA_LAST_N=4
export VE_LAYERS="9,10"

# bigbag 1.0810 hyperparameters
export QK_GAIN_INIT=5.25
export MATRIX_LR=0.03
export MUON_MOMENTUM=0.97
export MUON_WD=0.095
export ADAM_WD=0.095
export EMA_DECAY=0.9965
export WARMDOWN_ITERS=5000
export LATE_QAT_THRESHOLD=0.25
export BIGRAM_VOCAB_SIZE=4096

# V9 compression pipeline
export SDCLIP_K_INT6=12.85
export SDCLIP_K_INT8=20.0
export GPTQ_ENABLED=1
export SELECTIVE_PRUNE_ENABLED=1
export ARTIFACT_BUDGET_BYTES=16000000

# V9 eval-time boosts (no extra training cost; they run in the eval budget)
export OGD_BIAS_ENABLED=1
export OGD_BIAS_LR=0.1
export NGRAM_EVAL_ENABLED=1
export NGRAM_EVAL_WEIGHT=0.15

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
    [ -f "final_model.huff"     ] && cp final_model.huff     "$LOGDIR/model_seed${SEED}.huff"
    [ -f "final_model.int6.ptz" ] && cp final_model.int6.ptz "$LOGDIR/model_seed${SEED}.int6.ptz"
    [ -f "final_model.brotli"   ] && cp final_model.brotli   "$LOGDIR/model_seed${SEED}.brotli"
    echo ">>> SEED $SEED - Finished at $(date)"
    grep "final_best_eval_exact\|final_int6_sliding_window_exact\|final_ngram_oracle_exact\|final_ogd_exact\|WINNER\|Final submission size" "$LOGDIR/train_seed${SEED}.log" | tail -8 || echo "Score not found"
done

END=$(date +%s); ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "RUN COMPLETE - ${ELAPSED}s ($((ELAPSED/60))m)"
echo "=== RESULTS ==="
for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"
    grep "final_best_eval_exact\|final_int6_sliding_window_exact\|final_int6_roundtrip_exact\|final_ogd_exact\|final_ngram_oracle_exact\|WINNER\|Final submission size\|step_avg" "$LOGDIR/train_seed${SEED}.log" | tail -10
done
echo ""
echo "Logs: $LOGDIR/"
