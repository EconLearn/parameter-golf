# V9: Deep Retrieval — SP8192 + Proven Arch + Full Compression

**Goal:** close the gap to the current leaderboard frontier (~1.08 BPB) without
repeating V7's architectural mistakes.

## Why V9 exists (V7 post-mortem)

V7 attempted a recursive layer setup (7 base blocks × 2 loops with LoRA) plus
self-distillation and entropy regularization on top of SP8192. On the actual
RunPod pod it produced **1.5307 BPB in 1,017 training steps** — roughly 0.4
BPB worse than the 2-month-old SOTA baseline.

The crash was not conceptual but environmental:

| Issue | Cost |
|---|---|
| `flash_attn_3` not installed → SDPA fallback | ~3-5× slower attention |
| 14 "virtual" layers with LoRA deltas per pass | ~2× FLOPs per step |
| Self-distillation (two full forward passes) | ~2× more FLOPs |
| Entropy regularization with soft-histogram | extra 50M ops per step |
| Per-step n-gram loss weighting (GPU tensor lookup) | small, but not free |

Combined step time: ~590ms vs SOTA's ~86ms. In a 600s wall-clock cap, you simply
can't train a good model that slowly.

## V9 design principles

1. **Start from what works.** Copy the proven 1.1228 11-layer dense architecture
   wholesale. No recursive loops, no LoRA, no self-distillation.
2. **Adopt the one unambiguous win.** Every top submission on the current board
   uses SP8192 — swap the tokenizer and the accompanying hyperparameters.
3. **Re-use V7's non-training contributions.** GPTQ/Huffman/OGD/n-gram oracle
   all run *after* training or *during eval*, so they can't slow step time.
4. **Be paranoid about the pod.** Install `flash_attn_3`, `zstandard`, `brotli`
   before any data download; abort on missing tokenizers.

## What's inside

### Architecture (exactly matches 1.1228 baseline)
- 11 dense transformer blocks, `dim=512`, 8 heads, 4 KV heads (GQA 2:1)
- 3× MLP (1536 hidden), ReLU² activation
- U-Net skips (5 encoder, 6 decoder)
- XSA on last 4 layers (GQA-aware)
- Partial RoPE 16/64, LN scale 1/√(i+1)
- SmearGate, BigramHash(4096, dim=128), ValueEmbedding on layers 9-10
- Tied embeddings, logit softcap=30

### Training (bigbag 1.0810 hyperparameters)
| Knob | Value | vs 1.1228 SOTA |
|---|---|---|
| Tokenizer | SP8192 | SP1024 |
| `QK_GAIN_INIT` | 5.25 | 1.5 |
| Muon momentum | 0.97 | 0.99 |
| Muon / Adam WD | 0.095 | 0.04 |
| Matrix LR | 0.03 | 0.025 |
| EMA decay | 0.9965 | 0.997 |
| Warmdown iters | 5000 | 3500 |
| Late QAT threshold | 0.25 | 0.15 |
| Bigram vocab | 4096 | 2048 |

Everything else (EMA, SWA every 50 steps, Muon + AdamW split, grad clip 0.3,
batch 786,432 tokens, seq_len 2048) stays at the 1.1228 settings.

### Compression pipeline (ported from V7)
1. Collect input-activation Hessians from 64 training batches
2. Full-Hessian GPTQ (column reorder by diag(H), Cholesky inverse, block_size=128)
   with SDClip at k=12.85 σ per-row for int6, k=20.0 σ for int8
3. Selective pruning: binary search how many lowest-impact ±1 entries to zero
4. Encode three ways and keep the smallest: Huffman (canonical), zstd-22, Brotli-11
5. Roundtrip-decode the winner and evaluate

### Eval-time pipeline
1. Int6 roundtrip baseline (stride=seq_len)
2. Sliding window eval (stride=64, competition metric)
3. OGD vocab bias: online per-vocab gradient descent from scored tokens only
4. N-gram oracle: bigram+trigram table built from scored tokens, mixed with
   neural softmax via entropy gating (`ngram_weight=0.15`, tuned down from V7's
   too-aggressive 0.3 because the neural model is already far stronger here)

The final reported score is the minimum BPB across all eval methods.

## Expected scoreboard

Honest estimates (these compound sub-additively, not additively):

| Config | Expected BPB | Confidence |
|---|---|---|
| SOTA 1.1228 arch, SP1024 | 1.122 | reproduced |
| +SP8192 + bigbag hparams | 1.085 - 1.095 | medium-high |
| +Full GPTQ + Huffman | ~0 or −0.002 | (compression, not capacity) |
| +Sliding window stride=64 | −0.003 to −0.008 | high |
| +OGD vocab bias | −0.002 to −0.005 | medium |
| +N-gram oracle (gated) | −0.001 to −0.004 | medium |
| **V9 combined (target)** | **~1.07 - 1.08** | — |

Anything below 1.08 beats the current visible leaderboard. Anything below 1.10
is a real submission.

## Run it

```bash
# Validate on one seed (~15-20 min, ~$5):
bash records/track_10min_16mb/2026-04-15_V9_DeepRetrieval/run_8xh100.sh

# Full 3-seed submission (~60 min, ~$15):
bash records/track_10min_16mb/2026-04-15_V9_DeepRetrieval/run_8xh100.sh all
```

## Files

- `train_gpt.py` — full trainer + compression + eval (self-contained, ~1900 lines)
- `run_8xh100.sh` — installs deps, downloads SP8192, runs `torchrun`, scrapes scores
- `README.md` — this file

## Parameter budget

| Component | Params | Estimated compressed |
|---|---|---|
| Tied embedding (8192×512, int8) | 4.2M | ~3.0 MB |
| 11 blocks (attn + MLP, int6 GPTQ) | ~23.7M | ~10.8 MB |
| BigramHash (4096×128) + VE + skips + scalars | ~0.6M | ~0.3 MB |
| Python code | — | ~0.1 MB |
| **Total** | **~28.5M** | **~14.2 MB** |

SP8192 costs ~2 MB over SP1024 in the embedding alone. Huffman + full GPTQ +
selective pruning should keep us under the 16 MB artifact limit even so. If
a seed comes out over budget, the first two knobs to try are
`SDCLIP_K_INT6` (raise to 14-16 for tighter clip, more |q|=1 entries to prune)
and `NUM_LAYERS` (drop to 10 for ~1.4 MB of headroom).
