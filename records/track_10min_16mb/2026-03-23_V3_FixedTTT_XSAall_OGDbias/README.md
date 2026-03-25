# V3: Fixed TTT + XSA-all + Selective Pruning + OGD Bias

Based on 2026-03-22 SOTA (1.1233 BPB, PR #414). Target: sub-1.10 BPB.

## Changes from SOTA

| # | Technique | Source | Expected Impact |
|---|-----------|--------|-----------------|
| 1 | **Fixed LoRA TTT** | V2 bug fix | Enables TTT (was broken) |
| 2 | **XSA on all 11 layers** | PR #609 (1.1154) | -0.003 to -0.007 BPB |
| 3 | **LeakyReLU(0.5)²** | 10+ frontier adopters | -0.001 to -0.002 BPB |
| 4 | **Selective Pruning** | PR #609 | Better compression ratio |
| 5 | **Smart XSA/TTT targeting** | Gemini analysis | XSA off on L9,10 for TTT V-adaptation |
| 6 | **OGD vocab bias** | Novel | -0.003 BPB (free) |
| 7 | **Temp calibration T=0.98** | PR #576 | -0.003 BPB |
| 8 | **PCA row sort for zstd** | Gemini (disabled default) | 10-15% compression gain |

### 1. Fixed LoRA TTT
V2 had showstopper: `.data =` bypasses autograd. Fix: monkey-patch forward() with
`F.linear(x, orig_w + lora_B @ lora_A)` keeping LoRA in autograd graph.
AdamW + cosine LR, 3 epochs/window, warm-start 0.5x decay. Score-first pattern.

### 2. XSA on all layers + smart TTT targeting
XSA on all 11 layers during training (PR #609 proved better than last-4).
During TTT eval: XSA disabled on layers 9,10 only (via XSA_SKIP_LAYERS="9,10").
TTT adapts V-projections on those layers + Q-projections on all layers.
Insight: XSA removes self-value projection that TTT needs to function.

### 3. Selective Pruning (from PR #609)
Post-quantization: finds int6 weights with |q|==1 (smallest magnitude), ranks by
scale² impact, zeros out bottom 5%. More zeros = better zstd compression.

### 4. OGD vocab bias
Online gradient descent updating a vocab-sized bias vector after each scored window.
Additive with TTT, zero artifact cost.

## Architecture (unchanged from SOTA)
- 11L transformer, 512-dim, 8H (4 KV, GQA), 3× MLP (1536)
- U-Net skip connections, Partial RoPE (16/64), SmearGate + BigramHash + OrthoInit
- Shared Value Embedding (128-dim, layers 9,10), logit softcap=30.0
- EMA (0.997) + SWA, Late QAT at scale<0.15, GPTQ-lite + Selective Pruning
- Int6 (MLP+attn) + Int8 (embeddings) + zstd-22

## Run
```bash
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```
