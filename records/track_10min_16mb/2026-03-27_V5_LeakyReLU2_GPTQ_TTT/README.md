## V5: LeakyReLU^2 + Full Hessian GPTQ (Verified 8xH100)

**val_bpb: 1.1451** (sliding window stride=64, single seed) | **16.1 MB** (over limit) | 8xH100 SXM, 601s train

### Result Summary

| Eval Method | val_loss | val_bpb | Notes |
|---|---|---|---|
| Post-EMA diagnostic | 1.9348 | 1.1459 | Before quantization |
| Int6 roundtrip | 1.9740 | 1.1691 | After GPTQ quantization |
| Sliding window (stride=64) | 1.9334 | **1.1451** | Best score |
| TTT (warmstarted LoRA) | 1.9461 | 1.1526 | TTT degraded by 0.0075 |

### What Worked

1. **LeakyReLU(0.5)^2** replacing ReLU^2 — preserves gradient flow through negative pre-activations
2. **Full Hessian GPTQ** with column reordering by diag(H), Cholesky inverse, block-wise error propagation — lower quantization MSE than GPTQ-lite clip search
3. **Selective pruning** via binary search over lowest-impact |q|=1 weights

### What Failed

1. **TTT made the score worse** (1.1526 vs 1.1451 without). Only 5,100 of 121,136 windows were processed before the 570s timeout. Those windows averaged ~1.28 BPB, dragging the overall average up. The partial coverage means TTT windows hurt while non-TTT windows carry the real score.

2. **Artifact is 103KB over the 16MB limit** (16,103,628 bytes). The selective pruning binary search exhausted all |q|=1 weights and still couldn't fit. Needs to also target |q|=2 weights or reduce code size.

3. **Step time degradation**: started at 86ms/step (matching SOTA) but crept to 112ms by step 5341. SOTA maintains 84-90ms. This resulted in only 5,341 steps vs SOTA's ~7,100 — a 25% training deficit.

### Architecture (from PR #414)
- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), LeakyReLU(0.5)^2 activation
- U-Net skip connections (5 encoder, 6 decoder)
- XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training
- Muon optimizer (matrices): lr=0.025, momentum=0.99
- AdamW (embeddings): lr=0.035, WD=0.04
- EMA (decay=0.997) + Tight SWA
- Late QAT at scale<0.15
- OrthoInit + muP-scaled output projections

### Quantization
- Full GPTQ: Hessian-aware quantization with column reordering, Cholesky inverse, block-wise error propagation (block_size=128)
- Selective Pruning: binary search to zero smallest int6 weights
- Int6 per-row for MLP + attention, Int8 per-row for embeddings
- zstd level 22 compression (fell back to zlib on RunPod — possible compression difference)

### Training Log
```
step:500  train_loss:2.3794 step_avg:86.42ms
step:1000 train_loss:2.2562 step_avg:97.09ms
step:2000 train_loss:2.0477 step_avg:106.92ms
step:4000 train_loss:1.9166 val_bpb:1.1877 step_avg:111.29ms
step:5341 train_loss:— val_bpb:1.1465 step_avg:112.64ms (wallclock cap)
```

### Known Issues to Fix
1. Disable TTT or fix per-window adaptation (needs full coverage or better fallback merging)
2. Extend selective pruning to |q|=2 weights to fix artifact size
3. Diagnose step time degradation (86ms -> 112ms over training)
4. Ensure zstd-22 is available on RunPod (log shows zlib fallback)
