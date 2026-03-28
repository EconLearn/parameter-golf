## V5: LeakyReLU^2 + Full Hessian GPTQ + Legal TTT

**val_bpb: TBD** | 8xH100 SXM, 600s train + 600s eval

### Three edges over SOTA (PR #414, 1.1228 BPB):

1. **LeakyReLU(0.5)^2** replaces ReLU^2 in MLP — preserves negative gradient flow
2. **Full Hessian GPTQ** with column reordering + Cholesky inverse (vs GPTQ-lite clip search)
3. **Legal backward-looking TTT** with LoRA rank-8, OGD vocab bias, warmstart decay

### Architecture
- 11L, 512-dim, 8 heads (4 KV, GQA), 3x MLP (1536 hidden)
- XSA on last 4 layers (matching SOTA step time ~84-90ms/step)
- U-Net skip connections, Partial RoPE (16/64), LN Scale
- SmearGate + BigramHash + OrthoInit
- Shared Value Embedding (layers 9,10)
- EMA (0.997) + Tight SWA + Late QAT@0.15

### TTT Evaluation
- LoRA on Q projections (all layers) + V projections (layers 9,10 with XSA disabled)
- AdamW optimizer, cosine LR, 3 epochs per window
- OGD vocab bias (zero artifact cost)
- Warmstart LoRA between overlapping windows (0.5x decay)
- Temperature calibration T=0.98
- Time budget: 570s with graceful fallback to batched non-TTT eval

### Quantization
- Full Hessian GPTQ: column reordering by Hessian diagonal, Cholesky inverse, block-wise error propagation
- Selective pruning: binary search to fit under 16MB by zeroing lowest-impact quantized weights
- Int6 per-row for MLP + attention, Int8 for embeddings, FP32 for control tensors
- Zstd-22 compression
