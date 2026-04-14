# V6: Recursive Depth + Huffman Entropy Coding

## Design Rationale

### Core Idea: Depth Recurrence
Every competitive submission uses 11 unique transformer layers at ~2.4MB each (int6 quantized).
V6 uses **5 base blocks looped 2x** to get 10 effective layers at roughly half the
parameter cost. The freed budget (~7MB) is spent by widening the model from 512-dim
to 640-dim, which improves per-layer representation capacity.

### LoRA Adapters for Loop Differentiation
On the second pass through each base block, a tiny LoRA modulation (rank 8) is added
to the Q and K projections:

    W_effective = W_base + B @ A

- A: (rank=8, in_dim), initialised with small random values
- B: (out_dim, rank=8), initialised to zero (first pass = identity)
- This lets the second pass specialise without duplicating weight matrices
- Total LoRA overhead: ~62K params across all 5 blocks

### Virtual Layer Mapping
```
Virtual Layer  → Base Block  Loop Iter  LoRA  XSA  VE
vi=0           → block 0     0          No    No   No
vi=1           → block 1     0          No    No   No
vi=2           → block 2     0          No    No   No
vi=3           → block 3     0          No    No   No
vi=4           → block 4     0          No    No   No
vi=5           → block 0     1          Yes   No   No
vi=6           → block 1     1          Yes   Yes  No
vi=7           → block 2     1          Yes   Yes  No
vi=8           → block 3     1          Yes   Yes  Yes
vi=9           → block 4     1          Yes   Yes  Yes
```

### Architecture Summary
- 5 base blocks × 2 loops = 10 effective virtual layers
- 640-dim, 10 heads, 5 KV heads (GQA 2:1), head_dim=64
- 3× MLP expansion (1920 hidden), LeakyReLU(0.5)^2
- U-Net skip connections on virtual layer indices
- XSA on last 4 virtual layers (vi 6-9)
- LN Scale Factor: 1/sqrt(vi+1) per virtual layer (not per base block)
- Value Embedding on virtual layers 8, 9
- SmearGate, BigramHash, Partial RoPE (16/64), logit softcap=30

### What Changed from V5
1. **Recursive architecture** replaces 11 unique blocks with 5 base + LoRA
2. **Model widened** from 512 to 640 dimensions (25% wider)
3. **TTT removed** — it was expensive and hurt score by 0.0075 BPB
4. **Huffman entropy coding** replaces zstd, with per-tensor canonical tables
5. **Differentiable entropy regularization** during late QAT (soft-histogram approach)
6. **OGD vocab bias** at eval time (lightweight TTT replacement)
7. **Per-virtual-layer LN scaling** instead of per-base-block

### Compression Pipeline
1. Train with late QAT (int6 fake quantisation in the last ~15% of training)
2. Entropy regularization encourages compressible weight distributions
3. Collect Hessians and run full GPTQ for int6 quantisation
4. Selective pruning of ±1 weights to fit budget
5. Encode with Huffman coder (compare vs zstd-22, use whichever is smaller)
6. Final artifact = code bytes + weights blob ≤ 16MB

### Hyperparameters (env vars)
| Variable | Default | Description |
|---|---|---|
| NUM_BASE_LAYERS | 5 | Number of unique transformer blocks |
| LOOP_COUNT | 2 | Times each block is executed |
| LORA_RANK | 8 | LoRA rank for Q/K modulation |
| MODEL_DIM | 640 | Model width |
| NUM_HEADS | 10 | Attention heads |
| NUM_KV_HEADS | 5 | KV heads for GQA |
| XSA_LAST_N | 4 | XSA on last N virtual layers |
| ENTROPY_REG_LAMBDA | 0.01 | Entropy penalty weight during late QAT |
| OGD_BIAS_ENABLED | 1 | Enable OGD bias at eval time |
| OGD_BIAS_LR | 0.1 | Learning rate for OGD vocab bias |

### Parameter Budget
- 5 base blocks: ~14.4M params (int6 → ~10.8MB)
- LoRA adapters: ~62K params
- Embeddings + bigram + VE: ~1.1M params
- Skip weights + scalars: ~5K params
- **Total: ~19.7M params**
- Estimated artifact: ~13.7MB (well under 16MB limit)
- V5 comparison: 27M params, ~14-15MB artifact

### Run Instructions
```bash
# Single seed validation (~15 min, ~$4):
bash records/track_10min_16mb/2026-04-13_V6_Recursive_Entropy/run_8xh100.sh

# Full 3-seed submission (~45 min, ~$12):
bash records/track_10min_16mb/2026-04-13_V6_Recursive_Entropy/run_8xh100.sh all
```
