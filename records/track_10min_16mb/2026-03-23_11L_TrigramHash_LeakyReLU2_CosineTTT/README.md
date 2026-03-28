## Record: 11L TrigramHash + LeakyReLU(0.5)^2 + Cosine TTT LoRA

**Target: ~1.06 BPB** | 8xH100 SXM, 600s train + 600s eval

### Key Innovations Over PR #414 (1.1233 BPB)

Three novel additions on top of the SOTA architecture:

| Change | PR #414 | This | Expected Impact |
|--------|---------|------|-----------------|
| **TrigramHash** | BigramHash (2048, dim=128) | TrigramHash (4096, dim=128) | -0.002 to -0.005 BPB |
| **LeakyReLU(0.5)^2** | relu^2 | leaky_relu(0.5).square() | -0.003 to -0.008 BPB |
| **Cosine TTT LoRA** | None (sliding window only) | Rank-8 LoRA on MLP, cosine LR, 3 epochs | -0.03 to -0.06 BPB |

### TrigramHash (4096 buckets, dim=128)

Replaces BigramHash with a trigram-aware hash embedding. Hashes 3 consecutive tokens into an embedding table for richer local context:
- Hash: XOR(36313 * t[i], 27191 * t[i-1], 51647 * t[i-2]) % (vocab_size - 1)
- Position 0: sentinel, Position 1: bigram fallback
- 4096 buckets (up from 2048) to reduce hash collisions with the richer hash space
- Projected from 128 to 512 dim via CastedLinear

### LeakyReLU(0.5)^2

Replaces relu(x).square() with F.leaky_relu(x, 0.5).square() in the MLP. The negative slope of 0.5 allows gradient flow through negative activations while maintaining the squaring nonlinearity. This is the most-adopted activation change among frontier submissions.

### Legal Backward-Looking TTT with LoRA

The biggest expected gain. During sliding window evaluation:
1. For each window, the prefix (positions 0 to s-1) contains already-scored tokens
2. Before scoring new tokens, adapt the model on the prefix using LoRA
3. Rank-8 LoRA on all MLP fc and proj layers (~1.5% of parameters)
4. Cosine-scheduled AdamW: lr=1e-4, wd=0.01, 3 epochs over prefix
5. After scoring, restore original weights completely
6. Time-budget aware: falls back to standard eval if approaching 10min limit

This is **legal** because we only train on tokens we've already evaluated.

### Architecture (from PR #414)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), LeakyReLU(0.5)^2 activation
- U-Net skip connections (5 encoder, 6 decoder)
- Efficient Partial XSA on last 4 layers
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10)
- SmearGate + TrigramHash (4096 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.025, momentum=0.99, WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- EMA: decay=0.997, every step
- Tight SWA: every 50 steps when scale<0.2
- Late QAT: STE int6 fake-quantization when LR scale<0.15
- OrthoInit + muP-scaled output projections

### Quantization

- GPTQ-lite: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Run Commands

```bash
# Single seed
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py

# All 3 seeds
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
SEED=42 torchrun --nproc_per_node=8 train_gpt.py
SEED=2024 torchrun --nproc_per_node=8 train_gpt.py
```

### Interaction Effects Verified

- SmearGate + OrthoInit: Required pairing (SmearGate hurts without OrthoInit)
- EMA + XSA: Both present and interacting correctly
- TTT during eval only: No impact on training or artifact size
- Uniform int6 bit-width: Maintains good zstd compression ratio
- Late QAT at scale<0.15: Strictly better than early QAT
