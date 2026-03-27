## V4: Full GPTQ + Selective Pruning + LeakyReLU^2 + XSA-all

**val_bpb: TBD** (awaiting 8xH100 run) | 8xH100 SXM, 600s

### Changes Over PR #414 (1.1228 BPB)

| Change | PR #414 | This | Expected Impact |
|--------|---------|------|-----------------|
| **Activation** | relu(x).square() | leaky_relu(x, 0.5).square() | -0.001 to -0.003 BPB |
| **XSA** | Last 4 layers | All 11 layers | -0.003 to -0.005 BPB |
| **Quantization** | GPTQ-lite (5 clip percentiles) | Full Hessian-aware GPTQ | -0.002 to -0.005 BPB |
| **Compression** | Fixed 5% prune | Binary-search selective prune to 15.9MB | Better artifact utilization |

### Architecture (from PR #414)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), LeakyReLU(0.5)^2 activation
- U-Net skip connections (5 encoder, 6 decoder)
- XSA on all 11 layers (GQA-aware, zero-alloc)
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

- **Full GPTQ**: Hessian-aware quantization with column reordering, Cholesky inverse, block-wise error propagation (block_size=128), 5 clip percentiles
- **Selective Pruning**: Binary search to zero smallest int6 weights until compressed artifact fits under 15.9MB
- Int6 per-row for MLP + attention, Int8 per-row for embeddings
- zstd level 22 compression
