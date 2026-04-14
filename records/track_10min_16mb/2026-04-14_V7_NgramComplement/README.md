# V7: Complementary Neural-Ngram System

## The Core Idea (What Makes This Different)

Every top submission trains a neural model, quantizes it, and evaluates it.
V7 trains a neural model that is **deliberately complementary to a simple
bigram model**, then at eval time **mixes the two systems together** using the
neural model's own uncertainty as the mixing gate.

The insight: a standard neural LM wastes parameters memorizing that "of" follows
"because" or "the" follows "in". A bigram table handles those patterns for free.
If we train the neural model to focus on what bigrams CAN'T predict (syntax,
long-range dependencies, rare constructions), and then let a bigram oracle handle
the easy patterns at eval time, the combined system beats either one alone.

## Three Novel Techniques

### 1. N-gram Complementary Training (training-time)
Before training begins, we build a hash-based bigram frequency table from 5M
training tokens (~0.5s). During training, each token gets a loss weight based on
how confident the bigram model is about it:
- P(target|prev) > 0.8 → weight = 0.3 (easy, let n-gram handle it)
- P(target|prev) ≈ 0.0 → weight = 1.0 (hard, neural model must learn this)

This forces the neural model to specialise on genuinely difficult predictions
rather than memorising frequency patterns.

### 2. N-gram Oracle Eval with Entropy Gating (eval-time)
At eval time, we build a bigram+trigram frequency table from scored tokens as we
process windows sequentially. For each prediction, we mix the neural model's
output with the n-gram prediction:

    mixed = (1 - alpha) * neural_probs + alpha * ngram_probs

where alpha = neural_entropy / max_entropy * ngram_weight

- Neural model uncertain (high entropy) → trust n-gram more
- Neural model confident (low entropy) → trust neural more

This is more principled than fixed-alpha mixing used in n-gram track submissions.

### 3. Self-Distillation Between Recursive Passes
In our recursive architecture (5 base blocks × 2 loops), the first pass produces
intermediate predictions. We add a KL-divergence loss that encourages the second
pass (with LoRA) to REFINE the first pass's predictions, not just predict
independently. This makes the second pass a true refinement rather than
a redundant computation.

## Architecture
- **SP8192 tokenizer** (8× larger vocabulary than SP1024)
- **5 base blocks × 2 loops** = 10 effective virtual layers with LoRA
- **dim=512**, 8 heads, 4 KV heads (head_dim=64)
- **ReLU^2** activation (NOT LeakyReLU — preserves sparsity)
- U-Net skip connections, XSA on last 4 virtual layers
- Value Embedding on virtual layers 8, 9
- Per-virtual-layer LN scaling
- ~17.7M total params, estimated ~13.7MB artifact

## Changes from V6
| Change | V6 | V7 | Why |
|--------|-----|-----|-----|
| Tokenizer | SP1024 | SP8192 | 8× vocab = better bytes-per-token |
| Model dim | 640 | 512 | Fits larger embedding table |
| MLP activation | LeakyReLU(0.5)^2 | ReLU^2 | Restores sparsity (LeakyReLU destroys it) |
| LoRA optimizer | Shared with scalars | Dedicated (lr=0.06) | Faster LoRA adaptation |
| Entropy reg lambda | 0.01 | 0.03 | Stronger compression signal |
| Late QAT threshold | 0.15 | 0.25 | More training steps with entropy reg |
| N-gram training | None | Complementary loss weighting | Neural specialises on hard tokens |
| N-gram eval | None | Oracle with entropy gating | +0.05-0.15 BPB improvement |
| Self-distillation | None | KL loss between passes | Second pass refines first |

## Run Instructions
```bash
# Single seed validation (~20 min, ~$5):
bash records/track_10min_16mb/2026-04-14_V7_NgramComplement/run_8xh100.sh

# Full 3-seed submission (~60 min, ~$15):
bash records/track_10min_16mb/2026-04-14_V7_NgramComplement/run_8xh100.sh all
```

## Expected Score Breakdown
- Base model (no n-gram, no OGD): ~1.10 BPB (conservative, SP8192 helps a lot)
- + Sliding window eval: ~1.08 BPB
- + OGD bias: ~1.07 BPB
- + N-gram oracle mixing: ~1.00-1.03 BPB (the big win)
- Target: sub-1.05 BPB to be competitive with current frontier
