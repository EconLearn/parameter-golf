# Parameter Golf Run Plan — Minimize Wasted H100 Hours

## Cost: 8xH100 SXM = $21.52/hr. Each training run = ~12 min = ~$4.30.

## Phase 1: Baseline Validation (1 run, ~$5)
Verify V3 produces a valid artifact and the training loop works.
```bash
cd /workspace && git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
cp records/track_10min_16mb/2026-03-23_V3_FixedTTT_XSAall_OGDbias/train_gpt.py .
SEED=1337 TTT_ENABLED=0 torchrun --nproc_per_node=8 train_gpt.py
```
Expected: ~1.120-1.125 BPB without TTT. If >1.130, something is wrong.

## Phase 2: Ablation (4 runs, ~$20)
Test each new technique individually against the baseline.

### Run 2a: XSA-all vs XSA-4 (is XSA on all layers actually better?)
```bash
SEED=1337 TTT_ENABLED=0 XSA_LAST_N=4 torchrun --nproc_per_node=8 train_gpt.py
```

### Run 2b: LeakyReLU vs ReLU (does our activation help or hurt?)
Temporarily edit MLP.forward back to relu, run same seed.

### Run 2c: TTT only (does TTT improve over non-TTT baseline?)
```bash
SEED=1337 TTT_ENABLED=1 torchrun --nproc_per_node=8 train_gpt.py
```

### Run 2d: TTT with XSA-skip vs without (does Gemini's layer targeting help?)
```bash
SEED=1337 TTT_ENABLED=1 XSA_SKIP_LAYERS="" torchrun --nproc_per_node=8 train_gpt.py
```

## Phase 3: Final Submission (3 runs, ~$15)
Run the best configuration with 3 seeds.
```bash
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
SEED=42 torchrun --nproc_per_node=8 train_gpt.py
SEED=2024 torchrun --nproc_per_node=8 train_gpt.py
```

## Total estimated cost: ~$40 (~2 hours of 8xH100)
With $500 in credits, we have budget for ~100 runs. Use remaining budget for:
- Hyperparameter sweeps (TTT lr, epochs, warmstart decay)
- Testing kNN-Cache if line budget allows
- Testing with techniques from winning PRs (Selective Pruning, Parallel Muon)

## Key Decision Points:
- After Phase 1: If baseline >1.130, debug before proceeding
- After Phase 2: Drop any technique that hurts. Stack only proven winners.
- After Phase 3: If mean BPB < 1.1144, submit PR. Otherwise iterate.

## Critical Timing:
- Pod spin-up: ~5 min
- Data download: ~3 min
- Each training run: ~12 min
- Each TTT eval: ~8-10 min
- Total Phase 1-3: ~2.5 hours wall clock
