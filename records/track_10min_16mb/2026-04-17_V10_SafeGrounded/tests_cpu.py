"""CPU-only unit tests for V10's eval-time additions.
Run: python3 tests_cpu.py   (no GPU required)
Validates NgramEvalOracle and the OGD bias math on fake data.
If these fail, do NOT spend credits on a RunPod run."""
from __future__ import annotations
import math
import os
import sys
import importlib.util
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
# Import V10 module without running main()
spec = importlib.util.spec_from_file_location("v10", os.path.join(HERE, "train_gpt.py"))
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)
def _ok(msg: str) -> None:
    print(f"  OK  {msg}")
def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")
    sys.exit(1)
print("V10 CPU TESTS")
print("-" * 60)
# ------------------------------------------------------------------
# Test 1: NgramEvalOracle basic shape + normalization
# ------------------------------------------------------------------
print("test_ngram_shape_and_normalization")
torch.manual_seed(0)
V = 128  # small vocab for fast test
T = 64
H = 16384
oracle = v10.NgramEvalOracle(vocab_size=V, hash_size=H, device="cpu")
# Seed with some "training" data
fake_tokens = torch.randint(0, V, (2000,), dtype=torch.int64)
oracle.update(fake_tokens)
if oracle.bi_pair.sum().item() < 1000:
    _fail(f"bi_pair not accumulating: sum={oracle.bi_pair.sum().item()}")
_ok(f"oracle.update accumulated {int(oracle.bi_pair.sum().item())} bigram pairs from 2000 tokens (expected 1999)")
if oracle.tri_pair.sum().item() < 1000:
    _fail(f"tri_pair not accumulating: sum={oracle.tri_pair.sum().item()}")
_ok(f"oracle.update accumulated {int(oracle.tri_pair.sum().item())} trigram pairs (expected 1998)")
# Now mix
prev = torch.randint(0, V, (T,), dtype=torch.int64)
pprev = torch.cat([prev[:1], prev[:-1]])
logits = torch.randn(T, V)
mixed = oracle.mix_with_neural(logits, prev, pprev, ngram_weight=0.1)
if mixed.shape != (T, V):
    _fail(f"mix_with_neural output shape: got {mixed.shape}, expected ({T}, {V})")
_ok(f"mix output shape {tuple(mixed.shape)}")
row_sums = mixed.sum(-1)
if not torch.allclose(row_sums, torch.ones(T), atol=1e-5):
    _fail(f"mixed probs don't sum to 1: max dev = {(row_sums - 1.0).abs().max().item():.2e}")
_ok(f"mixed rows sum to 1 (max dev {(row_sums - 1.0).abs().max().item():.2e})")
if (mixed < 0).any() or (mixed > 1.0 + 1e-5).any():
    _fail("mixed contains out-of-range values")
_ok("mixed is in [0, 1]")
# ------------------------------------------------------------------
# Test 2: NgramEvalOracle — at ngram_weight=0, mix should equal softmax(logits)
# ------------------------------------------------------------------
print("test_ngram_weight_zero_passthrough")
mix0 = oracle.mix_with_neural(logits, prev, pprev, ngram_weight=0.0)
soft = F.softmax(logits.float(), dim=-1)
if not torch.allclose(mix0, soft, atol=1e-5):
    _fail(f"ngram_weight=0 should equal neural softmax: max dev = {(mix0 - soft).abs().max().item():.2e}")
_ok(f"ngram_weight=0 → pure neural (max dev {(mix0 - soft).abs().max().item():.2e})")
# ------------------------------------------------------------------
# Test 3: NgramEvalOracle — mix_with_neural works without pprev (bigram-only)
# ------------------------------------------------------------------
print("test_ngram_bigram_only")
mix_bi = oracle.mix_with_neural(logits, prev, None, ngram_weight=0.2)
if mix_bi.shape != (T, V):
    _fail("bigram-only mix shape wrong")
if not torch.allclose(mix_bi.sum(-1), torch.ones(T), atol=1e-5):
    _fail("bigram-only mix doesn't sum to 1")
_ok("bigram-only mix is a valid distribution")
# ------------------------------------------------------------------
# Test 4: NgramEvalOracle — update on very short sequence doesn't crash
# ------------------------------------------------------------------
print("test_ngram_edge_cases")
oracle.update(torch.tensor([5], dtype=torch.int64))  # 1 token → no-op
oracle.update(torch.tensor([5, 7], dtype=torch.int64))  # 2 tokens → one bigram, no trigram
oracle.update(torch.tensor([], dtype=torch.int64))  # empty
_ok("short sequence updates don't crash")
# ------------------------------------------------------------------
# Test 5: NgramEvalOracle — biased toward high-freq bigrams after heavy update
# ------------------------------------------------------------------
print("test_ngram_learns_frequency")
oracle2 = v10.NgramEvalOracle(vocab_size=V, hash_size=H, device="cpu")
# Create a sequence where token 42 always follows token 17
seq = []
for _ in range(200):
    seq += [17, 42]
oracle2.update(torch.tensor(seq, dtype=torch.int64))
# Now prev=17, check mixed distribution heavily favors 42
prev_17 = torch.tensor([17], dtype=torch.int64)
flat_logits = torch.zeros(1, V)  # uniform neural
mix_pred = oracle2.mix_with_neural(flat_logits, prev_17, None, ngram_weight=0.5)
best_token = mix_pred[0].argmax().item()
if best_token != 42:
    _fail(f"oracle should predict 42 after 17, got {best_token}. prob_42={mix_pred[0, 42].item():.4f}")
_ok(f"oracle correctly predicts token 42 after 17 (prob={mix_pred[0, 42].item():.4f})")
# ------------------------------------------------------------------
# Test 6: OGD gradient sign/magnitude
# ------------------------------------------------------------------
print("test_ogd_math")
# The OGD update is: vb -= lr * (softmax(logits) - one_hot(y)).mean(0)
# This should DECREASE NLL on the same (logits, y) pair.
V2 = 16
T2 = 32
torch.manual_seed(1)
logits = torch.randn(T2, V2)
y = torch.randint(0, V2, (T2,), dtype=torch.int64)
vb = torch.zeros(V2)
lr = 0.5
def nll(vb_v):
    p = F.softmax(logits + vb_v[None, :], dim=-1)
    return -p.clamp_min(1e-10).log().gather(1, y.unsqueeze(1)).squeeze(1).mean().item()
pre = nll(vb)
for _ in range(5):
    probs = F.softmax((logits + vb[None, :]).float(), dim=-1)
    onehot = F.one_hot(y, V2).float()
    vb = vb - lr * (probs - onehot).mean(0)
post = nll(vb)
if post >= pre:
    _fail(f"OGD should decrease NLL: pre={pre:.4f} post={post:.4f}")
_ok(f"OGD reduces NLL: {pre:.4f} → {post:.4f}")
# ------------------------------------------------------------------
# Test 7: Hyperparameters wiring
# ------------------------------------------------------------------
print("test_hyperparameters_wiring")
args = v10.Hyperparameters()
assert args.vocab_size == 1024, f"vocab_size should default to 1024 (SP1024), got {args.vocab_size}"
assert args.num_layers == 11, f"num_layers should be 11, got {args.num_layers}"
assert args.qk_gain_init == 1.5, f"qk_gain_init should be 1.5 (SOTA), got {args.qk_gain_init}"
assert args.matrix_lr == 0.025, f"matrix_lr should be 0.025 (SOTA), got {args.matrix_lr}"
assert args.muon_momentum == 0.99, f"muon_momentum should be 0.99 (SOTA), got {args.muon_momentum}"
assert args.muon_wd == 0.04, f"muon_wd should be 0.04 (SOTA), got {args.muon_wd}"
assert args.adam_wd == 0.04, f"adam_wd should be 0.04 (SOTA), got {args.adam_wd}"
assert args.warmdown_iters == 3500, f"warmdown_iters should be 3500 (SOTA), got {args.warmdown_iters}"
assert args.late_qat_threshold == 0.15, f"late_qat_threshold should be 0.15 (SOTA), got {args.late_qat_threshold}"
assert args.bigram_vocab_size == 2048, f"bigram_vocab_size should be 2048, got {args.bigram_vocab_size}"
assert args.eval_stride == 64, f"eval_stride should be 64"
# V10 additions
assert args.ogd_bias_enabled is True, "ogd_bias_enabled should default True"
assert args.ogd_bias_lr == 0.1, f"ogd_bias_lr should be 0.1, got {args.ogd_bias_lr}"
assert args.ngram_eval_enabled is True, "ngram_eval_enabled should default True"
assert args.ngram_eval_weight == 0.10, f"ngram_eval_weight should be 0.10, got {args.ngram_eval_weight}"
_ok("all hparams match SOTA 1.1228 + V10 additions")
# ------------------------------------------------------------------
# Test 8: Required module surface (make sure we didn't break SOTA)
# ------------------------------------------------------------------
print("test_module_surface_intact")
required = [
    "Hyperparameters", "Muon", "GPT", "Block", "CausalSelfAttention", "MLP",
    "BigramHashEmbedding", "ValueEmbedding", "SmearGate", "CastedLinear", "RMSNorm",
    "Rotary", "apply_rotary_emb", "eval_val", "eval_val_sliding",
    "quantize_int6_per_row", "mixed_quantize_int6", "dequantize_mixed_int6",
    "_classify_param", "build_sentencepiece_luts", "load_validation_tokens",
    "DistributedTokenLoader", "TokenStream", "load_data_shard",
    # V10 additions
    "eval_val_sliding_ogd", "NgramEvalOracle",
]
for name in required:
    if not hasattr(v10, name):
        _fail(f"missing symbol: {name}")
_ok(f"all {len(required)} required symbols present (SOTA intact, V10 additions present)")
print("-" * 60)
print("ALL V10 CPU TESTS PASSED")
