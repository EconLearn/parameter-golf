"""CPU integration tests — instantiate a tiny GPT, run forward, quantize,
dequantize, and verify the model still produces a sensible distribution.
This catches mismatches between the new code and SOTA's quantization path."""
from __future__ import annotations
import os
import sys
import importlib.util
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
spec = importlib.util.spec_from_file_location("v10", os.path.join(HERE, "train_gpt.py"))
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)
def _ok(msg): print(f"  OK  {msg}")
def _fail(msg): print(f"  FAIL {msg}"); sys.exit(1)
print("V10 INTEGRATION TESTS")
print("-" * 60)
# ------------------------------------------------------------------
# Test: Build a tiny GPT, forward pass, quantize, dequantize, verify model still works
# ------------------------------------------------------------------
print("test_tiny_gpt_roundtrip")
torch.manual_seed(0)
# Tiny config: 2 layers, dim=64, vocab=64
model = v10.GPT(
    vocab_size=64, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
    mlp_mult=2.0, tie_embeddings=True, tied_embed_init_std=0.1,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=64, bigram_dim=16,
    xsa_last_n=0, rope_dims=8, ln_scale=True, dtg=False,
    ve_enabled=False, ve_dim=0, ve_layers="",
).float()
for m in model.modules():
    if isinstance(m, v10.CastedLinear):
        m.float()
v10.restore_low_dim_params_to_fp32(model)
# Forward pass
x = torch.randint(0, 64, (1, 32), dtype=torch.int64)
with torch.no_grad():
    logits_pre = model.forward_logits(x)
_ok(f"tiny GPT instantiated and forward worked; logits shape {tuple(logits_pre.shape)}")
assert logits_pre.shape == (1, 32, 64), f"wrong logits shape: {logits_pre.shape}"
# Run the exact SOTA mixed_quantize_int6 → dequantize roundtrip
sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
quant_result, quant_meta = v10.mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
deq_state = v10.dequantize_mixed_int6(quant_result, quant_meta, sd_cpu)
# Verify every param in the original has a match in deq_state
missing = set(sd_cpu.keys()) - set(deq_state.keys())
if missing:
    _fail(f"dequantize missing keys: {missing}")
extra = set(deq_state.keys()) - set(sd_cpu.keys())
if extra:
    _fail(f"dequantize has extra keys: {extra}")
_ok(f"quantize+dequantize roundtrip preserves all {len(sd_cpu)} state_dict keys")
# Load dequantized weights and verify forward is stable (not identical — int6 is lossy)
model_q = v10.GPT(
    vocab_size=64, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
    mlp_mult=2.0, tie_embeddings=True, tied_embed_init_std=0.1,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=64, bigram_dim=16,
    xsa_last_n=0, rope_dims=8, ln_scale=True, dtg=False,
    ve_enabled=False, ve_dim=0, ve_layers="",
).float()
for m in model_q.modules():
    if isinstance(m, v10.CastedLinear):
        m.float()
v10.restore_low_dim_params_to_fp32(model_q)
model_q.load_state_dict(deq_state, strict=True)
with torch.no_grad():
    logits_post = model_q.forward_logits(x)
# Logits should be CLOSE but not identical (int6 lossy). Check they're not garbage.
diff = (logits_post - logits_pre).abs()
if diff.max().item() > 30.0:
    _fail(f"quantize roundtrip destroyed logits: max diff {diff.max().item()}")
_ok(f"quantized model forward stable; mean |diff| {diff.mean().item():.3f} max {diff.max().item():.3f}")
# Verify both produce valid probability distributions
p_pre = F.softmax(logits_pre.float(), dim=-1)
p_post = F.softmax(logits_post.float(), dim=-1)
assert torch.allclose(p_pre.sum(-1), torch.ones_like(p_pre.sum(-1)), atol=1e-5)
assert torch.allclose(p_post.sum(-1), torch.ones_like(p_post.sum(-1)), atol=1e-5)
_ok("both pre- and post-quant softmax distributions normalize correctly")
# ------------------------------------------------------------------
# Test: NgramEvalOracle.mix_with_neural integrates with real model logits
# ------------------------------------------------------------------
print("test_ngram_with_real_logits")
oracle = v10.NgramEvalOracle(vocab_size=64, hash_size=8192, device="cpu")
# Seed oracle with the same tokens the model was queried on
fake_corpus = torch.randint(0, 64, (5000,), dtype=torch.int64)
oracle.update(fake_corpus)
logits_2d = logits_post[0]  # (32, 64)
prev = x[0]                  # (32,)
pprev = torch.cat([prev[:1], prev[:-1]])
mixed = oracle.mix_with_neural(logits_2d, prev, pprev, ngram_weight=0.10)
assert mixed.shape == (32, 64), f"mix shape wrong: {mixed.shape}"
assert torch.allclose(mixed.sum(-1), torch.ones(32), atol=1e-5), "mix not normalized"
# NLL should be reasonable on a tiny untrained model
y = torch.randint(0, 64, (32,), dtype=torch.int64)
nll_neural = F.cross_entropy(logits_2d.float(), y).item()
nll_mixed = -mixed.clamp_min(1e-10).log().gather(1, y.unsqueeze(1)).squeeze(1).mean().item()
# With ngram_weight=0.10, the mixed NLL shouldn't be crazily different from neural
if abs(nll_mixed - nll_neural) > 1.5:
    _fail(f"mix at weight=0.10 too different from neural: {nll_neural:.3f} vs mixed {nll_mixed:.3f}")
_ok(f"mix integrates: neural NLL={nll_neural:.3f} mixed NLL={nll_mixed:.3f}")
# ------------------------------------------------------------------
# Test: flash_attn fallback actually works (SDPA path with GQA)
# ------------------------------------------------------------------
print("test_attention_fallback")
# The tiny model above used num_heads=4, num_kv_heads=2, so GQA was active.
# If the fallback flash_attn_3_func is broken, the forward pass above would
# have errored or produced NaNs. So reaching this point means SDPA+GQA works.
assert not torch.isnan(logits_pre).any(), "NaN in pre-quant logits"
assert not torch.isnan(logits_post).any(), "NaN in post-quant logits"
_ok(f"attention path runs without NaN (fallback mode: _FA3={v10._FA3})")
# ------------------------------------------------------------------
# Test: eval_val_sliding_ogd function signature compatible with training
# ------------------------------------------------------------------
print("test_eval_val_sliding_ogd_signature")
import inspect
sig = inspect.signature(v10.eval_val_sliding_ogd)
expected = ["args", "base_model", "rank", "world_size", "device", "val_tokens",
            "base_bytes_lut", "has_leading_space_lut", "is_boundary_token_lut",
            "stride", "eval_seq_len", "log_fn"]
actual = list(sig.parameters.keys())
assert actual == expected, f"signature mismatch: {actual} vs {expected}"
_ok("eval_val_sliding_ogd has expected parameters")
print("-" * 60)
print("ALL V10 INTEGRATION TESTS PASSED")
