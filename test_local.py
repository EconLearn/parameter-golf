#!/usr/bin/env python3
"""Local smoke-test harness for Parameter Golf submissions.
Patches FlashAttention 3 → PyTorch SDPA for non-Hopper GPUs (4060, etc).
Usage: python test_local.py [path_to_train_gpt.py] [--quick]

This does NOT produce valid competition scores — it just verifies:
1. Code runs without crashing
2. Training loop converges (loss decreases)
3. Quantization + compression produces valid artifact
4. TTT evaluation doesn't crash
"""
import sys, os, importlib, types

# Patch FlashAttention 3 → PyTorch SDPA before importing train_gpt
def _make_flash_attn_shim():
    """Create a fake flash_attn_interface module that uses PyTorch SDPA."""
    import torch
    import torch.nn.functional as F

    def flash_attn_func(q, k, v, causal=False, **kwargs):
        # FA3 expects [B, T, H, D], SDPA expects [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Handle GQA: repeat KV heads to match Q heads
        if k.shape[1] != q.shape[1]:
            ratio = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out.transpose(1, 2)  # back to [B, T, H, D]

    mod = types.ModuleType("flash_attn_interface")
    mod.flash_attn_func = flash_attn_func
    # Also alias as flash_attn_3_func since that's what train_gpt.py imports
    mod.__dict__["flash_attn_func"] = flash_attn_func
    return mod

# Install shim BEFORE any import of train_gpt
sys.modules["flash_attn_interface"] = _make_flash_attn_shim()

# Also patch zstandard if not available
try:
    import zstandard
except ImportError:
    pass  # train_gpt.py already falls back to zlib

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("script", nargs="?",
                       default="records/track_10min_16mb/2026-03-23_V3_FixedTTT_XSAall_OGDbias/train_gpt.py")
    parser.add_argument("--quick", action="store_true", help="Very short run (200 steps, small batch)")
    parser.add_argument("--no-ttt", action="store_true", help="Disable TTT for faster testing")
    args = parser.parse_args()

    # Set environment for local testing
    if args.quick:
        os.environ.setdefault("ITERATIONS", "200")
        os.environ.setdefault("TRAIN_BATCH_TOKENS", "8192")
        os.environ.setdefault("VAL_BATCH_SIZE", "8192")
        os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "120")
        os.environ.setdefault("WARMDOWN_ITERS", "50")
        os.environ.setdefault("VAL_LOSS_EVERY", "100")
        os.environ.setdefault("TRAIN_LOG_EVERY", "10")
    if args.no_ttt:
        os.environ["TTT_ENABLED"] = "0"

    # Force single GPU
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    # Check data exists
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at {data_path}")
        print("Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1")
        sys.exit(1)

    # Import and run
    script_path = os.path.abspath(args.script)
    spec = importlib.util.spec_from_file_location("train_gpt", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_gpt"] = mod
    spec.loader.exec_module(mod)

    if hasattr(mod, "main"):
        mod.main()
    else:
        print("ERROR: train_gpt.py has no main() function")

if __name__ == "__main__":
    main()
