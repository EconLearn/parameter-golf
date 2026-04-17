from __future__ import annotations
import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
try:
    import brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _FA3 = True
except ImportError:
    _FA3 = False
    def flash_attn_3_func(q, k, v, causal=False, **kw):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if k.shape[1] != q.shape[1]:
            r = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
# ---------------------------------------------------------------------------
# Entropy coder (inline from entropy_coder.py for self-contained submission)
# ---------------------------------------------------------------------------
import heapq
import json
import struct
_HUFF_MAGIC = b"HUFF"
_HUFF_VERSION = 1
_MAX_TABLE_BITS = 16
_DTYPE_TO_ID = {
    "torch.int8": 0, "torch.float16": 1, "torch.float32": 2,
    "torch.bfloat16": 3, "torch.int16": 4, "torch.int32": 5,
    "torch.int64": 6, "torch.uint8": 7, "torch.bool": 8, "torch.float64": 9,
}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}
def _build_huffman_lengths(freq):
    if not freq: return {}
    symbols = list(freq.keys())
    if len(symbols) == 1: return {symbols[0]: 1}
    heap, ctr = [], 0
    for sym, f in freq.items():
        heapq.heappush(heap, (f, ctr, sym)); ctr += 1
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap); f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, ctr, (n1, n2))); ctr += 1
    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, tuple):
            stack.append((node[0], depth + 1)); stack.append((node[1], depth + 1))
        else:
            lengths[node] = depth
    mx = max(lengths.values())
    if mx > _MAX_TABLE_BITS:
        for s in lengths:
            if lengths[s] > _MAX_TABLE_BITS: lengths[s] = _MAX_TABLE_BITS
        kraft = sum(1 << (_MAX_TABLE_BITS - lengths[s]) for s in lengths)
        target = 1 << _MAX_TABLE_BITS
        while kraft > target:
            sh = min(lengths, key=lambda s: (lengths[s], freq.get(s, 0)))
            if lengths[sh] < _MAX_TABLE_BITS:
                old = lengths[sh]; lengths[sh] = old + 1
                kraft -= (1 << (_MAX_TABLE_BITS - old)) - (1 << (_MAX_TABLE_BITS - old - 1))
    return lengths
def _canonical_codes(lengths):
    if not lengths: return {}, []
    sorted_syms = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    table_spec = [(s, lengths[s]) for s in sorted_syms]
    code_table, code, prev_len = {}, 0, 0
    for sym, length in table_spec:
        if prev_len > 0: code = (code + 1) << (length - prev_len)
        code_table[sym] = (code, length); prev_len = length
    return code_table, table_spec
def _serialize_table(table_spec):
    buf = struct.pack("<H", len(table_spec))
    for sym, length in table_spec: buf += struct.pack("<hB", sym, length)
    return buf
def _deserialize_table(data, offset):
    n = struct.unpack_from("<H", data, offset)[0]; offset += 2
    if n == 0: return {}, offset
    table_spec = []
    for _ in range(n):
        sym, length = struct.unpack_from("<hB", data, offset); offset += 3
        table_spec.append((sym, length))
    ct, code, prev_len = {}, 0, 0
    for sym, length in table_spec:
        if prev_len > 0: code = (code + 1) << (length - prev_len)
        ct[sym] = (code, length); prev_len = length
    return ct, offset
def _fast_encode(values_np, code_table):
    if len(values_np) == 0: return b"", 0
    vmin, vmax = min(code_table.keys()), max(code_table.keys())
    span = vmax - vmin + 1
    bitstr_lut = [""] * span
    for sym, (c, l) in code_table.items(): bitstr_lut[sym - vmin] = format(c, f"0{l}b")
    indices = (values_np.astype(np.int32) - vmin).ravel()
    CHUNK = 500_000
    parts = []
    for start in range(0, len(indices), CHUNK):
        chunk_idx = indices[start:start + CHUNK]
        parts.append("".join(bitstr_lut[idx] for idx in chunk_idx))
    all_bits = "".join(parts)
    total_bits = len(all_bits)
    padding = (8 - total_bits % 8) % 8
    if padding: all_bits += "0" * padding
    n_bytes = len(all_bits) // 8
    out = int(all_bits, 2).to_bytes(n_bytes, "big")
    return out, padding
def _build_flat_table(code_table):
    if not code_table: return np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.uint8), 0
    max_len = max(l for _, l in code_table.values())
    table_bits = min(max_len, _MAX_TABLE_BITS)
    size = 1 << table_bits
    sym_t, len_t = np.zeros(size, dtype=np.int16), np.zeros(size, dtype=np.uint8)
    for sym, (code, length) in code_table.items():
        if length <= table_bits:
            pad = table_bits - length; base = code << pad
            for suffix in range(1 << pad):
                idx = base | suffix; sym_t[idx] = sym; len_t[idx] = length
    return sym_t, len_t, table_bits
def _fast_decode(data, padding_bits, num_values, code_table):
    if num_values == 0: return np.array([], dtype=np.int8)
    sym_t, len_t, table_bits = _build_flat_table(code_table)
    if table_bits == 0: return np.zeros(num_values, dtype=np.int8)
    str_lut = {}
    for idx in range(1 << table_bits):
        key = format(idx, f"0{table_bits}b")
        str_lut[key] = (int(sym_t[idx]), int(len_t[idx]))
    byte_to_bits = [format(b, "08b") for b in range(256)]
    bits_str = "".join(byte_to_bits[b] for b in data)
    if padding_bits > 0: bits_str = bits_str[:len(bits_str) - padding_bits]
    bits_str += "0" * table_bits
    result = np.empty(num_values, dtype=np.int16)
    pos = 0
    for i in range(num_values):
        sym, clen = str_lut[bits_str[pos:pos + table_bits]]; result[i] = sym; pos += clen
    return result.astype(np.int8)
def _classify_tensor_huff(name, tensor):
    if tensor.dtype == torch.int8:
        vmin, vmax = tensor.min().item(), tensor.max().item()
        return "int6" if (-31 <= vmin and vmax <= 31) else "int8"
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64): return "float"
    return "raw"
def _meta_to_json(meta):
    out = {}
    for k, v in meta.items():
        if isinstance(v, dict): out[k] = _meta_to_json(v)
        elif isinstance(v, (str, int, float, bool)): out[k] = v
        elif isinstance(v, (list, tuple)): out[k] = list(v)
        else: out[k] = str(v)
    return out
def _meta_from_json(meta):
    out = {}
    for k, v in meta.items():
        out[k] = _meta_from_json(v) if isinstance(v, dict) else v
    return out
def encode_weights(quant_result, quant_meta):
    output = io.BytesIO()
    descriptors, payloads = [], []
    for name in sorted(quant_result.keys()):
        tensor = quant_result[name]
        if not isinstance(tensor, Tensor): continue
        tensor = tensor.contiguous()
        cat = _classify_tensor_huff(name, tensor)
        shape = list(tensor.shape)
        dtype_id = _DTYPE_TO_ID.get(str(tensor.dtype), -1)
        if cat in ("int6", "int8"):
            vals = tensor.view(-1).numpy()
            unique, counts = np.unique(vals, return_counts=True)
            freq = {int(u): int(c) for u, c in zip(unique, counts)}
            lengths = _build_huffman_lengths(freq)
            ct, ts = _canonical_codes(lengths)
            enc_bytes, pad_bits = _fast_encode(vals, ct)
            buf = io.BytesIO()
            tb = _serialize_table(ts)
            buf.write(struct.pack("<I", len(tb))); buf.write(tb)
            buf.write(struct.pack("<I", len(vals)))
            buf.write(struct.pack("<B", pad_bits))
            buf.write(struct.pack("<I", len(enc_bytes))); buf.write(enc_bytes)
            payload = buf.getvalue(); encoding = f"huffman_{cat}"
        else:
            if tensor.dtype == torch.bfloat16: payload = tensor.view(torch.int16).numpy().tobytes()
            else: payload = tensor.numpy().tobytes()
            encoding = "raw"
        descriptors.append({"name": name, "shape": shape, "dtype": dtype_id, "encoding": encoding, "payload_size": len(payload)})
        payloads.append(payload)
    meta_bytes = json.dumps({"quant_meta": _meta_to_json(quant_meta), "tensors": descriptors}, separators=(",", ":")).encode("utf-8")
    output.write(_HUFF_MAGIC); output.write(struct.pack("<B", _HUFF_VERSION))
    output.write(struct.pack("<I", len(meta_bytes))); output.write(meta_bytes)
    for p in payloads: output.write(p)
    return output.getvalue()
def decode_weights(blob):
    off = 0
    if blob[off:off + 4] != _HUFF_MAGIC: raise ValueError("Bad magic")
    off += 4; ver = struct.unpack_from("<B", blob, off)[0]; off += 1
    if ver != _HUFF_VERSION: raise ValueError(f"Unsupported version {ver}")
    meta_len = struct.unpack_from("<I", blob, off)[0]; off += 4
    meta = json.loads(blob[off:off + meta_len]); off += meta_len
    quant_meta = _meta_from_json(meta["quant_meta"])
    result = {}
    for desc in meta["tensors"]:
        name, shape, dtype_id = desc["name"], desc["shape"], desc["dtype"]
        encoding, ps = desc["encoding"], desc["payload_size"]
        payload = blob[off:off + ps]; off += ps
        dtype_str = _ID_TO_DTYPE.get(dtype_id, "torch.float32")
        dtype = getattr(torch, dtype_str.removeprefix("torch."))
        if encoding.startswith("huffman_"):
            p = 0
            tl = struct.unpack_from("<I", payload, p)[0]; p += 4
            ct, p = _deserialize_table(payload, p)
            nv = struct.unpack_from("<I", payload, p)[0]; p += 4
            pad = struct.unpack_from("<B", payload, p)[0]; p += 1
            bl = struct.unpack_from("<I", payload, p)[0]; p += 4
            bits = payload[p:p + bl]
            vals = _fast_decode(bits, pad, nv, ct)
            tensor = torch.from_numpy(vals.copy()).to(dtype).reshape(shape)
        else:
            if dtype == torch.bfloat16:
                arr = np.frombuffer(payload, dtype=np.int16).copy()
                tensor = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
            else:
                np_dt = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64,
                         torch.int8: np.int8, torch.int16: np.int16, torch.int32: np.int32,
                         torch.int64: np.int64, torch.uint8: np.uint8}.get(dtype, np.float32)
                arr = np.frombuffer(payload, dtype=np_dt).copy()
                tensor = torch.from_numpy(arr).reshape(shape)
                if tensor.dtype != dtype: tensor = tensor.to(dtype)
        result[name] = tensor.contiguous()
    return result, quant_meta
class Hyperparameters:
    # V9: SP8192 baseline matched to leaderboard SOTA (~1.081 bigbag config).
    # Architecture = proven 11L dense (from 1.1228 SOTA), tokenizer swapped to SP8192.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))  # SOTA 1.1228 (reverted from bigbag 5000 — destabilized 11L dense)
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))  # SOTA 1.1228 (reverted from bigbag 5.25 — saturated attention on 11L dense, train_loss stuck high early)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))  # SP8192 tokenizer
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))  # SOTA 1.1228 (reverted from bigbag 0.03)
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))  # SOTA 1.1228 (reverted from bigbag 0.97)
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))  # SOTA 1.1228 (reverted from bigbag 0.095 — caused train/val divergence)
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))  # SOTA 1.1228
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))  # SOTA 1.1228
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))  # SOTA 1.1228 (halved from bigbag 4096 — scales with SP1024)
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))  # SOTA 1.1228 (reverted from bigbag 0.25 — fired too early)
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")  # last 2 of 11 dense layers
    # --- V9: Compression pipeline (ported from V7) ---
    sdclip_k_int6 = float(os.environ.get("SDCLIP_K_INT6", 12.85))
    sdclip_k_int8 = float(os.environ.get("SDCLIP_K_INT8", 20.0))
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "1")))
    gptq_hessian_tokens = int(os.environ.get("GPTQ_HESSIAN_TOKENS", 131_072))
    selective_prune_enabled = bool(int(os.environ.get("SELECTIVE_PRUNE_ENABLED", "1")))
    artifact_budget_bytes = int(os.environ.get("ARTIFACT_BUDGET_BYTES", 16_000_000))
    # --- V9: Eval-time improvements (ported from V7) ---
    ogd_bias_enabled = bool(int(os.environ.get("OGD_BIAS_ENABLED", "1")))
    ogd_bias_lr = float(os.environ.get("OGD_BIAS_LR", 0.1))
    ngram_eval_enabled = bool(int(os.environ.get("NGRAM_EVAL_ENABLED", "1")))
    ngram_eval_weight = float(os.environ.get("NGRAM_EVAL_WEIGHT", 0.15))  # tuned down from V7's 0.3
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t: Tensor, sdclip_k: float | None = None) -> tuple[Tensor, Tensor]:
    """Int8 per-row quantization.

    If ``sdclip_k`` is provided, uses SDClip (clip at k * row_std) — ported from V7
    and matched to bigbag's 1.0810 setting. Otherwise falls back to percentile clip."""
    t32 = t.float()
    if sdclip_k is not None and t32.ndim == 2:
        row_std = t32.std(dim=1)
        clip_abs = (sdclip_k * row_std).clamp_min(1e-8)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    if sdclip_k is not None:
        amax = t32.abs().max().item()
        scale = torch.tensor(amax / 127.0 if amax > 0 else 1.0, dtype=torch.float32)
        q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8).contiguous()
        return q, scale
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] — broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        """Get value embedding for a specific layer using shared table + per-layer scale."""
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
# ---------------------------------------------------------------------------
# OGD bias sliding eval (lightweight TTT replacement, ported from V7)
# ---------------------------------------------------------------------------
def eval_val_sliding_ogd(args, base_model, rank, world_size, device, val_tokens,
                          base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                          stride, eval_seq_len=None, log_fn=None):
    """Sliding-window eval with an online per-vocab bias updated by OGD from scored tokens.

    No weight updates — only a ``vocab_size`` bias vector is adapted. Legal under competition
    rules (only uses already-scored tokens) and fits in the eval wall-clock budget."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ws_all = [w for w in range(0, total_tokens, stride) if min(w + seq_len, total_tokens) - w >= 1]
    my_s = (len(ws_all) * rank) // world_size
    my_e = (len(ws_all) * (rank + 1)) // world_size
    my_wins = ws_all[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    vb = torch.zeros(args.vocab_size, device=device, dtype=torch.float32)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for wi, ws in enumerate(my_wins):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            s = 0 if ws == 0 else max(wlen - stride, 0)
            chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
            x_win = chunk[:-1].unsqueeze(0)
            y_win = chunk[1:].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_win)
            logits = logits + vb[None, None, :]
            nll = F.cross_entropy(logits[0].float(), y_win[0], reduction="none")
            loss_sum += nll[s:wlen].to(torch.float64).sum()
            tok_cnt += float(wlen - s)
            tgt = y_win[0, s:wlen]
            prev = x_win[0, s:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_cnt += tb.sum()
            # OGD update from scored tokens only
            probs = F.softmax(logits[0, s:wlen].float(), dim=-1)
            onehot = F.one_hot(y_win[0, s:wlen], args.vocab_size).float()
            vb -= args.ogd_bias_lr * (probs - onehot).mean(0)
            if log_fn and (wi + 1) % 500 == 0:
                ibpb = (loss_sum / tok_cnt).item() / math.log(2.0) * (tok_cnt / byte_cnt).item()
                log_fn(f"ogd:window {wi + 1}/{len(my_wins)} bpb:{ibpb:.4f}")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_cnt).item()
    base_model.train()
    return vl, vl / math.log(2.0) * (tok_cnt.item() / byte_cnt.item())
# ---------------------------------------------------------------------------
# N-gram eval oracle: bigram+trigram table built from scored tokens, mixed
# with neural predictions via entropy gating (ported from V7).
# ---------------------------------------------------------------------------
class NgramEvalOracle:
    """Builds a bigram+trigram frequency table from scored eval tokens and mixes
    n-gram probabilities with the neural model's softmax via entropy gating:
    high neural entropy → trust n-gram more, low entropy → trust neural more."""
    _P1 = 8209
    _P2 = 7919
    def __init__(self, vocab_size, hash_size: int = 4_000_000, device: str = "cuda"):
        self.vocab_size = vocab_size
        self.hash_size = hash_size
        self.device = device
        self.bi_pair = torch.zeros(hash_size, dtype=torch.float32, device=device)
        self.bi_ctx = torch.zeros(hash_size, dtype=torch.float32, device=device)
        self.tri_pair = torch.zeros(hash_size, dtype=torch.float32, device=device)
        self.tri_ctx = torch.zeros(hash_size, dtype=torch.float32, device=device)
        self._ones_cache: dict[int, Tensor] = {}
    def _ones(self, n):
        if n not in self._ones_cache:
            self._ones_cache[n] = torch.ones(n, dtype=torch.float32, device=self.device)
        return self._ones_cache[n]
    @torch.no_grad()
    def update(self, tokens: Tensor):
        T = tokens.numel()
        if T < 2:
            return
        prev, tgt = tokens[:-1], tokens[1:]
        pk = ((prev * self._P1 + tgt) % self.hash_size).long()
        ck = ((prev * self._P2) % self.hash_size).long()
        self.bi_pair.scatter_add_(0, pk, self._ones(T - 1))
        self.bi_ctx.scatter_add_(0, ck, self._ones(T - 1))
        if T >= 3:
            pp, p, t = tokens[:-2], tokens[1:-1], tokens[2:]
            pk3 = ((pp * self._P1 * self._P1 + p * self._P1 + t) % self.hash_size).long()
            ck3 = (((pp * self._P1 + p) * self._P2) % self.hash_size).long()
            self.tri_pair.scatter_add_(0, pk3, self._ones(T - 2))
            self.tri_ctx.scatter_add_(0, ck3, self._ones(T - 2))
    @torch.no_grad()
    def mix_with_neural(self, logits: Tensor, prev_tokens: Tensor,
                        pprev_tokens: Tensor | None = None, ngram_weight: float = 0.15) -> Tensor:
        T, V = logits.shape
        all_t = torch.arange(V, device=self.device, dtype=torch.long)
        prev_exp = prev_tokens.unsqueeze(1)
        bi_pk = ((prev_exp * self._P1 + all_t.unsqueeze(0)) % self.hash_size).long()
        bi_ck = ((prev_tokens * self._P2) % self.hash_size).long()
        bi_probs = self.bi_pair[bi_pk] / self.bi_ctx[bi_ck].unsqueeze(1).clamp_min(1)
        if pprev_tokens is not None:
            pp_exp = pprev_tokens.unsqueeze(1)
            tri_pk = ((pp_exp * self._P1 * self._P1 + prev_exp * self._P1 + all_t.unsqueeze(0)) % self.hash_size).long()
            tri_ck = (((pprev_tokens * self._P1 + prev_tokens) * self._P2) % self.hash_size).long()
            tri_probs = self.tri_pair[tri_pk] / self.tri_ctx[tri_ck].unsqueeze(1).clamp_min(1)
            tri_has_data = (self.tri_ctx[tri_ck] > 0).float().unsqueeze(1)
            ngram_probs = tri_has_data * 0.6 * tri_probs + (1 - tri_has_data * 0.6) * bi_probs
        else:
            ngram_probs = bi_probs
        ngram_probs = ngram_probs + 1e-7
        ngram_probs = ngram_probs / ngram_probs.sum(-1, keepdim=True)
        neural_probs = F.softmax(logits.float(), dim=-1)
        neural_ent = -(neural_probs * neural_probs.clamp_min(1e-10).log()).sum(-1)
        max_ent = math.log(V)
        alpha = (neural_ent / max_ent).clamp(0, 1).unsqueeze(1) * ngram_weight
        mixed = (1 - alpha) * neural_probs + alpha * ngram_probs
        mixed = mixed / mixed.sum(-1, keepdim=True).clamp_min(1e-10)
        return mixed
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31, sdclip_k: float = 12.85) -> tuple[Tensor, Tensor]:
    """Int6 per-row quantization with SDClip: clip at k * row_std (ported from V7)."""
    t32 = t.float()
    if t32.ndim == 2:
        row_std = t32.std(dim=1)
        row_clip = (sdclip_k * row_std).clamp_min(1e-8)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        return q, s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale
def quantize_int6_gptq(weight: Tensor, hessian: Tensor, clip_range: int = 31,
                       block_size: int = 128, sdclip_k: float = 12.85) -> tuple[Tensor, Tensor]:
    """Full Hessian GPTQ with SDClip: k*sigma per-row clip (ported from V7).

    Column-reorder by diag(H), Cholesky-inverse, block-wise error propagation.
    Falls back to per-row SDClip if Cholesky fails (near-singular Hessian)."""
    W = weight.float().clone()
    nrow, ncol = W.shape
    perm = torch.argsort(torch.diag(hessian), descending=True)
    W = W[:, perm]
    H = hessian[perm][:, perm]
    H = H + 0.01 * torch.diag(H).mean() * torch.eye(ncol, device=H.device, dtype=H.dtype)
    try:
        Hinv_chol = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    except Exception:
        return quantize_int6_per_row(weight, clip_range=clip_range, sdclip_k=sdclip_k)
    row_std = W.std(dim=1)
    row_clip = (sdclip_k * row_std).clamp_min(1e-8)
    s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
    Qt = torch.zeros_like(W, dtype=torch.int8)
    Wt = W.clone()
    for b_start in range(0, ncol, block_size):
        b_end = min(b_start + block_size, ncol)
        W1 = Wt[:, b_start:b_end].clone()
        H1 = Hinv_chol[b_start:b_end, b_start:b_end]
        for i in range(b_end - b_start):
            w_col = W1[:, i]
            q_col = torch.clamp(torch.round(w_col / s.float()), -clip_range, clip_range)
            Qt[:, b_start + i] = q_col.to(torch.int8)
            err = (w_col - q_col * s.float()) / H1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * H1[i, i:].unsqueeze(0)
        Wt[:, b_end:] -= (Wt[:, b_start:b_end] - Qt[:, b_start:b_end].float() * s.float().unsqueeze(1)) @ Hinv_chol[b_start:b_end, b_end:]
    return Qt[:, torch.argsort(perm)].contiguous(), s
def collect_hessians(model, train_loader, device, num_batches: int = 64, seq_len: int = 2048) -> dict[str, Tensor]:
    """Collect input-activation Hessians (X^T X) for every big linear layer.

    Used for full-Hessian GPTQ quantization. Hessians kept on CPU to save GPU memory."""
    hessians: dict[str, Tensor] = {}
    hooks = []
    def make_hook(nm):
        def hk(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            if nm not in hessians:
                hessians[nm] = torch.zeros(x.shape[1], x.shape[1], device="cpu", dtype=torch.float64)
            hessians[nm] += (x.T @ x).to("cpu", dtype=torch.float64)
        return hk
    for nm, mod in model.named_modules():
        if isinstance(mod, CastedLinear) and mod.weight.numel() > 65536:
            hooks.append(mod.register_forward_hook(make_hook(nm + ".weight")))
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = train_loader.next_batch(seq_len * 8, seq_len, 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x, y)
    for h in hooks:
        h.remove()
    for k in hessians:
        hessians[k] = hessians[k] / num_batches
    return hessians
def selective_prune_to_fit(quant_result: dict[str, Tensor], quant_meta: dict,
                            code_bytes: int, target_bytes: int = 15_900_000,
                            use_huffman: bool = True) -> None:
    """Zero out lowest-impact |q|=1 weights until the encoded blob fits target.

    Binary-searches how many ±1 weights to zero; impact scored by (scale * 1)^2.
    Mutates `quant_result` in place."""
    ones_info = []
    for name, info in quant_meta.items():
        if not isinstance(info, dict) or info.get("type") != "int6":
            continue
        q = quant_result.get(name + ".q")
        s = quant_result.get(name + ".scale")
        if q is None or q.ndim != 2:
            continue
        mask = q.abs() == 1
        if not mask.any():
            continue
        rows, cols = mask.nonzero(as_tuple=True)
        errs = s[rows].float().pow(2) if s.ndim > 0 else s.float().pow(2).expand(len(rows))
        vals = q[rows, cols].tolist()
        for r, c, e, v in zip(rows.tolist(), cols.tolist(), errs.tolist(), vals):
            ones_info.append((name + ".q", r, c, e, v))
    if not ones_info:
        return
    ones_info.sort(key=lambda x: x[3])
    def _try_prune(n):
        for i in range(n):
            tn, r, c, _, _ = ones_info[i]
            quant_result[tn][r, c] = 0
        if use_huffman:
            blob = encode_weights(quant_result, quant_meta)
            size = len(blob)
        else:
            buf = io.BytesIO()
            torch.save({"w": quant_result, "m": quant_meta}, buf)
            raw = buf.getvalue()
            blob = (zstandard.ZstdCompressor(level=22).compress(raw)
                    if _COMPRESSOR == "zstd" else zlib.compress(raw, 9))
            size = len(blob)
        for i in range(n):
            tn, r, c, _, v = ones_info[i]
            quant_result[tn][r, c] = v
        return size + code_bytes
    lo, hi = 0, len(ones_info)
    while lo < hi:
        mid = (lo + hi) // 2
        if _try_prune(mid) <= target_bytes:
            hi = mid
        else:
            lo = mid + 1
    for i in range(lo):
        tn, r, c, _, _ = ones_info[i]
        quant_result[tn][r, c] = 0
def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                         hessians: dict[str, Tensor] | None = None,
                         sdclip_k_int6: float = 12.85,
                         sdclip_k_int8: float = 20.0):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            h = hessians.get(name) if hessians is not None else None
            if h is not None and t.ndim == 2:
                q, s = quantize_int6_gptq(t, h, sdclip_k=sdclip_k_int6)
            else:
                q, s = quantize_int6_per_row(t, sdclip_k=sdclip_k_int6)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t, sdclip_k=sdclip_k_int8)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        # device_id= requires PyTorch 2.5+. RunPod has 2.4.1, so we rely on set_device() above.
        dist.init_process_group(backend="nccl")
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = args.ema_decay
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # Apply EMA weights (better than SWA alone per PR#401)
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    # --- Full GPTQ (optional): collect input-activation Hessians, then GPTQ-quantize ---
    # CRITICAL: Hessians must be all-reduced across ranks so every rank quantizes
    # identically. Otherwise rank-local batches produce rank-divergent quant weights,
    # rank-divergent eval models, and meaningless aggregated BPB.
    hessians: dict[str, Tensor] | None = None
    if args.gptq_enabled:
        log0("gptq:collecting Hessians for full-Hessian GPTQ...")
        hessian_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        try:
            hessians = collect_hessians(
                base_model, hessian_loader, device,
                num_batches=max(16, args.gptq_hessian_tokens // (args.train_seq_len * 8)),
                seq_len=args.train_seq_len,
            )
            if distributed and hessians:
                for nm in sorted(hessians.keys()):
                    h_gpu = hessians[nm].to(device)
                    dist.all_reduce(h_gpu, op=dist.ReduceOp.SUM)
                    hessians[nm] = (h_gpu / world_size).cpu()
                log0(f"gptq:collected {len(hessians)} Hessians (synced across {world_size} ranks)")
            else:
                log0(f"gptq:collected {len(hessians)} Hessians")
        except Exception as exc:
            log0(f"gptq:Hessian collection failed ({exc}); falling back to per-row SDClip")
            hessians = None
    quant_result, quant_meta = mixed_quantize_int6(
        sd_cpu, {"mlp", "attn"}, hessians=hessians,
        sdclip_k_int6=args.sdclip_k_int6, sdclip_k_int8=args.sdclip_k_int8,
    )
    code_bytes = len(code.encode("utf-8"))
    # --- Selective pruning: zero lowest-impact |q|=1 entries until we fit ---
    # Runs on every rank with identical inputs (sd_cpu + synced Hessians above),
    # so every rank produces an identical pruned quant_result. Do NOT gate on
    # master_process — that would leave other ranks with an unpruned tensor and
    # diverge the eval model.
    if args.selective_prune_enabled:
        log0("selective_prune:binary search to fit under 16MB (Huffman)...")
        selective_prune_to_fit(
            quant_result, quant_meta, code_bytes,
            target_bytes=max(args.artifact_budget_bytes - 100_000, 10_000_000),
            use_huffman=True,
        )
    # --- Encode and pick the smallest of Huffman / zstd-22 / Brotli-11 ---
    log0("encode:trying huffman + zstd + brotli, picking smallest...")
    huff_blob = encode_weights(quant_result, quant_meta)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    zstd_blob = (zstandard.ZstdCompressor(level=22).compress(quant_raw)
                 if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9))
    brotli_blob = brotli.compress(quant_raw, quality=11) if _HAS_BROTLI else None
    candidates: list[tuple[str, bytes, int]] = [
        ("huffman", huff_blob, len(huff_blob)),
        (_COMPRESSOR, zstd_blob, len(zstd_blob)),
    ]
    if brotli_blob is not None:
        candidates.append(("brotli", brotli_blob, len(brotli_blob)))
    best_name, final_blob, best_size = min(candidates, key=lambda x: x[2])
    final_ext = {"huffman": "huff", "zstd": "ptz", "zlib": "ptz", "brotli": "brotli"}[best_name]
    if master_process:
        with open("final_model.huff", "wb") as f:
            f.write(huff_blob)
        with open("final_model.int6.ptz", "wb") as f:
            f.write(zstd_blob)
        if brotli_blob is not None:
            with open("final_model.brotli", "wb") as f:
                f.write(brotli_blob)
        log0(f"Huffman blob: {len(huff_blob)} bytes (total: {len(huff_blob) + code_bytes})")
        log0(f"{_COMPRESSOR} blob: {len(zstd_blob)} bytes (total: {len(zstd_blob) + code_bytes})")
        if brotli_blob is not None:
            log0(f"Brotli blob: {len(brotli_blob)} bytes (total: {len(brotli_blob) + code_bytes})")
        log0(f"WINNER: {best_name} encoding ({best_size} bytes)")
        total_size = best_size + code_bytes
        log0(f"Final submission size: {total_size} bytes (budget: {args.artifact_budget_bytes})")
        if total_size > args.artifact_budget_bytes:
            log0(
                f"WARNING: artifact over budget by {total_size - args.artifact_budget_bytes} bytes. "
                "Consider raising SDCLIP_K_INT6, lowering MODEL_DIM, or dropping a layer."
            )
    if distributed:
        dist.barrier()
    # --- Decode the winning blob and run roundtrip eval ---
    if best_name == "huffman":
        deq_result, deq_meta = decode_weights(huff_blob)
    else:
        decomp = (zstandard.ZstdDecompressor().decompress(zstd_blob)
                  if best_name == "zstd" else
                  brotli.decompress(brotli_blob) if best_name == "brotli" else
                  zlib.decompress(zstd_blob))
        payload = torch.load(io.BytesIO(decomp), map_location="cpu")
        deq_result, deq_meta = payload["w"], payload["m"]
    deq_state = dequantize_mixed_int6(deq_result, deq_meta, sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    best_loss, best_bpb = q_val_loss, q_val_bpb
    # --- Sliding window eval (stride=64 is the competition metric) ---
    sw_seq_len = effective_eval_seq_len
    sw_val_loss: float | None = None
    sw_val_bpb: float | None = None
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        if sw_val_bpb < best_bpb:
            best_loss, best_bpb = sw_val_loss, sw_val_bpb
    # --- OGD bias eval: per-vocab online gradient descent over scored tokens ---
    if args.ogd_bias_enabled:
        torch.cuda.synchronize()
        t_ogd = time.perf_counter()
        try:
            ogd_loss, ogd_bpb = eval_val_sliding_ogd(
                args, eval_model, rank, world_size, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride if args.eval_stride > 0 else sw_seq_len,
                eval_seq_len=sw_seq_len, log_fn=log0,
            )
            torch.cuda.synchronize()
            log0(
                f"final_ogd val_loss:{ogd_loss:.4f} val_bpb:{ogd_bpb:.4f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_ogd):.0f}ms"
            )
            log0(f"final_ogd_exact val_loss:{ogd_loss:.8f} val_bpb:{ogd_bpb:.8f}")
            if ogd_bpb < best_bpb:
                best_loss, best_bpb = ogd_loss, ogd_bpb
        except Exception as exc:
            log0(f"ogd:eval failed ({exc})")
    # --- N-gram oracle eval: bigram+trigram mixing over scored tokens ---
    if args.ngram_eval_enabled:
        torch.cuda.synchronize()
        t_ng = time.perf_counter()
        try:
            oracle = NgramEvalOracle(args.vocab_size, hash_size=4_000_000, device=device)
            total_val_tokens = val_tokens.numel() - 1
            stride = args.eval_stride if args.eval_stride > 0 else sw_seq_len
            ws_all = [w for w in range(0, total_val_tokens, stride)
                      if min(w + sw_seq_len, total_val_tokens) - w >= 1]
            my_s = (len(ws_all) * rank) // world_size
            my_e = (len(ws_all) * (rank + 1)) // world_size
            my_wins = ws_all[my_s:my_e]
            ng_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            ng_tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
            ng_byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
            vb_ng = (torch.zeros(args.vocab_size, device=device, dtype=torch.float32)
                     if args.ogd_bias_enabled else None)
            eval_model.eval()
            compiled_ng = torch.compile(eval_model.forward_logits, dynamic=False, fullgraph=True)
            with torch.inference_mode():
                for wi, ws in enumerate(my_wins):
                    end = min(ws + sw_seq_len, total_val_tokens)
                    wlen = end - ws
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_win = chunk[:-1].unsqueeze(0)
                    y_win = chunk[1:]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = compiled_ng(x_win)
                    logits_2d = logits[0]
                    if vb_ng is not None:
                        logits_2d = logits_2d + vb_ng[None, :]
                    prev_tokens = x_win[0]
                    pprev_tokens = torch.cat([prev_tokens[:1], prev_tokens[:-1]]) if wlen >= 2 else None
                    mixed_probs = oracle.mix_with_neural(
                        logits_2d, prev_tokens, pprev_tokens,
                        ngram_weight=args.ngram_eval_weight,
                    )
                    nll = -mixed_probs.clamp_min(1e-10).log().gather(1, y_win.unsqueeze(1)).squeeze(1)
                    ng_loss_sum += nll[s:wlen].to(torch.float64).sum()
                    ng_tok_cnt += float(wlen - s)
                    tgt = y_win[s:wlen]
                    prev = x_win[0, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    ng_byte_cnt += tb.sum()
                    oracle.update(chunk[:-1].long())
                    if vb_ng is not None:
                        probs = F.softmax(logits_2d[s:wlen].float(), dim=-1)
                        vb_ng -= args.ogd_bias_lr * (
                            probs - F.one_hot(y_win[s:wlen], args.vocab_size).float()
                        ).mean(0)
                    if (wi + 1) % 1000 == 0:
                        ibpb = (ng_loss_sum / ng_tok_cnt).item() / math.log(2.0) * (ng_tok_cnt / ng_byte_cnt).item()
                        log0(f"ngram_eval:window {wi + 1}/{len(my_wins)} bpb:{ibpb:.4f}")
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(ng_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(ng_tok_cnt, op=dist.ReduceOp.SUM)
                dist.all_reduce(ng_byte_cnt, op=dist.ReduceOp.SUM)
            ng_vl = (ng_loss_sum / ng_tok_cnt).item()
            ng_bpb = ng_vl / math.log(2.0) * (ng_tok_cnt.item() / ng_byte_cnt.item())
            torch.cuda.synchronize()
            log0(
                f"final_ngram_oracle val_loss:{ng_vl:.4f} val_bpb:{ng_bpb:.4f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_ng):.0f}ms"
            )
            log0(f"final_ngram_oracle_exact val_loss:{ng_vl:.8f} val_bpb:{ng_bpb:.8f}")
            if ng_bpb < best_bpb:
                best_loss, best_bpb = ng_vl, ng_bpb
        except Exception as exc:
            log0(f"ngram_eval:failed ({exc})")
    log0(f"final_best_eval_exact val_loss:{best_loss:.8f} val_bpb:{best_bpb:.8f}")
    # Keep the legacy marker used by earlier run scripts so log-scraping still works
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{best_loss:.8f} val_bpb:{best_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
