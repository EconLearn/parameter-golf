"""
Custom Huffman entropy coder for parameter-golf competition.

Replaces zstd/zlib compression of quantized weight tensors with per-tensor
canonical Huffman coding. The blob is self-contained: Huffman tables, tensor
shapes, dtypes, and metadata are all stored inline.

Public API:
    encode_weights(quant_result, quant_meta) -> bytes
    decode_weights(blob) -> (quant_result, quant_meta)
    measure_savings(quant_result, quant_meta) -> dict

Pure Python -- no C extensions required.

Performance strategy:
    Encoder: numpy vectorized symbol -> bit-string lookup, bulk string concat,
             then batch binary-to-bytes conversion.
    Decoder: flat lookup table (2^table_bits entries), byte-array bit window,
             one table probe per symbol.
"""

from __future__ import annotations

import heapq
import io
import json
import struct
import zlib
from typing import Any

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAGIC = b"HUFF"
_VERSION = 1
_MAX_TABLE_BITS = 16  # flat decode table size cap (2^16 = 64K entries)

_DTYPE_TO_ID: dict[str, int] = {
    "torch.int8": 0, "torch.float16": 1, "torch.float32": 2,
    "torch.bfloat16": 3, "torch.int16": 4, "torch.int32": 5,
    "torch.int64": 6, "torch.uint8": 7, "torch.bool": 8, "torch.float64": 9,
}
_ID_TO_DTYPE: dict[int, str] = {v: k for k, v in _DTYPE_TO_ID.items()}


# ===========================================================================
# Huffman tree construction and canonical coding
# ===========================================================================

def _build_huffman_lengths(freq: dict[int, int]) -> dict[int, int]:
    """Build a Huffman tree and return symbol -> code length."""
    if not freq:
        return {}
    symbols = list(freq.keys())
    if len(symbols) == 1:
        return {symbols[0]: 1}

    heap: list[tuple[int, int, Any]] = []
    ctr = 0
    for sym, f in freq.items():
        heapq.heappush(heap, (f, ctr, sym))
        ctr += 1
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, ctr, (n1, n2)))
        ctr += 1

    root = heap[0][2]
    lengths: dict[int, int] = {}
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, tuple):
            stack.append((node[0], depth + 1))
            stack.append((node[1], depth + 1))
        else:
            lengths[node] = depth

    mx = max(lengths.values())
    if mx > _MAX_TABLE_BITS:
        lengths = _limit_code_lengths(lengths, freq, _MAX_TABLE_BITS)
    return lengths


def _limit_code_lengths(lengths: dict[int, int], freq: dict[int, int],
                        max_len: int) -> dict[int, int]:
    for s in lengths:
        if lengths[s] > max_len:
            lengths[s] = max_len
    kraft = sum(1 << (max_len - lengths[s]) for s in lengths)
    target = 1 << max_len
    while kraft > target:
        sh = min(lengths, key=lambda s: (lengths[s], freq.get(s, 0)))
        if lengths[sh] < max_len:
            old = lengths[sh]
            lengths[sh] = old + 1
            kraft -= (1 << (max_len - old)) - (1 << (max_len - old - 1))
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> tuple[dict[int, tuple[int, int]], list[tuple[int, int]]]:
    """Canonical Huffman codes from lengths.
    Returns (code_table: sym->(code, len), table_spec: sorted [(sym, len)])."""
    if not lengths:
        return {}, []
    sorted_syms = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    table_spec = [(s, lengths[s]) for s in sorted_syms]
    code_table: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for sym, length in table_spec:
        if prev_len > 0:
            code = (code + 1) << (length - prev_len)
        code_table[sym] = (code, length)
        prev_len = length
    return code_table, table_spec


# ===========================================================================
# Table serialization
# ===========================================================================

def _serialize_table(table_spec: list[tuple[int, int]]) -> bytes:
    buf = struct.pack("<H", len(table_spec))
    for sym, length in table_spec:
        buf += struct.pack("<hB", sym, length)
    return buf


def _deserialize_table(data: bytes, offset: int) -> tuple[dict[int, tuple[int, int]], int]:
    n = struct.unpack_from("<H", data, offset)[0]; offset += 2
    if n == 0:
        return {}, offset
    table_spec: list[tuple[int, int]] = []
    for _ in range(n):
        sym, length = struct.unpack_from("<hB", data, offset); offset += 3
        table_spec.append((sym, length))
    ct: dict[int, tuple[int, int]] = {}
    code = 0; prev_len = 0
    for sym, length in table_spec:
        if prev_len > 0:
            code = (code + 1) << (length - prev_len)
        ct[sym] = (code, length)
        prev_len = length
    return ct, offset


# ===========================================================================
# Fast encoder
# ===========================================================================
# Strategy: for each symbol, precompute its bit-string as a Python string of
# '0'/'1' characters.  Use a numpy object array to vectorize the mapping from
# values to bit-strings.  Join them all into one big string, then convert
# 8 characters at a time into bytes.

def _fast_encode(values_np: np.ndarray,
                 code_table: dict[int, tuple[int, int]]) -> tuple[bytes, int]:
    """Encode int8 values using Huffman codes.  Returns (encoded_bytes, padding_bits)."""
    if len(values_np) == 0:
        return b"", 0

    vmin = min(code_table.keys())
    vmax = max(code_table.keys())
    span = vmax - vmin + 1

    # Build a lookup table: index (value - vmin) -> bit string
    bitstr_lut = [""] * span
    for sym, (c, l) in code_table.items():
        bitstr_lut[sym - vmin] = format(c, f"0{l}b")

    # Convert the numpy int8 array to object array of bit-strings via the LUT.
    # We use np.take on a Python list to vectorize the mapping.
    indices = (values_np.astype(np.int32) - vmin).ravel()

    # Build the full bit-string by joining LUT entries in order.
    # For speed, we process in chunks and join.
    CHUNK = 500_000
    parts = []
    for start in range(0, len(indices), CHUNK):
        chunk_idx = indices[start:start + CHUNK]
        chunk_parts = [bitstr_lut[idx] for idx in chunk_idx]
        parts.append("".join(chunk_parts))
    all_bits = "".join(parts)

    # Convert binary string to bytes
    total_bits = len(all_bits)
    padding = (8 - total_bits % 8) % 8
    if padding:
        all_bits += "0" * padding

    # Convert 8 chars at a time to bytes using int.to_bytes on bulk conversion.
    # Process in large chunks to let Python's int() do the heavy lifting.
    n_bytes = len(all_bits) // 8
    # Fast path: convert the entire binary string to a big int, then to bytes.
    # Python's int(s, 2) and int.to_bytes are implemented in C and are fast.
    out = int(all_bits, 2).to_bytes(n_bytes, "big")

    return out, padding


# ===========================================================================
# Fast decoder
# ===========================================================================
# Flat lookup table: for each possible bit pattern of `table_bits` bits,
# store (symbol, code_length).  Decode by peeking table_bits bits at the
# current position, looking up symbol and length, advancing by length bits.

def _build_flat_table(code_table: dict[int, tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (sym_table, len_table, table_bits)."""
    if not code_table:
        return np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.uint8), 0
    max_len = max(l for _, l in code_table.values())
    table_bits = min(max_len, _MAX_TABLE_BITS)
    size = 1 << table_bits
    sym_t = np.zeros(size, dtype=np.int16)
    len_t = np.zeros(size, dtype=np.uint8)
    for sym, (code, length) in code_table.items():
        if length <= table_bits:
            pad = table_bits - length
            base = code << pad
            for suffix in range(1 << pad):
                idx = base | suffix
                sym_t[idx] = sym
                len_t[idx] = length
    return sym_t, len_t, table_bits


def _fast_decode(data: bytes, padding_bits: int, num_values: int,
                 code_table: dict[int, tuple[int, int]]) -> np.ndarray:
    """Decode Huffman-coded bytes using a string-keyed lookup table.

    Converts the byte data to a binary string ('0'/'1' characters), builds a
    dict mapping every possible table_bits-wide bit-pattern string to
    (symbol, code_length), then decodes by slicing and dict lookup -- both of
    which are fast C-level operations in CPython.
    """
    if num_values == 0:
        return np.array([], dtype=np.int8)

    sym_t, len_t, table_bits = _build_flat_table(code_table)
    if table_bits == 0:
        return np.zeros(num_values, dtype=np.int8)

    # Build a string-keyed lookup: bit-pattern string -> (symbol, length).
    # This avoids the expensive int(s, 2) call per symbol in the hot loop.
    str_lut: dict[str, tuple[int, int]] = {}
    for idx in range(1 << table_bits):
        key = format(idx, f"0{table_bits}b")
        str_lut[key] = (int(sym_t[idx]), int(len_t[idx]))

    # Convert bytes to binary string
    byte_to_bits = [format(b, "08b") for b in range(256)]
    bits_str = "".join(byte_to_bits[b] for b in data)
    if padding_bits > 0:
        bits_str = bits_str[:len(bits_str) - padding_bits]
    bits_str += "0" * table_bits  # safety pad for last peek

    result = np.empty(num_values, dtype=np.int16)
    pos = 0

    for i in range(num_values):
        sym, clen = str_lut[bits_str[pos:pos + table_bits]]
        result[i] = sym
        pos += clen

    return result.astype(np.int8)


# ===========================================================================
# Tensor classification
# ===========================================================================

def _classify_tensor(name: str, tensor: Tensor) -> str:
    if tensor.dtype == torch.int8:
        vmin = tensor.min().item()
        vmax = tensor.max().item()
        return "int6" if (-31 <= vmin and vmax <= 31) else "int8"
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
        return "float"
    return "raw"


# ===========================================================================
# Encode / Decode API
# ===========================================================================

def encode_weights(quant_result: dict, quant_meta: dict) -> bytes:
    """Encode quantized weights into a self-contained Huffman-coded blob.

    Blob layout:
        [4B]  Magic "HUFF"
        [1B]  Version
        [4B]  JSON metadata length
        [NB]  JSON metadata (quant_meta + tensor descriptors)
        [MB]  Concatenated tensor payloads

    Each Huffman payload: table_len(4B) + table + num_values(4B) +
                          padding(1B) + bitstream_len(4B) + bitstream
    Each raw payload: raw numpy bytes
    """
    output = io.BytesIO()
    descriptors: list[dict] = []
    payloads: list[bytes] = []

    for name in sorted(quant_result.keys()):
        tensor = quant_result[name]
        if not isinstance(tensor, Tensor):
            continue
        tensor = tensor.contiguous()
        cat = _classify_tensor(name, tensor)
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
            buf.write(struct.pack("<I", len(tb)))
            buf.write(tb)
            buf.write(struct.pack("<I", len(vals)))
            buf.write(struct.pack("<B", pad_bits))
            buf.write(struct.pack("<I", len(enc_bytes)))
            buf.write(enc_bytes)
            payload = buf.getvalue()
            encoding = f"huffman_{cat}"
        else:
            if tensor.dtype == torch.bfloat16:
                payload = tensor.view(torch.int16).numpy().tobytes()
            else:
                payload = tensor.numpy().tobytes()
            encoding = "raw"

        descriptors.append({
            "name": name, "shape": shape, "dtype": dtype_id,
            "encoding": encoding, "payload_size": len(payload),
        })
        payloads.append(payload)

    meta_bytes = json.dumps(
        {"quant_meta": _meta_to_json(quant_meta), "tensors": descriptors},
        separators=(",", ":"),
    ).encode("utf-8")

    output.write(_MAGIC)
    output.write(struct.pack("<B", _VERSION))
    output.write(struct.pack("<I", len(meta_bytes)))
    output.write(meta_bytes)
    for p in payloads:
        output.write(p)
    return output.getvalue()


def decode_weights(blob: bytes) -> tuple[dict, dict]:
    """Decode a Huffman-coded weight blob back to (quant_result, quant_meta)."""
    off = 0
    if blob[off:off + 4] != _MAGIC:
        raise ValueError("Bad magic")
    off += 4
    ver = struct.unpack_from("<B", blob, off)[0]; off += 1
    if ver != _VERSION:
        raise ValueError(f"Unsupported version {ver}")

    meta_len = struct.unpack_from("<I", blob, off)[0]; off += 4
    meta = json.loads(blob[off:off + meta_len])
    off += meta_len

    quant_meta = _meta_from_json(meta["quant_meta"])
    result: dict[str, Tensor] = {}

    for desc in meta["tensors"]:
        name = desc["name"]
        shape = desc["shape"]
        dtype_id = desc["dtype"]
        encoding = desc["encoding"]
        ps = desc["payload_size"]
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
                np_dt = {
                    torch.float16: np.float16, torch.float32: np.float32,
                    torch.float64: np.float64, torch.int8: np.int8,
                    torch.int16: np.int16, torch.int32: np.int32,
                    torch.int64: np.int64, torch.uint8: np.uint8,
                }.get(dtype, np.float32)
                arr = np.frombuffer(payload, dtype=np_dt).copy()
                tensor = torch.from_numpy(arr).reshape(shape)
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)

        result[name] = tensor.contiguous()

    return result, quant_meta


# ===========================================================================
# Metadata helpers
# ===========================================================================

def _meta_to_json(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            out[k] = _meta_to_json(v)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out

def _meta_from_json(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        out[k] = _meta_from_json(v) if isinstance(v, dict) else v
    return out


# ===========================================================================
# Measurement
# ===========================================================================

def measure_savings(quant_result: dict, quant_meta: dict) -> dict:
    """Compare Huffman coding vs zstd-22 vs zlib-9."""
    huff_blob = encode_weights(quant_result, quant_meta)
    huff_size = len(huff_blob)

    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    raw = buf.getvalue()
    raw_size = len(raw)

    zlib_blob = zlib.compress(raw, 9)
    zlib_size = len(zlib_blob)

    zstd_size = None
    try:
        import zstandard
        zstd_size = len(zstandard.ZstdCompressor(level=22).compress(raw))
    except ImportError:
        pass

    tensor_bytes = sum(
        t.numel() * t.element_size()
        for t in quant_result.values() if isinstance(t, Tensor)
    )

    stats = {"int6": [0, 0], "int8": [0, 0], "float": [0, 0]}
    for n, t in quant_result.items():
        if isinstance(t, Tensor):
            c = _classify_tensor(n, t)
            if c in stats:
                stats[c][0] += 1
                stats[c][1] += t.numel() * t.element_size()

    res = dict(raw_tensor_bytes=tensor_bytes, torch_raw=raw_size,
               zlib9=zlib_size, zstd22=zstd_size, huffman=huff_size)

    print("=" * 70)
    print("ENTROPY CODING SIZE COMPARISON")
    print("=" * 70)
    for c in ("int6", "int8", "float"):
        print(f"  {c:18s} {stats[c][0]:4d} tensors  {stats[c][1]:>12,} bytes")
    print(f"  {'Raw tensor data':18s} {tensor_bytes:>12,} bytes")
    print(f"  {'torch.save':18s} {raw_size:>12,} bytes")
    print("-" * 70)
    print(f"  zlib-9:    {zlib_size:>12,} bytes  ({100*zlib_size/tensor_bytes:.1f}% of raw)")
    if zstd_size is not None:
        print(f"  zstd-22:   {zstd_size:>12,} bytes  ({100*zstd_size/tensor_bytes:.1f}% of raw)")
    else:
        print(f"  zstd-22:   (not installed)")
    print(f"  Huffman:   {huff_size:>12,} bytes  ({100*huff_size/tensor_bytes:.1f}% of raw)")
    print("-" * 70)
    if zstd_size is not None:
        d = zstd_size - huff_size
        print(f"  vs zstd-22:  {d:>+,} bytes ({abs(100*d/zstd_size):.2f}% {'saved' if d>0 else 'larger'})")
    d = zlib_size - huff_size
    print(f"  vs zlib-9:   {d:>+,} bytes ({abs(100*d/zlib_size):.2f}% {'saved' if d>0 else 'larger'})")
    print("=" * 70)
    return res


# ===========================================================================
# Test
# ===========================================================================

def _test():
    import time

    print("=" * 70)
    print("ENTROPY CODER TEST SUITE")
    print("=" * 70)

    np.random.seed(42); torch.manual_seed(42)

    model_dim = 512; num_layers = 11; num_heads = 8; num_kv_heads = 4
    mlp_mult = 3; vocab_size = 1024

    qr: dict[str, Tensor] = {}
    qm: dict[str, Any] = {}

    def add_int6(name: str, shape: tuple) -> None:
        raw = np.random.laplace(0, 3.5, shape).astype(np.float32)
        qr[name + ".q"] = torch.from_numpy(np.clip(np.round(raw), -31, 31).astype(np.int8))
        qr[name + ".scale"] = torch.from_numpy(
            np.abs(np.random.normal(0.01, 0.003, shape[0])).astype(np.float16))
        qm[name] = {"type": "int6"}

    def add_int8(name: str, shape: tuple) -> None:
        raw = np.random.laplace(0, 20, shape).astype(np.float32)
        qr[name + ".q"] = torch.from_numpy(np.clip(np.round(raw), -127, 127).astype(np.int8))
        qr[name + ".scale"] = torch.from_numpy(
            np.abs(np.random.normal(0.005, 0.001, shape[0])).astype(np.float16))
        qm[name] = {"type": "int8"}

    def add_pt(name: str, shape: tuple, dt: str = "float16") -> None:
        t = torch.randn(shape)
        qr[name] = t.half() if dt == "float16" else t.float()
        qm[name] = "passthrough" if dt == "float16" else "passthrough_ctrl"

    # Build model
    add_int8("tok_emb.weight", (vocab_size, model_dim))
    hd = model_dim // num_heads; kv = num_kv_heads * hd; mh = model_dim * mlp_mult
    for i in range(num_layers):
        p = f"blocks.{i}"
        add_int6(f"{p}.attn.q_proj.weight", (model_dim, model_dim))
        add_int6(f"{p}.attn.k_proj.weight", (kv, model_dim))
        add_int6(f"{p}.attn.v_proj.weight", (kv, model_dim))
        add_int6(f"{p}.attn.o_proj.weight", (model_dim, model_dim))
        add_int6(f"{p}.mlp.up_proj.weight", (mh, model_dim))
        add_int6(f"{p}.mlp.gate_proj.weight", (mh, model_dim))
        add_int6(f"{p}.mlp.down_proj.weight", (model_dim, mh))
        add_pt(f"{p}.attn_scale", (1,), "float32")
        add_pt(f"{p}.mlp_scale", (1,), "float32")
    add_pt("final_norm.weight", (model_dim,), "float16")

    tp = sum(t.numel() for t in qr.values())
    tb = sum(t.numel() * t.element_size() for t in qr.values())
    ni6 = sum(1 for n, t in qr.items() if _classify_tensor(n, t) == "int6")
    ni8 = sum(1 for n, t in qr.items() if _classify_tensor(n, t) == "int8")
    nf = sum(1 for n, t in qr.items() if _classify_tensor(n, t) == "float")
    print(f"\nMock model: {num_layers}L dim={model_dim} | "
          f"{tp:,} params | {tb:,} raw bytes")
    print(f"  {ni6} int6, {ni8} int8, {nf} float tensors\n")

    # Encode
    print("Encoding...")
    t0 = time.perf_counter()
    blob = encode_weights(qr, qm)
    print(f"  {len(blob):,} bytes in {1000*(time.perf_counter()-t0):.0f} ms")

    # Decode
    print("Decoding...")
    t0 = time.perf_counter()
    dr, dm = decode_weights(blob)
    print(f"  decoded in {1000*(time.perf_counter()-t0):.0f} ms")

    # Verify
    print("\nRoundtrip verification...")
    assert set(dm.keys()) == set(qm.keys()), "meta key mismatch"
    for k in qm:
        assert str(dm[k]) == str(qm[k]), f"meta mismatch: {k}"
    assert set(dr.keys()) == set(qr.keys()), "tensor key mismatch"
    mfe = 0.0
    for name in sorted(qr.keys()):
        o, d = qr[name], dr[name]
        assert o.shape == d.shape, f"shape: {name}"
        assert o.dtype == d.dtype, f"dtype: {name}"
        if o.is_floating_point():
            e = (o.float() - d.float()).abs().max().item()
            mfe = max(mfe, e)
            assert e < 1e-3, f"float err {name}: {e}"
        else:
            assert torch.equal(o, d), f"int mismatch: {name}"
    print(f"  {len(qr)} tensors OK (max float err: {mfe:.2e})")
    print("  ROUNDTRIP PASSED")

    # Size comparison
    print()
    measure_savings(qr, qm)

    # Edge cases
    print("\n--- Edge cases ---")

    print("  All-zero tensor...", end=" ")
    d1r, _ = decode_weights(encode_weights({"x": torch.zeros(10, dtype=torch.int8)}, {"i": "t"}))
    assert torch.equal(d1r["x"], torch.zeros(10, dtype=torch.int8)); print("OK")

    print("  Uniform tensor...", end=" ")
    u = torch.full((1000,), 5, dtype=torch.int8)
    b2 = encode_weights({"y": u}, {"i": "u"})
    d2r, _ = decode_weights(b2)
    assert torch.equal(d2r["y"], u); print(f"OK ({len(b2)} bytes)")

    print("  Mixed dtypes...", end=" ")
    mr = {"a": torch.tensor([1,-1,0,31,-31], dtype=torch.int8),
          "b": torch.tensor([1.5,-2.5,0.0], dtype=torch.float16),
          "c": torch.tensor([1e-6,3.14], dtype=torch.float32)}
    d3r, _ = decode_weights(encode_weights(mr, {"i": "m"}))
    for k in mr:
        if mr[k].dtype == torch.int8:
            assert torch.equal(d3r[k], mr[k])
        else:
            assert torch.allclose(d3r[k], mr[k], atol=1e-3)
    print("OK")

    print("  Large 1M-value tensor...", end=" ")
    big = np.clip(np.round(np.random.laplace(0, 3.5, 1_000_000)), -31, 31).astype(np.int8)
    bt = torch.from_numpy(big).reshape(1000, 1000)
    t0 = time.perf_counter()
    b4 = encode_weights({"big": bt}, {"i": "s"})
    et = time.perf_counter() - t0
    t0 = time.perf_counter()
    d4r, _ = decode_weights(b4)
    dt = time.perf_counter() - t0
    assert torch.equal(d4r["big"], bt)
    print(f"OK (enc {et:.2f}s dec {dt:.2f}s | "
          f"1,000,000B -> {len(b4):,}B = {100*len(b4)/1_000_000:.1f}%)")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _test()
