"""T3: mixed-precision sharded export of Pythia-1B.

- 64 projection matrices -> bfloat16
- all other parameters -> float32
- 48 non-parameter buffers dropped
- sharded safetensors output (<= 256 MiB of tensor data per shard) + index
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
OUT_DIR = os.path.join(SANDBOX, "out", "T3")
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456
N_LAYERS = 16

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\."
    r"(attention\.query_key_value|attention\.dense|mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)"
    r"\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # ---- load ------------------------------------------------------------
    state = {}
    with safe_open(SRC, framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    if len(state) != 244:
        fail(f"expected 244 input tensors, got {len(state)}")

    # ---- classify --------------------------------------------------------
    proj_keys = [k for k in state if PROJ_RE.match(k)]
    buf_keys = [k for k in state if BUFFER_RE.match(k)]
    if len(proj_keys) != 64:
        fail(f"expected 64 projection matrices, matched {len(proj_keys)}")
    if len(buf_keys) != 48:
        fail(f"expected 48 buffers, matched {len(buf_keys)}")
    if set(proj_keys) & set(buf_keys):
        fail("projection / buffer sets overlap")
    layers_seen = {int(PROJ_RE.match(k).group(1)) for k in proj_keys}
    if layers_seen != set(range(N_LAYERS)):
        fail(f"projection matrices do not cover layers 0..{N_LAYERS - 1}: {sorted(layers_seen)}")

    # ---- transform (preserve original key order) -------------------------
    out = {}
    for k, t in state.items():
        if k in buf_keys:
            continue
        if k in proj_keys:
            out[k] = t.to(torch.bfloat16).contiguous()
        else:
            out[k] = t.to(torch.float32).contiguous()

    # ---- required checks -------------------------------------------------
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 64:
        fail(f"expected exactly 64 bfloat16 tensors, got {n_bf16}")
    if out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
        fail("layers.0 query_key_value.weight is not bfloat16")
    if out["gpt_neox.embed_in.weight"].dtype != torch.float32:
        fail("embed_in.weight is not float32")
    if len(out) != 196:
        fail(f"expected 196 output tensors, got {len(out)}")
    n_f32 = sum(1 for t in out.values() if t.dtype == torch.float32)
    if n_f32 + n_bf16 != len(out):
        fail("some output tensor is neither bfloat16 nor float32")
    for k, t in out.items():
        if tuple(t.shape) != tuple(state[k].shape):
            fail(f"shape changed for {k}")
    for k in ("gpt_neox.embed_in.weight", "embed_out.weight"):
        if k not in out:
            fail(f"missing parameter {k}")

    # ---- shard (greedy, in key order; oversized tensors alone) ----------
    def nbytes(t):
        return t.numel() * t.element_size()

    shards = []  # list of lists of keys
    cur, cur_size = [], 0
    for k, t in out.items():
        sz = nbytes(t)
        if sz > MAX_SHARD_BYTES:
            shards.append([k])
            continue
        if cur and cur_size + sz > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(k)
        cur_size += sz
    if cur:
        shards.append(cur)

    # verify sharding rules before writing
    for keys in shards:
        total = sum(nbytes(out[k]) for k in keys)
        if total > MAX_SHARD_BYTES:
            if len(keys) != 1:
                fail(f"shard exceeds budget and holds {len(keys)} tensors")
    if sum(len(s) for s in shards) != 196:
        fail("shard assignment does not cover all tensors")

    # ---- write -----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    for fn in os.listdir(OUT_DIR):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            fail(f"output already exists: {fn}")

    n = len(shards)
    weight_map = {}
    total_size = 0
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        shard_dict = {k: out[k] for k in keys}
        save_file(shard_dict, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname
            total_size += nbytes(out[k])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    # ---- post-write verification -----------------------------------------
    seen = 0
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt") as f:
            keys = list(f.keys())
            seen += len(keys)
            for k in keys:
                if weight_map[k] != fname:
                    fail(f"index mismatch for {k}")
                t = f.get_tensor(k)
                if t.dtype != out[k].dtype or tuple(t.shape) != tuple(out[k].shape):
                    fail(f"re-read mismatch for {k}")
    if seen != 196:
        fail(f"re-read {seen} tensors, expected 196")

    print(f"OK: wrote {n} shards, {len(weight_map)} tensors, {total_size} bytes to {OUT_DIR}")
    for i, keys in enumerate(shards, start=1):
        print(f"  shard {i}: {len(keys)} tensors, {sum(nbytes(out[k]) for k in keys)} bytes")


if __name__ == "__main__":
    main()
