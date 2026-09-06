"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Casts exactly the 112 per-layer projection matrices to bfloat16, keeps every
other tensor float32 with unchanged values, and writes a sharded safetensors
checkpoint (<= 256 MiB of tensor data per shard, oversized tensors alone)
with a model.safetensors.index.json.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IN_DIR = os.path.join(ROOT, "inputs", "base")
OUT_DIR = os.path.join(ROOT, "out", "T3")

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456
NUM_LAYERS = 16
PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_input():
    index_path = os.path.join(IN_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(IN_DIR, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in tensors:
                    fail(f"duplicate tensor {name} across input shards")
                tensors[name] = f.get_tensor(name)
    missing = set(weight_map) - set(tensors)
    if missing:
        fail(f"index lists tensors not found in shards: {sorted(missing)[:5]}")
    return tensors


def main():
    src = load_input()
    if len(src) != 114:
        fail(f"expected 114 input tensors, got {len(src)}")

    out = {}
    cast_names = []
    for name, t in src.items():
        if t.dtype != torch.float32:
            fail(f"input tensor {name} is {t.dtype}, expected float32")
        m = PROJ_RE.match(name)
        if m and int(m.group(1)) < NUM_LAYERS:
            out[name] = t.to(torch.bfloat16).contiguous()
            cast_names.append(name)
        else:
            out[name] = t.contiguous()

    # Sanity: exactly the expected set of projection names was targeted.
    expected = {
        f"model.layers.{i}.{p}.weight"
        for i in range(NUM_LAYERS)
        for p in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    }
    if set(cast_names) != expected:
        fail(f"cast set mismatch: extra={sorted(set(cast_names) - expected)[:5]} "
             f"missing={sorted(expected - set(cast_names))[:5]}")

    # Required checks (before writing).
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 112:
        fail(f"expected exactly 112 bfloat16 tensors, got {n_bf16}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        fail("model.embed_tokens.weight is not float32")
    if len(out) != 114:
        fail(f"expected exactly 114 output tensors, got {len(out)}")
    if set(out) != set(src):
        fail("output key set differs from input key set")
    for name, t in out.items():
        if t.dtype == torch.float32 and not torch.equal(t, src[name]):
            fail(f"float32 tensor {name} changed value")
        if t.shape != src[name].shape:
            fail(f"shape changed for {name}")

    # Plan shards: greedy in name order; a tensor that alone exceeds the budget
    # goes in its own shard.
    def nbytes(t):
        return t.numel() * t.element_size()

    shards = []  # list of lists of names
    cur, cur_size = [], 0
    for name in sorted(out):
        sz = nbytes(out[name])
        if sz > MAX_SHARD_BYTES:
            if cur:
                shards.append(cur)
                cur, cur_size = [], 0
            shards.append([name])
            continue
        if cur and cur_size + sz > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += sz
    if cur:
        shards.append(cur)

    for names in shards:
        total = sum(nbytes(out[n]) for n in names)
        if total > MAX_SHARD_BYTES and len(names) != 1:
            fail(f"shard exceeds budget with {len(names)} tensors: {total} bytes")

    os.makedirs(OUT_DIR, exist_ok=True)
    for existing in os.listdir(OUT_DIR):
        if existing.endswith(".safetensors") or existing == "model.safetensors.index.json":
            fail(f"output already exists: {os.path.join(OUT_DIR, existing)}")

    n = len(shards)
    weight_map = {}
    total_size = 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: out[k] for k in names}, os.path.join(OUT_DIR, fname),
                  metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
            total_size += nbytes(out[k])
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map},
                  f, indent=2, sort_keys=True)

    # Post-write verification.
    seen = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                seen[k] = f.get_tensor(k)
    if len(seen) != 114 or set(seen) != set(out):
        fail("written checkpoint does not contain exactly the 114 expected tensors")
    for k, t in seen.items():
        if t.dtype != out[k].dtype or not torch.equal(t, out[k]):
            fail(f"written tensor {k} differs from planned output")

    print(f"wrote {len(weight_map)} tensors in {n} shards to {OUT_DIR}")
    for i, names in enumerate(shards, start=1):
        print(f"  shard {i}: {len(names)} tensors, {sum(nbytes(out[k]) for k in names)} bytes")
    print(f"  bfloat16: {n_bf16}, float32: {114 - n_bf16}, total_size: {total_size}")


if __name__ == "__main__":
    main()
