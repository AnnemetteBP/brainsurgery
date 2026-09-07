"""T3: mixed-precision export of GPT-2 (124M) with sharding.

- cast exactly the 48 projection matrices to bfloat16
- keep every other parameter float32 with unchanged values
- drop the 12 causal-mask buffers h.<i>.attn.bias
- write out/T3/ as sharded safetensors (<= 64 MiB of tensor data per shard,
  oversized tensors alone) plus model.safetensors.index.json
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
OUT_DIR = os.path.join(ROOT, "out", "T3")

NUM_LAYERS = 12
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67,108,864


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


# ---- explicit targets: no regex, no chance of overmatching -------------------
CAST_KEYS = set()
for i in range(NUM_LAYERS):
    CAST_KEYS.update(
        {
            f"h.{i}.attn.c_attn.weight",
            f"h.{i}.attn.c_proj.weight",
            f"h.{i}.mlp.c_fc.weight",
            f"h.{i}.mlp.c_proj.weight",
        }
    )
DROP_KEYS = {f"h.{i}.attn.bias" for i in range(NUM_LAYERS)}
EXPECTED_CAST_SHAPES = {
    "attn.c_attn.weight": (768, 2304),
    "attn.c_proj.weight": (768, 768),
    "mlp.c_fc.weight": (768, 3072),
    "mlp.c_proj.weight": (3072, 768),
}

# ---- load -------------------------------------------------------------------
check(os.path.isfile(SRC), f"input not found: {SRC}")
src = {}
with safe_open(SRC, framework="pt", device="cpu") as f:
    for k in f.keys():
        src[k] = f.get_tensor(k)

check(len(src) == 160, f"expected 160 input tensors, found {len(src)}")
check(all(t.dtype == torch.float32 for t in src.values()), "input is not all float32")
missing_cast = CAST_KEYS - src.keys()
check(not missing_cast, f"projection matrices missing from input: {sorted(missing_cast)}")
missing_drop = DROP_KEYS - src.keys()
check(not missing_drop, f"buffers missing from input: {sorted(missing_drop)}")
for k in CAST_KEYS:
    suffix = re.sub(r"^h\.\d+\.", "", k)
    check(
        tuple(src[k].shape) == EXPECTED_CAST_SHAPES[suffix],
        f"{k}: unexpected shape {tuple(src[k].shape)}",
    )
for k in DROP_KEYS:
    check(
        tuple(src[k].shape) == (1, 1, 1024, 1024),
        f"{k}: unexpected buffer shape {tuple(src[k].shape)}, refusing to delete",
    )

# ---- transform --------------------------------------------------------------
out = {}
for k, t in src.items():  # safetensors yields keys in sorted order
    if k in DROP_KEYS:
        continue
    if k in CAST_KEYS:
        out[k] = t.to(torch.bfloat16).contiguous()
    else:
        out[k] = t.contiguous()  # untouched float32

# ---- required checks (before writing) ---------------------------------------
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
check(n_bf16 == 48, f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
check(out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, "h.0.attn.c_attn.weight is not bfloat16")
check(out["wte.weight"].dtype == torch.float32, "wte.weight is not float32")
check(len(out) == 148, f"expected 148 output tensors, got {len(out)}")

# extra sanity: bf16 set is exactly CAST_KEYS, everything else fp32 and bit-identical
bf16_keys = {k for k, t in out.items() if t.dtype == torch.bfloat16}
check(bf16_keys == CAST_KEYS, f"bf16 set differs from intended: {sorted(bf16_keys ^ CAST_KEYS)}")
for k, t in out.items():
    if k not in CAST_KEYS:
        check(t.dtype == torch.float32, f"{k} should be float32, is {t.dtype}")
        check(torch.equal(t, src[k]), f"{k}: values changed")
check(set(out) == set(src) - DROP_KEYS, "output key set is not input minus buffers")
check(not any(k in out for k in DROP_KEYS), "a buffer survived")

# ---- shard ------------------------------------------------------------------
def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


shards: list[list[str]] = []
cur: list[str] = []
cur_size = 0
for k, t in out.items():
    sz = nbytes(t)
    if sz > MAX_SHARD_BYTES:
        # oversized tensor lives alone in its own shard
        if cur:
            shards.append(cur)
            cur, cur_size = [], 0
        shards.append([k])
        continue
    if cur and cur_size + sz > MAX_SHARD_BYTES:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(k)
    cur_size += sz
if cur:
    shards.append(cur)

n_shards = len(shards)
shard_names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]

# verify sharding invariants before writing
weight_map = {}
for name, keys in zip(shard_names, shards):
    total = sum(nbytes(out[k]) for k in keys)
    if len(keys) == 1 and nbytes(out[keys[0]]) > MAX_SHARD_BYTES:
        pass  # allowed: single oversized tensor
    else:
        check(total <= MAX_SHARD_BYTES, f"shard {name} holds {total} bytes > {MAX_SHARD_BYTES}")
    for k in keys:
        check(k not in weight_map, f"{k} assigned to two shards")
        weight_map[k] = name
check(len(weight_map) == 148, f"weight_map has {len(weight_map)} entries, expected 148")
check(set(weight_map) == set(out), "weight_map key set differs from output tensors")

# ---- write ------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for name, keys in zip(shard_names, shards):
    path = os.path.join(OUT_DIR, name)
    check(not os.path.exists(path), f"refusing to overwrite existing shard: {path}")
for name, keys in zip(shard_names, shards):
    save_file({k: out[k] for k in keys}, os.path.join(OUT_DIR, name), metadata={"format": "pt"})

index = {
    "metadata": {"total_size": sum(nbytes(t) for t in out.values())},
    "weight_map": weight_map,
}
index_path = os.path.join(OUT_DIR, "model.safetensors.index.json")
check(not os.path.exists(index_path), f"refusing to overwrite existing index: {index_path}")
with open(index_path, "w") as fh:
    json.dump(index, fh, indent=2, sort_keys=True)
    fh.write("\n")

# ---- verify from disk -------------------------------------------------------
seen = {}
for name in shard_names:
    with safe_open(os.path.join(OUT_DIR, name), framework="pt", device="cpu") as f:
        for k in f.keys():
            check(k not in seen, f"{k} duplicated on disk")
            t = f.get_tensor(k)
            seen[k] = t
            check(weight_map[k] == name, f"{k}: index says {weight_map[k]}, found in {name}")
check(len(seen) == 148, f"{len(seen)} tensors on disk, expected 148")
for k, t in out.items():
    check(t.dtype == seen[k].dtype, f"{k}: dtype mismatch on disk")
    check(torch.equal(t, seen[k]), f"{k}: values differ on disk")

print(f"OK: wrote {n_shards} shards + index to {OUT_DIR}")
for name, keys in zip(shard_names, shards):
    print(f"  {name}: {len(keys)} tensors, {sum(nbytes(out[k]) for k in keys):,} bytes")
