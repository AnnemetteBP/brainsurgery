#!/usr/bin/env python
"""T3: mixed-precision sharded export of Pythia-1B.

Projection matrices -> bfloat16, everything else -> float32, attention
buffers dropped, written as a sharded safetensors checkpoint with an index.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456
N_LAYERS = 16

PROJ_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)

# Exact names, built from the layer range -- no regex that could over-match
# embeddings, layer norms or biases.
PROJECTIONS = {
    f"gpt_neox.layers.{i}.{s}" for i in range(N_LAYERS) for s in PROJ_SUFFIXES
}
BUFFERS = {
    f"gpt_neox.layers.{i}.{s}" for i in range(N_LAYERS) for s in BUFFER_SUFFIXES
}

DTYPE_BYTES = {torch.bfloat16: 2, torch.float32: 4}


def die(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> None:
    if len(PROJECTIONS) != 64:
        die(f"expected 64 projection names, built {len(PROJECTIONS)}")
    if len(BUFFERS) != 48:
        die(f"expected 48 buffer names, built {len(BUFFERS)}")

    # ---- pass 1: plan from the header only (no tensor data loaded) ----
    with safe_open(IN_PATH, framework="pt") as f:
        in_keys = list(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in in_keys}

    print(f"input: {len(in_keys)} tensors")
    if len(in_keys) != 244:
        die(f"input has {len(in_keys)} tensors, expected 244")

    in_set = set(in_keys)
    missing = (PROJECTIONS | BUFFERS) - in_set
    if missing:
        die(f"{len(missing)} expected tensors absent from input, e.g. {sorted(missing)[:3]}")

    # Keep the input order; safetensors yields keys sorted.
    out_keys = [k for k in in_keys if k not in BUFFERS]
    out_dtype = {k: (torch.bfloat16 if k in PROJECTIONS else torch.float32) for k in out_keys}
    nbytes = {
        k: DTYPE_BYTES[out_dtype[k]] * (torch.Size(shapes[k]).numel())
        for k in out_keys
    }

    # ---- required checks, before anything is written ----
    n_bf16 = sum(1 for k in out_keys if out_dtype[k] is torch.bfloat16)
    if n_bf16 != 64:
        die(f"{n_bf16} tensors would be bfloat16, expected exactly 64")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if out_dtype.get(probe) is not torch.bfloat16:
        die(f"{probe} is not bfloat16")
    if out_dtype.get("gpt_neox.embed_in.weight") is not torch.float32:
        die("gpt_neox.embed_in.weight is not float32")
    if len(out_keys) != 196:
        die(f"output would have {len(out_keys)} tensors, expected 196")
    dropped = in_set - set(out_keys)
    if dropped != BUFFERS:
        die(f"dropped set is not exactly the 48 buffers (dropped {len(dropped)})")
    print(f"checks passed: {len(out_keys)} tensors out, {n_bf16} bfloat16, {len(dropped)} dropped")

    # ---- shard assignment: greedy, in key order ----
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for k in out_keys:
        n = nbytes[k]
        if cur and cur_bytes + n > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += n
    if cur:
        shards.append(cur)

    n_shards = len(shards)
    names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
    for name, keys in zip(names, shards):
        total = sum(nbytes[k] for k in keys)
        if total > MAX_SHARD_BYTES and len(keys) != 1:
            die(f"{name}: {len(keys)} tensors totalling {total} B exceed the shard budget")
        print(f"  {name}: {len(keys):>3} tensors, {total:>12,} B")

    # ---- pass 2: cast and write, one shard at a time ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.safetensors"):
        stale.unlink()

    weight_map: dict[str, str] = {}
    with safe_open(IN_PATH, framework="pt") as f:
        for name, keys in zip(names, shards):
            tensors = {}
            for k in keys:
                t = f.get_tensor(k).to(out_dtype[k]).contiguous()
                if t.dtype is not out_dtype[k]:
                    die(f"{k}: cast to {out_dtype[k]} produced {t.dtype}")
                if tuple(t.shape) != shapes[k]:
                    die(f"{k}: shape changed to {tuple(t.shape)}")
                tensors[k] = t
            save_file(tensors, OUT_DIR / name)
            weight_map.update({k: name for k in keys})
            del tensors

    index = {
        "metadata": {"total_size": sum(nbytes.values())},
        "weight_map": weight_map,
    }
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

    # ---- verify what landed on disk ----
    seen: dict[str, torch.dtype] = {}
    for name in names:
        with safe_open(OUT_DIR / name, framework="pt") as f:
            keys = list(f.keys())
            total = 0
            for k in keys:
                sl = f.get_slice(k)
                t = f.get_tensor(k)
                seen[k] = t.dtype
                total += t.numel() * t.element_size()
                if tuple(sl.get_shape()) != shapes[k]:
                    die(f"{name}/{k}: shape on disk {tuple(sl.get_shape())} != {shapes[k]}")
            if total > MAX_SHARD_BYTES and len(keys) != 1:
                die(f"{name}: {total} B of tensor data over budget with {len(keys)} tensors")

    if set(seen) != set(out_keys):
        die("shards on disk do not hold exactly the intended key set")
    if set(weight_map) != set(out_keys):
        die("weight_map does not cover exactly the output tensors")
    n_bf16_disk = sum(1 for d in seen.values() if d is torch.bfloat16)
    n_f32_disk = sum(1 for d in seen.values() if d is torch.float32)
    if n_bf16_disk != 64:
        die(f"{n_bf16_disk} bfloat16 tensors on disk, expected 64")
    if n_f32_disk != 132:
        die(f"{n_f32_disk} float32 tensors on disk, expected 132")
    if seen[probe] is not torch.bfloat16:
        die(f"{probe} on disk is {seen[probe]}")
    if seen["gpt_neox.embed_in.weight"] is not torch.float32:
        die(f"gpt_neox.embed_in.weight on disk is {seen['gpt_neox.embed_in.weight']}")
    if len(seen) != 196:
        die(f"{len(seen)} tensors on disk, expected 196")

    print(
        f"OK: wrote {len(seen)} tensors ({n_bf16_disk} bf16 / {n_f32_disk} f32) "
        f"across {n_shards} shards + {INDEX_NAME}"
    )


if __name__ == "__main__":
    sys.exit(main())
