"""T3: mixed-precision sharded export of Pythia-1B.

Casts the 64 projection matrices to bfloat16, upcasts everything else to
float32, drops the 48 non-parameter attention buffers, and writes a sharded
safetensors checkpoint with an index file.
"""

import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
SHARD_LIMIT = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(?:attention\.(?:query_key_value|dense)|mlp\.(?:dense_h_to_4h|dense_4h_to_h))"
    r"\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(?:bias|masked_bias|rotary_emb\.inv_freq)$"
)

ITEMSIZE = {torch.bfloat16: 2, torch.float32: 4}


def main() -> None:
    with safe_open(IN_PATH, framework="pt") as f:
        all_keys = list(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in all_keys}

    print(f"input tensors: {len(all_keys)}")

    dropped = [k for k in all_keys if BUFFER_RE.match(k)]
    kept = [k for k in all_keys if not BUFFER_RE.match(k)]
    target_dtype = {k: (torch.bfloat16 if PROJ_RE.match(k) else torch.float32) for k in kept}

    # --- Required checks, before writing anything ---
    n_bf16 = sum(1 for k in kept if target_dtype[k] is torch.bfloat16)
    if n_bf16 != 64:
        raise SystemExit(f"CHECK FAILED: expected 64 bfloat16 tensors, got {n_bf16}")
    qkv = "gpt_neox.layers.0.attention.query_key_value.weight"
    if target_dtype.get(qkv) is not torch.bfloat16:
        raise SystemExit(f"CHECK FAILED: {qkv} is not bfloat16")
    if target_dtype.get("gpt_neox.embed_in.weight") is not torch.float32:
        raise SystemExit("CHECK FAILED: gpt_neox.embed_in.weight is not float32")
    if len(kept) != 196:
        raise SystemExit(f"CHECK FAILED: expected 196 output tensors, got {len(kept)}")
    if len(dropped) != 48:
        raise SystemExit(f"CHECK FAILED: expected to drop 48 buffers, got {len(dropped)}")

    # --- Plan the shards: greedy pack in key order, oversized tensors stand alone ---
    def nbytes(k: str) -> int:
        n = 1
        for d in shapes[k]:
            n *= d
        return n * ITEMSIZE[target_dtype[k]]

    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for k in kept:
        size = nbytes(k)
        if size > SHARD_LIMIT:  # too big for any shard: gets its own
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([k])
            continue
        if cur and cur_bytes + size > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += size
    if cur:
        shards.append(cur)

    total = len(shards)
    names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]
    for name, shard in zip(names, shards):
        data = sum(nbytes(k) for k in shard)
        if data > SHARD_LIMIT and len(shard) > 1:
            raise SystemExit(f"CHECK FAILED: shard {name} holds {data} bytes with {len(shard)} tensors")

    # --- Write ---
    if os.path.isdir(OUT_DIR):
        for entry in os.listdir(OUT_DIR):
            p = os.path.join(OUT_DIR, entry)
            if entry == "solution.py" or entry == "REPORT.md":
                continue
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(OUT_DIR, exist_ok=True)

    weight_map: dict[str, str] = {}
    total_size = 0
    with safe_open(IN_PATH, framework="pt") as f:
        for name, shard in zip(names, shards):
            tensors = {}
            for k in shard:
                t = f.get_tensor(k)
                if t.dtype is not torch.float16:
                    raise SystemExit(f"CHECK FAILED: {k} has unexpected input dtype {t.dtype}")
                tensors[k] = t.to(torch.float32).to(target_dtype[k]).contiguous()
                weight_map[k] = name
                total_size += tensors[k].numel() * tensors[k].element_size()
            save_file(tensors, os.path.join(OUT_DIR, name), metadata={"format": "pt"})
            del tensors
            print(f"wrote {name}: {len(shard)} tensors")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    # --- Verify what landed on disk ---
    seen: dict[str, torch.dtype] = {}
    for name in names:
        path = os.path.join(OUT_DIR, name)
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            data = 0
            for k in keys:
                t = f.get_slice(k)
                n = 1
                for d in t.get_shape():
                    n *= d
                dt = torch.bfloat16 if t.get_dtype() == "BF16" else torch.float32
                if t.get_dtype() not in ("BF16", "F32"):
                    raise SystemExit(f"CHECK FAILED: {k} written as {t.get_dtype()}")
                seen[k] = dt
                data += n * ITEMSIZE[dt]
            if data > SHARD_LIMIT and len(keys) > 1:
                raise SystemExit(f"CHECK FAILED: {name} exceeds shard budget with {len(keys)} tensors")

    if len(seen) != 196:
        raise SystemExit(f"CHECK FAILED: output has {len(seen)} tensors, expected 196")
    written_bf16 = sum(1 for d in seen.values() if d is torch.bfloat16)
    if written_bf16 != 64:
        raise SystemExit(f"CHECK FAILED: output has {written_bf16} bfloat16 tensors, expected 64")
    if seen[qkv] is not torch.bfloat16:
        raise SystemExit(f"CHECK FAILED: {qkv} written as {seen[qkv]}")
    if seen["gpt_neox.embed_in.weight"] is not torch.float32:
        raise SystemExit("CHECK FAILED: gpt_neox.embed_in.weight written as non-float32")
    if set(weight_map) != set(seen):
        raise SystemExit("CHECK FAILED: index weight_map does not match written tensors")

    print(f"OK: {len(seen)} tensors, {written_bf16} bfloat16, {total} shards, {total_size} bytes")


if __name__ == "__main__":
    main()
