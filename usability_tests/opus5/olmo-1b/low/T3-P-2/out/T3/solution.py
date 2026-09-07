"""T3: mixed-precision (bf16 projections) sharded export of OLMo-1B-0724-hf."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
SHARD_LIMIT = 256 * 1024 * 1024

PROJ = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_state_dict():
    index = json.load(open(os.path.join(IN_DIR, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(IN_DIR, shard), framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    assert set(tensors) == set(weight_map), "loaded keys differ from the index"
    return tensors


def main():
    src = load_state_dict()
    out = {}
    for name in sorted(src):
        t = src[name]
        out[name] = t.to(torch.bfloat16) if PROJ.match(name) else t.to(torch.float32)

    # Required checks, before writing anything.
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 112, f"expected 112 bfloat16 tensors, got {n_bf16}"
    assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out["model.embed_tokens.weight"].dtype == torch.float32
    assert len(out) == 114, f"expected 114 tensors, got {len(out)}"
    for name, t in out.items():
        if t.dtype is not torch.bfloat16:
            assert t.dtype == torch.float32, f"{name} is {t.dtype}"

    # Greedy sharding: a tensor larger than the budget gets its own shard.
    shards, cur, cur_bytes = [], {}, 0
    for name, t in out.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[name] = t
        cur_bytes += nbytes
        if cur_bytes >= SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = {}, 0
    if cur:
        shards.append(cur)

    total = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in shard.items()}, os.path.join(OUT_DIR, fname))
        for k in shard:
            weight_map[k] = fname
        size = sum(v.numel() * v.element_size() for v in shard.values())
        assert size <= SHARD_LIMIT or len(shard) == 1, f"{fname} over budget with {len(shard)} tensors"
        print(f"{fname}: {len(shard)} tensors, {size} bytes")

    assert set(weight_map) == set(out), "weight_map does not cover every tensor"
    total_size = sum(t.numel() * t.element_size() for t in out.values())
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    print(f"wrote {len(out)} tensors in {total} shards, {total_size} bytes")


if __name__ == "__main__":
    main()
