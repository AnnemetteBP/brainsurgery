"""T3: mixed-precision sharded export of OLMo-1B-0724-hf.

Plain torch + safetensors script: the task is a per-tensor dtype cast plus a
sharding policy, both of which are exactly expressible here without the
guesswork of a merge/export toolkit.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
SHARD_BUDGET = 256 * 1024 * 1024  # 256 MiB of tensor data per shard

# Exactly the projection matrices, anchored: no embeddings, norms or biases.
PROJ = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_state_dict():
    index = json.load(open(os.path.join(IN_DIR, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    order = list(weight_map)
    handles = {}
    tensors = {}
    for name in order:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(IN_DIR, shard), framework="pt")
        tensors[name] = handles[shard].get_tensor(name)
    return order, tensors


def plan_shards(order, tensors):
    """Greedy sequential packing; a tensor over budget gets its own shard."""
    shards, cur, cur_bytes = [], [], 0
    for name in order:
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if nbytes > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([name])
            continue
        if cur and cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)
    return shards


def main():
    order, tensors = load_state_dict()

    out = {}
    for name in order:
        t = tensors[name]
        out[name] = t.to(torch.bfloat16) if PROJ.match(name) else t.to(torch.float32)

    # Required checks: fail loudly before writing anything.
    bf16 = [n for n, t in out.items() if t.dtype is torch.bfloat16]
    assert len(bf16) == 112, f"expected 112 bfloat16 tensors, got {len(bf16)}"
    assert out["model.layers.0.self_attn.q_proj.weight"].dtype is torch.bfloat16
    assert out["model.embed_tokens.weight"].dtype is torch.float32
    assert len(out) == 114, f"expected 114 tensors, got {len(out)}"
    assert set(out) == set(tensors), "tensor names changed"
    for n, t in out.items():
        if t.dtype is torch.float32:
            assert torch.equal(t, tensors[n]), f"value changed for {n}"

    shards = plan_shards(order, out)
    n = len(shards)
    weight_map, total = {}, 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, names in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(out[x].numel() * out[x].element_size() for x in names)
        assert size <= SHARD_BUDGET or len(names) == 1, f"{fn} over budget"
        total += size
        save_file({x: out[x].contiguous() for x in names}, os.path.join(OUT_DIR, fn),
                  metadata={"format": "pt"})
        for x in names:
            weight_map[x] = fn
    assert len(weight_map) == 114
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes")


if __name__ == "__main__":
    main()
