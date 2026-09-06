"""T3: mixed-precision export of GPT-2 (124M) with sharding."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
SHARD_LIMIT = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data per shard

# exactly the four projection matrices per layer
CAST_RE = re.compile(r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
DROP_RE = re.compile(r"^h\.\d+\.attn\.bias$")


def main() -> None:
    state = {}
    with safe_open(IN_PATH, framework="pt") as f:
        names = list(f.keys())
        for name in names:
            if DROP_RE.match(name):
                continue
            t = f.get_tensor(name)
            state[name] = t.to(torch.bfloat16) if CAST_RE.match(name) else t.to(torch.float32)

    dropped = [n for n in names if DROP_RE.match(n)]
    n_bf16 = sum(1 for t in state.values() if t.dtype is torch.bfloat16)

    # required checks -- fail loudly before writing
    assert len(dropped) == 12, f"expected 12 mask buffers dropped, got {len(dropped)}"
    assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
    assert state["h.0.attn.c_attn.weight"].dtype is torch.bfloat16, "c_attn.weight not bfloat16"
    assert state["wte.weight"].dtype is torch.float32, "wte.weight not float32"
    assert len(state) == 148, f"expected 148 tensors, got {len(state)}"
    for n, t in state.items():
        assert t.dtype is (torch.bfloat16 if CAST_RE.match(n) else torch.float32), f"dtype {n}"

    # greedy sharding in file key order; a tensor larger than the budget gets its own shard
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_size = 0
    for name, t in state.items():
        size = t.numel() * t.element_size()
        if cur and cur_size + size > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += size
    if cur:
        shards.append(cur)

    total = len(shards)
    weight_map = {}
    total_size = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        body = sum(state[k].numel() * state[k].element_size() for k in keys)
        assert body <= SHARD_LIMIT or len(keys) == 1, f"shard {fname} over budget"
        total_size += body
        save_file({k: state[k].contiguous() for k in keys}, os.path.join(OUT_DIR, fname),
                  metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname

    assert len(weight_map) == 148, f"weight_map has {len(weight_map)} entries"
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    print(f"wrote {total} shards, {len(weight_map)} tensors, {total_size} bytes to {OUT_DIR}")


if __name__ == "__main__":
    main()
