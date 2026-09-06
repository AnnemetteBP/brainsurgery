"""T3: mixed-precision export of Pythia-1B with sharding."""

import json
import os
import re
import struct

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
OUT = os.path.join(ROOT, "out", "T3")

SHARD_LIMIT = 256 * 1024 * 1024

PROJ = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)"
    r"|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def header_order(path):
    """Tensor names in the order they appear in the safetensors header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def main():
    from safetensors import safe_open

    names = header_order(SRC)
    out = {}
    with safe_open(SRC, framework="pt") as f:
        assert set(f.keys()) == set(names), "header order mismatch"
        for name in names:
            if BUFFER.match(name):
                continue
            t = f.get_tensor(name)
            out[name] = t.to(torch.bfloat16) if PROJ.match(name) else t.to(torch.float32)

    # Required checks, before writing anything.
    n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
    if n_bf16 != 64:
        raise SystemExit(f"expected 64 bfloat16 tensors, got {n_bf16}")
    qkv = "gpt_neox.layers.0.attention.query_key_value.weight"
    if out[qkv].dtype is not torch.bfloat16:
        raise SystemExit(f"{qkv} is {out[qkv].dtype}, expected bfloat16")
    if out["gpt_neox.embed_in.weight"].dtype is not torch.float32:
        raise SystemExit("gpt_neox.embed_in.weight is not float32")
    if len(out) != 196:
        raise SystemExit(f"expected 196 tensors, got {len(out)}")
    if any(t.dtype not in (torch.bfloat16, torch.float32) for t in out.values()):
        raise SystemExit("unexpected dtype in output")

    # Greedy sharding in checkpoint order; an oversized tensor lands alone.
    shards, cur, cur_size = [], {}, 0
    for name, t in out.items():
        size = t.numel() * t.element_size()
        if cur and cur_size + size > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = t
        cur_size += size
    if cur:
        shards.append(cur)

    total = len(shards)
    os.makedirs(OUT, exist_ok=True)
    weight_map, total_size = {}, 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if size > SHARD_LIMIT and len(shard) > 1:
            raise SystemExit(f"{fname} exceeds the shard budget with {len(shard)} tensors")
        total_size += size
        save_file({k: v.contiguous() for k, v in shard.items()}, os.path.join(OUT, fname))
        for k in shard:
            weight_map[k] = fname

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"wrote {len(out)} tensors ({n_bf16} bfloat16) into {total} shards, {total_size} bytes")


main()
