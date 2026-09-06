"""T3: mixed-precision export with sharding for GPT-2 (124M).

Casts the 48 projection matrices to bfloat16, keeps everything else float32,
drops the 12 causal-mask buffers, and writes a sharded safetensors checkpoint
with an index file into out/T3/.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
OUT_DIR = HERE

N_LAYERS = 12
SHARD_LIMIT = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data per shard

# Exactly the projection matrices that must become bfloat16.
BF16_KEYS = set()
for i in range(N_LAYERS):
    BF16_KEYS.add(f"h.{i}.attn.c_attn.weight")
    BF16_KEYS.add(f"h.{i}.attn.c_proj.weight")
    BF16_KEYS.add(f"h.{i}.mlp.c_fc.weight")
    BF16_KEYS.add(f"h.{i}.mlp.c_proj.weight")

# Non-parameter buffers to drop.
DROP_KEYS = {f"h.{i}.attn.bias" for i in range(N_LAYERS)}


def main() -> None:
    # ---- load, preserving the on-file key order -------------------------
    with safe_open(IN_PATH, framework="pt", device="cpu") as f:
        order = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in order}

    missing = (BF16_KEYS | DROP_KEYS) - set(order)
    if missing:
        raise SystemExit(f"input is missing expected tensors: {sorted(missing)}")

    # ---- transform ------------------------------------------------------
    out_order = []
    out = {}
    for k in order:
        if k in DROP_KEYS:
            continue
        t = tensors[k]
        if k in BF16_KEYS:
            if t.dtype != torch.float32:
                raise SystemExit(f"{k}: expected float32 input, got {t.dtype}")
            t = t.to(torch.bfloat16)
        else:
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
        out[k] = t.contiguous().clone()
        out_order.append(k)

    # ---- required checks (fail loudly, before writing) ------------------
    n_bf16 = sum(1 for v in out.values() if v.dtype == torch.bfloat16)
    if n_bf16 != 48:
        raise SystemExit(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        raise SystemExit("h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        raise SystemExit("wte.weight is not float32")
    if len(out) != 148:
        raise SystemExit(f"expected exactly 148 output tensors, got {len(out)}")
    for k in DROP_KEYS:
        if k in out:
            raise SystemExit(f"buffer {k} was not dropped")
    for k in BF16_KEYS:
        if out[k].dtype != torch.bfloat16:
            raise SystemExit(f"{k} is not bfloat16")
    for k in out_order:
        if k not in BF16_KEYS and out[k].dtype != torch.float32:
            raise SystemExit(f"{k} should be float32, is {out[k].dtype}")

    # ---- greedy sharding in key order -----------------------------------
    shards = []  # list of list-of-keys
    cur, cur_bytes = [], 0
    for k in out_order:
        nbytes = out[k].numel() * out[k].element_size()
        if cur and cur_bytes + nbytes > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    total = len(shards)
    names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]

    for keys, name in zip(shards, names):
        data = sum(out[k].numel() * out[k].element_size() for k in keys)
        if len(keys) > 1 and data > SHARD_LIMIT:
            raise SystemExit(f"shard {name} exceeds the budget: {data} bytes")

    # ---- write ----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map = {}
    total_size = 0
    for keys, name in zip(shards, names):
        save_file({k: out[k] for k in keys}, os.path.join(OUT_DIR, name))
        for k in keys:
            weight_map[k] = name
            total_size += out[k].numel() * out[k].element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=False)
        fh.write("\n")

    if len(weight_map) != 148:
        raise SystemExit(f"weight_map has {len(weight_map)} entries, expected 148")

    print(f"wrote {total} shards, {len(weight_map)} tensors, {total_size} bytes")
    for keys, name in zip(shards, names):
        data = sum(out[k].numel() * out[k].element_size() for k in keys)
        print(f"  {name}: {len(keys):3d} tensors, {data} bytes")


if __name__ == "__main__":
    main()
