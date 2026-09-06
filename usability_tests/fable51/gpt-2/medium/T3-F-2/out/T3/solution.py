"""T3: mixed-precision sharded export of GPT-2 (124M).

Plain torch + safetensors script. Casts exactly the 48 projection matrices to
bfloat16, keeps every other parameter in float32, drops the 12 causal-mask
buffers, and writes a sharded safetensors checkpoint with an index file.
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
OUT = HERE
SHARD_BUDGET = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data

PROJ_RE = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    src = load_file(SRC)
    if len(src) != 160:
        fail(f"expected 160 input tensors, got {len(src)}")

    out: dict[str, torch.Tensor] = {}
    dropped = []
    for name, t in src.items():
        if BUFFER_RE.match(name):
            dropped.append(name)
            continue
        if t.dtype != torch.float32:
            fail(f"{name} is {t.dtype}, expected float32 input")
        if PROJ_RE.match(name):
            out[name] = t.to(torch.bfloat16)
        else:
            out[name] = t.contiguous()

    # ---- required checks (before writing) ----
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 48:
        fail(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail("h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        fail("wte.weight is not float32")
    if len(out) != 148:
        fail(f"expected 148 output tensors, got {len(out)}")
    if len(dropped) != 12:
        fail(f"expected to drop 12 buffers, dropped {len(dropped)}")
    for name, t in out.items():
        if t.dtype == torch.float32 and not torch.equal(t, src[name]):
            fail(f"{name} float32 values changed")
    n_f32 = sum(1 for t in out.values() if t.dtype == torch.float32)
    if n_f32 + n_bf16 != 148:
        fail("unexpected dtypes present")

    # ---- greedy sharding in key order, oversized tensors alone ----
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name, t in out.items():
        nbytes = t.numel() * t.element_size()
        if nbytes > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([name])
            continue
        if cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        shard = {k: out[k] for k in names}
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if len(shard) > 1 and size > SHARD_BUDGET:
            fail(f"shard {fname} exceeds budget: {size}")
        total += size
        save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname

    if len(weight_map) != 148:
        fail(f"weight_map has {len(weight_map)} entries, expected 148")
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes; dropped {dropped}")


if __name__ == "__main__":
    main()
