"""T3: mixed-precision export of GPT-2 (124M) with sharding."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
SHARD_BUDGET = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data per shard

PROJ_RE = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    sd = load_file(SRC)
    if len(sd) != 160:
        fail(f"expected 160 input tensors, got {len(sd)}")

    out: dict[str, torch.Tensor] = {}
    n_cast = n_dropped = 0
    for name, t in sd.items():
        if BUFFER_RE.match(name):
            n_dropped += 1
            continue
        if PROJ_RE.match(name):
            if t.dtype != torch.float32:
                fail(f"{name} expected float32 input, got {t.dtype}")
            out[name] = t.to(torch.bfloat16).contiguous()
            n_cast += 1
        else:
            if t.dtype != torch.float32:
                fail(f"{name} expected float32, got {t.dtype}")
            out[name] = t.contiguous()

    # Required checks (before writing).
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 48:
        fail(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if n_cast != 48 or n_dropped != 12:
        fail(f"cast {n_cast} (want 48), dropped {n_dropped} (want 12)")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail("h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        fail("wte.weight is not float32")
    if len(out) != 148:
        fail(f"expected 148 output tensors, got {len(out)}")
    for name, t in out.items():
        if t.dtype not in (torch.float32, torch.bfloat16):
            fail(f"{name} has unexpected dtype {t.dtype}")
        if t.dtype == torch.float32 and PROJ_RE.match(name):
            fail(f"{name} should be bfloat16")

    # Greedy sharding in input key order; an oversized tensor gets its own shard.
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
        if cur and cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    total_size = sum(t.numel() * t.element_size() for t in out.values())
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard = {n: out[n] for n in names}
        shard_bytes = sum(t.numel() * t.element_size() for t in shard.values())
        if shard_bytes > SHARD_BUDGET and len(shard) != 1:
            fail(f"shard {fname} exceeds budget with {len(shard)} tensors")
        save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for n in names:
            weight_map[n] = fname
        print(f"{fname}: {len(names)} tensors, {shard_bytes} bytes")

    if len(weight_map) != 148:
        fail(f"weight_map has {len(weight_map)} entries, want 148")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"wrote {n_shards} shards, {len(weight_map)} tensors, total_size={total_size}")


if __name__ == "__main__":
    main()
