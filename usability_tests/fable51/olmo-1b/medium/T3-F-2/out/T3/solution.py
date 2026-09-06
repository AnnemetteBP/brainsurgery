"""T3: mixed-precision export of OLMo-1B-0724-hf with 256 MiB sharding.

Plain torch + safetensors script. Casts exactly the 112 per-layer projection
matrices to bfloat16, leaves everything else float32 and untouched, and writes
a sharded safetensors checkpoint with an index file.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
DST = ROOT / "out" / "T3"
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard

PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names = sorted(weight_map)  # deterministic order

    # Load every tensor (5 GB fp32; fits in RAM) and cast the projections.
    tensors: dict[str, torch.Tensor] = {}
    handles = {}
    for name in names:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(SRC / shard, framework="pt", device="cpu")
        t = handles[shard].get_tensor(name)
        if t.dtype != torch.float32:
            fail(f"unexpected input dtype {t.dtype} for {name}")
        if PROJ_RE.match(name):
            t = t.to(torch.bfloat16)
        tensors[name] = t.contiguous()

    # Required checks, before writing anything.
    n_bf16 = sum(t.dtype == torch.bfloat16 for t in tensors.values())
    if n_bf16 != 112:
        fail(f"expected 112 bfloat16 tensors, got {n_bf16}")
    if tensors["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if tensors["model.embed_tokens.weight"].dtype != torch.float32:
        fail("model.embed_tokens.weight is not float32")
    if tensors["lm_head.weight"].dtype != torch.float32:
        fail("lm_head.weight is not float32")
    if len(tensors) != 114:
        fail(f"expected 114 tensors, got {len(tensors)}")
    if set(tensors) != set(weight_map):
        fail("tensor name set changed")

    # Greedy sharding: fill up to SHARD_BUDGET; an oversized tensor goes alone.
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if nbytes > SHARD_BUDGET:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current_bytes + nbytes > SHARD_BUDGET:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    for group in shards:
        total = sum(tensors[n].numel() * tensors[n].element_size() for n in group)
        if total > SHARD_BUDGET and len(group) != 1:
            fail(f"shard exceeds budget: {total} bytes, {len(group)} tensors")

    DST.mkdir(parents=True, exist_ok=True)
    n = len(shards)
    out_map: dict[str, str] = {}
    total_size = 0
    for i, group in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: tensors[k] for k in group}, DST / fname, metadata={"format": "pt"})
        for k in group:
            out_map[k] = fname
            total_size += tensors[k].numel() * tensors[k].element_size()
    (DST / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": out_map}, indent=2)
        + "\n"
    )

    # Post-write verification of the on-disk result.
    seen = 0
    for fname in sorted(set(out_map.values())):
        with safe_open(DST / fname, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            seen += len(keys)
            size = 0
            for k in keys:
                t = f.get_tensor(k)
                size += t.numel() * t.element_size()
                if k in ("model.embed_tokens.weight", "lm_head.weight"):
                    if not torch.equal(t, tensors[k]):
                        fail(f"{k} changed on disk")
            if size > SHARD_BUDGET and len(keys) != 1:
                fail(f"{fname} exceeds shard budget")
    if seen != 114 or len(out_map) != 114:
        fail(f"output has {seen} tensors on disk, {len(out_map)} in index")
    print(f"OK: wrote {n} shards, {seen} tensors, {n_bf16} bfloat16, total {total_size} bytes")


if __name__ == "__main__":
    main()
