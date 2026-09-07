"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf).

Cast the 112 per-layer projection matrices to bfloat16, keep everything else
(embeddings, lm_head) as float32, and write a sharded safetensors checkpoint
with a valid index file, using plain torch + safetensors (no HF trainer
plumbing, since we need exact control over shard byte budget).
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parents[1] / "inputs" / "base"
OUT_DIR = HERE

SHARD_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MiB, tensor data only

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_index(in_dir: Path) -> dict:
    with open(in_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def load_all_tensors(in_dir: Path, weight_map: dict) -> dict[str, torch.Tensor]:
    tensors = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        with safe_open(in_dir / shard_file, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    weight_map_in = load_index(IN_DIR)["weight_map"]
    tensors = load_all_tensors(IN_DIR, weight_map_in)

    assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"

    proj_keys = {k for k in tensors if PROJ_RE.match(k)}
    assert len(proj_keys) == 112, f"expected 112 projection matrices, got {len(proj_keys)}"

    out_tensors: dict[str, torch.Tensor] = {}
    for key, t in tensors.items():
        if key in proj_keys:
            out_tensors[key] = t.to(torch.bfloat16).contiguous()
        else:
            assert t.dtype == torch.float32, f"{key} unexpectedly not float32 in input"
            out_tensors[key] = t.contiguous()

    # --- Required checks: fail loudly before writing anything ---
    n_bf16 = sum(1 for t in out_tensors.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 112, f"expected exactly 112 bfloat16 tensors, got {n_bf16}"
    assert out_tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out_tensors["model.embed_tokens.weight"].dtype == torch.float32
    assert len(out_tensors) == 114, f"expected 114 output tensors, got {len(out_tensors)}"

    # Value check: non-cast tensors must be bit-identical to input.
    for key in tensors:
        if key not in proj_keys:
            assert torch.equal(tensors[key], out_tensors[key]), f"{key} value changed unexpectedly"

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # --- Greedy bin-packing into shards under the byte budget. ---
    # Oversized tensors (> limit) get their own shard, as required.
    # Otherwise fill shards in key order, first-fit into the current shard.
    keys_sorted = sorted(out_tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for key in keys_sorted:
        size = tensor_bytes(out_tensors[key])
        if size > SHARD_LIMIT_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([key])
            continue
        if current and current_bytes + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += size

    if current:
        shards.append(current)

    n_shards = len(shards)
    digits = max(5, len(str(n_shards)))
    shard_names = [
        f"model-{i + 1:0{digits}d}-of-{n_shards:0{digits}d}.safetensors" for i in range(n_shards)
    ]

    weight_map_out: dict[str, str] = {}
    total_size = 0
    for shard_name, keys in zip(shard_names, shards):
        shard_dict = {k: out_tensors[k] for k in keys}
        save_file(shard_dict, OUT_DIR / shard_name, metadata={"format": "pt"})
        for k in keys:
            weight_map_out[k] = shard_name
            total_size += tensor_bytes(out_tensors[k])

    assert set(weight_map_out) == set(out_tensors), "weight_map does not cover all tensors"

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map_out}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(out_tensors)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED CHECK: {e}", file=sys.stderr)
        sys.exit(1)
