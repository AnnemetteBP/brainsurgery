"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). No
higher-level merge/export tool in the allowed list gives per-tensor dtype
control plus a from-scratch shard packer, so this is the most direct route:
load every tensor once, cast exactly the 112 named projection matrices to
bfloat16, keep everything else float32, run the required checks, then pack
into <=256MiB shards (oversized tensors alone in their own shard) and write
a matching index.json.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes, tensor data only

# Exactly the 112 projection-matrix names: 7 per layer, layers 0..15.
PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_index(in_dir: Path) -> dict:
    with open(in_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def load_all_tensors(in_dir: Path, weight_map: dict) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(in_dir / shard_name, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def cast_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name, t in tensors.items():
        if PROJ_RE.match(name):
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            assert t.dtype == torch.float32, f"{name} unexpectedly not float32 in input"
            out[name] = t.contiguous()
    return out


def run_required_checks(tensors: dict[str, torch.Tensor]) -> None:
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 112, f"expected 112 bfloat16 tensors, got {len(bf16_names)}"
    for n in bf16_names:
        assert PROJ_RE.match(n), f"unexpected tensor cast to bfloat16: {n}"

    assert tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16, (
        "model.layers.0.self_attn.q_proj.weight is not bfloat16"
    )
    assert tensors["model.embed_tokens.weight"].dtype == torch.float32, (
        "model.embed_tokens.weight is not float32"
    )
    assert tensors["lm_head.weight"].dtype == torch.float32, "lm_head.weight is not float32"
    assert len(tensors) == 114, f"expected 114 tensors total, got {len(tensors)}"


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def plan_shards(tensors: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedy bin-pack in a stable name order; oversized tensors get their own shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in sorted(tensors.keys()):
        size = tensor_nbytes(tensors[name])
        if size > SHARD_BUDGET:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > SHARD_BUDGET:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size

    if current:
        shards.append(current)
    return shards


def write_shards(tensors: dict[str, torch.Tensor], shard_plan: list[list[str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(shard_plan)
    weight_map = {}
    total_size = 0

    for i, names in enumerate(shard_plan, start=1):
        shard_file = f"model-{i:05d}-of-{n:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in names}
        save_file(shard_tensors, out_dir / shard_file, metadata={"format": "pt"})
        for name in names:
            weight_map[name] = shard_file
            total_size += tensor_nbytes(tensors[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def main() -> None:
    weight_map = load_index(IN_DIR)["weight_map"]
    assert len(weight_map) == 114, f"expected 114 tensors in input, got {len(weight_map)}"

    tensors = load_all_tensors(IN_DIR, weight_map)
    tensors = cast_tensors(tensors)
    run_required_checks(tensors)

    shard_plan = plan_shards(tensors)
    for names in shard_plan:
        shard_bytes = sum(tensor_nbytes(tensors[n]) for n in names)
        if len(names) > 1:
            assert shard_bytes <= SHARD_BUDGET, f"shard over budget: {shard_bytes} bytes"

    write_shards(tensors, shard_plan, OUT_DIR)
    print(f"Wrote {len(tensors)} tensors across {len(shard_plan)} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
