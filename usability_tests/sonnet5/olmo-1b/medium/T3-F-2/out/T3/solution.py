"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Plain script on top of `safetensors` + `torch` (both on the F-allowed list).
Rejected `transformers`-level `save_pretrained(dtype=...)` because it applies
one dtype to the whole model; there is no per-tensor-name dtype override in
its public API, and this task needs an exact regex-targeted split (112
projection matrices to bf16, everything else stays float32). Reading and
writing the safetensors shards directly gives full control over which
tensors are cast and how they are packed into shards, and makes the
required checks trivial to enforce before anything is written.

Usage: python solution.py [--in-dir inputs/base] [--out-dir out/T3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard

# Exactly the 112 large projection matrices named in TASK.md. Deliberately a
# tight regex (not ".*weight") so embeddings / lm_head are never matched.
PROJECTION_RE = re.compile(
    r"^model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q|k|v|o)_proj\.weight"
    r"|mlp\.(?:gate|up|down)_proj\.weight"
    r")$"
)

DTYPE_BYTES = {
    torch.float32: 4,
    torch.bfloat16: 2,
}


def load_all_tensors(in_dir: Path) -> dict[str, torch.Tensor]:
    index_path = in_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    tensors: dict[str, torch.Tensor] = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        with safe_open(in_dir / shard_file, framework="pt") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)

    if set(tensors.keys()) != set(weight_map.keys()):
        raise RuntimeError(
            "Loaded tensor keys do not match the input index's weight_map keys"
        )
    return tensors


def cast_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if PROJECTION_RE.match(name):
            out[name] = tensor.to(torch.bfloat16)
        else:
            # Everything else stays float32 with unchanged values (no buffers
            # to drop in this checkpoint, per TASK.md).
            if tensor.dtype != torch.float32:
                raise RuntimeError(f"unexpected non-float32 input dtype for {name}")
            out[name] = tensor
    return out


def run_required_checks(tensors: dict[str, torch.Tensor]) -> None:
    bf16_count = sum(1 for t in tensors.values() if t.dtype == torch.bfloat16)
    if bf16_count != 112:
        raise AssertionError(f"expected exactly 112 bfloat16 tensors, got {bf16_count}")

    probe = "model.layers.0.self_attn.q_proj.weight"
    if tensors[probe].dtype != torch.bfloat16:
        raise AssertionError(f"{probe} must be bfloat16")

    if tensors["model.embed_tokens.weight"].dtype != torch.float32:
        raise AssertionError("model.embed_tokens.weight must be float32")

    if len(tensors) != 114:
        raise AssertionError(f"expected exactly 114 tensors, got {len(tensors)}")


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * DTYPE_BYTES[t.dtype]


def plan_shards(tensors: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedily bin-pack tensors into shards <= MAX_SHARD_BYTES.

    A tensor larger than the limit on its own gets its own shard, as
    required for model.embed_tokens.weight and lm_head.weight.
    """
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in sorted(tensors.keys()):
        size = tensor_bytes(tensors[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size

    if current:
        shards.append(current)

    return shards


def write_sharded(tensors: dict[str, torch.Tensor], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_plan = plan_shards(tensors)
    n = len(shard_plan)
    digits = 5

    weight_map: dict[str, str] = {}
    total_size = 0
    shard_names = [f"model-{i + 1:0{digits}d}-of-{n:0{digits}d}.safetensors" for i in range(n)]

    for shard_name, names in zip(shard_names, shard_plan):
        shard_tensors = {name: tensors[name].contiguous() for name in names}
        save_file(shard_tensors, out_dir / shard_name, metadata={"format": "pt"})
        for name in names:
            weight_map[name] = shard_name
            total_size += tensor_bytes(tensors[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("inputs/base"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/T3"))
    args = parser.parse_args()

    tensors = load_all_tensors(args.in_dir)
    cast = cast_tensors(tensors)
    run_required_checks(cast)  # fail loudly before writing anything
    write_sharded(cast, args.out_dir)

    print(f"Wrote {len(cast)} tensors to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
