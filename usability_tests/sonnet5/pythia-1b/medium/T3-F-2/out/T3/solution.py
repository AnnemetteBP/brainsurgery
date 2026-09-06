"""
T3: Mixed-precision export with sharding (Pythia-1B).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). Chosen
over mergekit/transformers dtype export because this task needs per-tensor
dtype targeting by exact name pattern (64 specific projection matrices) plus
buffer deletion plus a custom 256 MiB shard budget with a single-tensor
exception -- more precisely controlled as a short script than coerced out of
a merge-config or `save_pretrained` dtype flag.

Usage: python solution.py <input_safetensors> <output_dir>
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

SHARD_BUDGET_BYTES = 256 * 1024 * 1024  # 268,435,456

# Exactly the 4 projection-matrix families, per layer, that must become bf16.
BF16_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)\.weight"
    r"|mlp\.(dense_h_to_4h|dense_4h_to_h)\.weight)$"
)

# The 3 non-parameter buffers per layer, to be dropped entirely.
BUFFER_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def load_and_transform(input_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(input_path), framework="pt") as f:
        for name in f.keys():
            if BUFFER_PATTERN.match(name):
                continue
            t = f.get_tensor(name)
            if BF16_PATTERN.match(name):
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            tensors[name] = t.contiguous()
    return tensors


def run_required_checks(tensors: dict[str, torch.Tensor]) -> None:
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 64, (
        f"expected exactly 64 bfloat16 tensors, got {len(bf16_names)}: {bf16_names}"
    )

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert qkv0 in tensors, f"missing {qkv0}"
    assert tensors[qkv0].dtype == torch.bfloat16, (
        f"{qkv0} must be bfloat16, got {tensors[qkv0].dtype}"
    )

    embed_in = "gpt_neox.embed_in.weight"
    assert embed_in in tensors, f"missing {embed_in}"
    assert tensors[embed_in].dtype == torch.float32, (
        f"{embed_in} must be float32, got {tensors[embed_in].dtype}"
    )

    assert len(tensors) == 196, f"expected exactly 196 tensors, got {len(tensors)}"

    # No dropped parameters, no leftover buffers.
    dropped_but_not_buffer = [
        n for n in tensors if BUFFER_PATTERN.match(n)
    ]
    assert not dropped_but_not_buffer, f"buffers leaked into output: {dropped_but_not_buffer}"


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def plan_shards(tensors: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedy bin-packing in insertion order; oversized tensors get their own shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, t in tensors.items():
        size = tensor_nbytes(t)
        if size > SHARD_BUDGET_BYTES:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def write_sharded(tensors: dict[str, torch.Tensor], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_groups = plan_shards(tensors)
    n = len(shard_groups)

    weight_map: dict[str, str] = {}
    total_size = 0
    shard_filenames = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

    for filename, names in zip(shard_filenames, shard_groups):
        shard_tensors = {name: tensors[name] for name in names}
        save_file(shard_tensors, str(out_dir / filename), metadata={"format": "pt"})
        for name in names:
            weight_map[name] = filename
            total_size += tensor_nbytes(tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input_safetensors> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    tensors = load_and_transform(input_path)
    run_required_checks(tensors)
    write_sharded(tensors, out_dir)
    print(f"wrote {len(tensors)} tensors across shards to {out_dir}")


if __name__ == "__main__":
    main()
