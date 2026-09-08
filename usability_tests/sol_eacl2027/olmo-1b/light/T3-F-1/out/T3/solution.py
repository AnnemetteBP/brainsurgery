#!/usr/bin/env python3
"""Create the T3 mixed-precision, size-bounded safetensors export."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUT = Path("out/T3")
CAP = 256 * 1024 * 1024
INDEX_NAME = "model.safetensors.index.json"
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
CAST_NAMES = {
    f"model.layers.{layer}.{projection}.weight"
    for layer in range(16)
    for projection in PROJECTIONS
}


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    with (BASE / INDEX_NAME).open() as f:
        source_index = json.load(f)
    source_map = source_index["weight_map"]
    names = sorted(source_map)

    # All required checks occur before the first output checkpoint file is written.
    assert len(CAST_NAMES) == 112, f"expected 112 cast names, got {len(CAST_NAMES)}"
    assert len(names) == 114, f"expected 114 input/output tensors, got {len(names)}"
    missing = CAST_NAMES.difference(names)
    assert not missing, f"projection tensors missing from input: {sorted(missing)}"
    assert "model.layers.0.self_attn.q_proj.weight" in CAST_NAMES
    assert "model.embed_tokens.weight" in names

    OUT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(BASE / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(source_map.values()))
        }
        for name in names:
            source = handles[source_map[name]].get_tensor(name)
            assert source.dtype == torch.float32, f"input {name} is {source.dtype}, not float32"

        # Remove only stale generated checkpoint artifacts, preserving this script/report.
        for path in OUT.glob("model-*-of-*.safetensors"):
            path.unlink()
        (OUT / INDEX_NAME).unlink(missing_ok=True)

        # Build logical shards from output tensor sizes. Oversized tensors stand alone.
        groups: list[list[str]] = []
        current: list[str] = []
        current_size = 0
        for name in names:
            source = handles[source_map[name]].get_tensor(name)
            output_element_size = 2 if name in CAST_NAMES else 4
            size = source.numel() * output_element_size
            if current and (size > CAP or current_size + size > CAP):
                groups.append(current)
                current, current_size = [], 0
            if size > CAP:
                groups.append([name])
            else:
                current.append(name)
                current_size += size
        if current:
            groups.append(current)

        weight_map: dict[str, str] = {}
        total_size = 0
        shard_count = len(groups)
        for number, group in enumerate(groups, 1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {}
            for name in group:
                tensor = handles[source_map[name]].get_tensor(name)
                tensor = tensor.to(torch.bfloat16) if name in CAST_NAMES else tensor
                expected = torch.bfloat16 if name in CAST_NAMES else torch.float32
                assert tensor.dtype == expected, f"wrong output dtype for {name}: {tensor.dtype}"
                tensors[name] = tensor
                weight_map[name] = filename
                total_size += tensor_bytes(tensor)
            shard_size = sum(tensor_bytes(tensor) for tensor in tensors.values())
            assert shard_size <= CAP or len(tensors) == 1, (
                f"shard {filename} exceeds cap with multiple tensors: {shard_size}"
            )
            save_file(tensors, OUT / filename)

    assert len(weight_map) == 114
    assert set(weight_map) == set(names)
    with (OUT / INDEX_NAME).open("w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print(f"Wrote {len(weight_map)} tensors in {shard_count} shards ({total_size} bytes)")


if __name__ == "__main__":
    main()
