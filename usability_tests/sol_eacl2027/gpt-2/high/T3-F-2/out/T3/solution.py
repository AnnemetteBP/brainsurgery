#!/usr/bin/env python3
"""Export the supplied GPT-2 checkpoint with precise mixed dtypes and sharding."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 64 * 1024 * 1024


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def expected_projection_names() -> set[str]:
    suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    return {f"h.{layer}.{suffix}" for layer in range(12) for suffix in suffixes}


def make_shards(
    tensors: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    """Greedily pack sorted tensors without splitting any tensor."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_bytes(tensor)

        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = {}
                current_bytes = 0
            current[name] = tensor
            current_bytes += size

    if current:
        shards.append(current)
    return shards


def validate_before_writing(
    source: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    projections: set[str],
    buffers: set[str],
) -> None:
    """Enforce task requirements before any checkpoint artifact is written."""
    assert len(projections) == 48
    assert projections <= source.keys(), "one or more projection names are absent"
    assert buffers <= source.keys(), "one or more causal-mask buffers are absent"
    assert set(output) == set(source) - buffers, "output key set is incorrect"

    bf16_names = {name for name, tensor in output.items() if tensor.dtype == torch.bfloat16}
    assert bf16_names == projections, (
        f"expected exactly the 48 projections in bfloat16; got {len(bf16_names)}"
    )
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert output["wte.weight"].dtype == torch.float32
    assert len(output) == 148, f"expected 148 tensors, got {len(output)}"

    for name, tensor in output.items():
        if name in projections:
            expected = source[name].to(torch.bfloat16)
            assert torch.equal(tensor, expected), f"bfloat16 values differ for {name}"
        else:
            assert tensor.dtype == torch.float32, f"non-projection is not float32: {name}"
            assert torch.equal(tensor, source[name]), f"values changed for {name}"


def validate_written_checkpoint(
    tensors: dict[str, torch.Tensor],
    shards: list[dict[str, torch.Tensor]],
    weight_map: dict[str, str],
) -> None:
    assert set(weight_map) == set(tensors)
    seen: set[str] = set()

    for shard in shards:
        filename = weight_map[next(iter(shard))]
        shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
        if shard_size > MAX_SHARD_BYTES:
            assert len(shard) == 1, f"oversized shard is not a singleton: {filename}"

        loaded = load_file(OUTPUT_DIR / filename, device="cpu")
        assert set(loaded) == set(shard), f"wrong tensor keys in {filename}"
        for name, expected in shard.items():
            assert weight_map[name] == filename
            assert loaded[name].dtype == expected.dtype, f"wrong dtype for {name}"
            assert torch.equal(loaded[name], expected), f"wrong saved values for {name}"
            assert name not in seen, f"duplicate tensor {name}"
            seen.add(name)

    assert seen == set(tensors)
    with INDEX_PATH.open(encoding="utf-8") as handle:
        written_index = json.load(handle)
    assert written_index["weight_map"] == weight_map
    assert written_index["metadata"]["total_size"] == sum(
        tensor_bytes(tensor) for tensor in tensors.values()
    )


def main() -> None:
    source = load_file(INPUT, device="cpu")
    projections = expected_projection_names()
    buffers = {f"h.{layer}.attn.bias" for layer in range(12)}

    output = {
        name: tensor.to(torch.bfloat16).contiguous()
        if name in projections
        else tensor.to(torch.float32).contiguous()
        for name, tensor in source.items()
        if name not in buffers
    }

    # These assertions intentionally precede all checkpoint/index writes.
    validate_before_writing(source, output, projections, buffers)
    shards = make_shards(output)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    validate_written_checkpoint(output, shards, weight_map)
    print(
        f"Wrote and verified {len(output)} tensors in {shard_count} shards "
        f"({len(projections)} bfloat16 projections)."
    )


if __name__ == "__main__":
    main()
