#!/usr/bin/env python3
"""Create the mixed-precision, sharded GPT-2 checkpoint required by T3."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


MAX_SHARD_BYTES = 64 * 1024 * 1024
OUTPUT_TENSOR_COUNT = 148

SCRIPT_DIR = Path(__file__).resolve().parent
SANDBOX_DIR = SCRIPT_DIR.parents[1]
INPUT_PATH = SANDBOX_DIR / "inputs" / "base" / "model.safetensors"
INDEX_PATH = SCRIPT_DIR / "model.safetensors.index.json"


def projection_names() -> set[str]:
    suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    return {f"h.{layer}.{suffix}" for layer in range(12) for suffix in suffixes}


def buffer_names() -> set[str]:
    return {f"h.{layer}.attn.bias" for layer in range(12)}


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(
    tensors: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    """Greedily pack sorted tensors; any oversized tensor forms its own shard."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0

    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_nbytes(tensor)

        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_size = 0
            shards.append({name: tensor})
        elif current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {name: tensor}
            current_size = size
        else:
            current[name] = tensor
            current_size += size

    if current:
        shards.append(current)
    return shards


def check_before_writing(
    source: dict[str, torch.Tensor], output: dict[str, torch.Tensor]
) -> None:
    projections = projection_names()
    buffers = buffer_names()
    source_names = set(source)
    output_names = set(output)

    missing_projections = projections - source_names
    missing_buffers = buffers - source_names
    assert not missing_projections, f"missing projection tensors: {sorted(missing_projections)}"
    assert not missing_buffers, f"missing causal-mask buffers: {sorted(missing_buffers)}"
    assert output_names == source_names - buffers, "output key set is not input minus buffers"

    bf16_names = {name for name, tensor in output.items() if tensor.dtype == torch.bfloat16}
    assert bf16_names == projections, (
        f"expected exactly the 48 projection tensors in bfloat16; got {len(bf16_names)} "
        f"with missing={sorted(projections - bf16_names)} and extra={sorted(bf16_names - projections)}"
    )
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert output["wte.weight"].dtype == torch.float32
    assert len(output) == OUTPUT_TENSOR_COUNT, (
        f"expected {OUTPUT_TENSOR_COUNT} output tensors, got {len(output)}"
    )

    non_float32 = {
        name: str(tensor.dtype)
        for name, tensor in output.items()
        if name not in projections and tensor.dtype != torch.float32
    }
    assert not non_float32, f"non-projection tensors not float32: {non_float32}"


def verify_written_checkpoint(expected_names: set[str]) -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    assert set(weight_map) == expected_names, "index key set differs from output key set"

    seen: set[str] = set()
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = SCRIPT_DIR / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            sizes = {name: tensor_nbytes(handle.get_tensor(name)) for name in names}
        payload_size = sum(sizes.values())
        oversized = [name for name, size in sizes.items() if size > MAX_SHARD_BYTES]
        assert payload_size <= MAX_SHARD_BYTES or (
            len(names) == 1 and len(oversized) == 1
        ), f"shard {shard_name} exceeds the payload limit without a single oversized tensor"
        assert all(weight_map[name] == shard_name for name in names)
        assert not (seen & names), f"duplicate tensors across shards: {sorted(seen & names)}"
        seen.update(names)

    assert seen == expected_names, "physical shard contents differ from indexed tensors"


def main() -> None:
    source = load_file(str(INPUT_PATH), device="cpu")
    projections = projection_names()
    buffers = buffer_names()

    output = {
        name: (tensor.to(torch.bfloat16) if name in projections else tensor)
        for name, tensor in source.items()
        if name not in buffers
    }

    # All task-mandated checks, plus exact key/dtype checks, happen before the
    # first checkpoint or index file is written.
    check_before_writing(source, output)
    shards = make_shards(output)
    for shard in shards:
        payload_size = sum(tensor_nbytes(tensor) for tensor in shard.values())
        assert payload_size <= MAX_SHARD_BYTES or (
            len(shard) == 1
            and next(iter(shard.values())).numel()
            * next(iter(shard.values())).element_size()
            > MAX_SHARD_BYTES
        ), "invalid pre-write shard plan"

    # Remove only checkpoint artifacts produced by an earlier invocation.
    for old_shard in SCRIPT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    INDEX_PATH.unlink(missing_ok=True)

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    for shard_number, shard in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, str(SCRIPT_DIR / filename), metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_nbytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_written_checkpoint(set(output))

    print(
        f"Wrote {len(output)} tensors ({len(projections)} bfloat16) "
        f"across {shard_count} shards to {SCRIPT_DIR}"
    )


if __name__ == "__main__":
    main()
