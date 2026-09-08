#!/usr/bin/env python3
"""Export the GPT-2 state dict as a mixed-precision sharded checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
INDEX = OUTPUT / "model.safetensors.index.json"
MAX_SHARD_BYTES = 64 * 1024 * 1024


def projection_names() -> set[str]:
    suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    return {f"h.{layer}.{suffix}" for layer in range(12) for suffix in suffixes}


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(
    tensors: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    """Greedily pack sorted tensors, leaving any oversized tensor alone."""
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


def main() -> None:
    source = load_file(INPUT, device="cpu")
    projections = projection_names()
    buffers = {f"h.{layer}.attn.bias" for layer in range(12)}

    assert len(source) == 160, f"expected 160 input tensors, got {len(source)}"
    assert projections <= source.keys(), "one or more projection matrices are missing"
    assert buffers <= source.keys(), "one or more causal-mask buffers are missing"

    output = {
        name: (tensor.to(torch.bfloat16) if name in projections else tensor.to(torch.float32))
        for name, tensor in source.items()
        if name not in buffers
    }

    # Required pre-write checks, plus exact-set checks to prevent overmatching.
    bf16_names = {name for name, tensor in output.items() if tensor.dtype == torch.bfloat16}
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert bf16_names == projections, "bfloat16 tensor set differs from projection set"
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert output["wte.weight"].dtype == torch.float32
    assert len(output) == 148, f"expected 148 output tensors, got {len(output)}"
    assert set(output) == set(source) - buffers, "output key set is incorrect"
    assert all(
        tensor.dtype == torch.float32
        for name, tensor in output.items()
        if name not in projections
    ), "a non-projection tensor is not float32"

    shards = make_shards(output)
    for shard in shards:
        size = sum(tensor_bytes(tensor) for tensor in shard.values())
        assert size <= MAX_SHARD_BYTES or len(shard) == 1, "invalid oversized shard"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    total = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total:05d}.safetensors"
        save_file(shard, OUTPUT / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in output.values())},
        "weight_map": dict(sorted(weight_map.items())),
    }
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} tensors across {total} shards")


if __name__ == "__main__":
    main()
