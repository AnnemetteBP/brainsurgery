#!/usr/bin/env python3
"""Create the mixed-precision, sharded T3 checkpoint."""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
MAX_SHARD_BYTES = 64 * 1024 * 1024
PROJECTION = re.compile(
    r"h\.(?:[0-9]|1[01])\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight"
)
BUFFER = re.compile(r"h\.(?:[0-9]|1[01])\.attn\.bias")


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    source = load_file(INPUT, device="cpu")
    result: dict[str, torch.Tensor] = {}

    for name, tensor in source.items():
        if BUFFER.fullmatch(name):
            continue
        dtype = torch.bfloat16 if PROJECTION.fullmatch(name) else torch.float32
        result[name] = tensor.to(dtype=dtype).contiguous()

    # Mandatory checks are deliberately completed before the first output write.
    bf16_names = {name for name, tensor in result.items() if tensor.dtype == torch.bfloat16}
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert result["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert result["wte.weight"].dtype == torch.float32
    assert len(result) == 148, f"expected 148 output tensors, got {len(result)}"
    assert bf16_names == {name for name in result if PROJECTION.fullmatch(name)}
    assert all(t.dtype == torch.float32 for n, t in result.items() if n not in bf16_names)

    # Greedy packing in stable key order. An oversized tensor forms a shard alone.
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    for name in sorted(result):
        tensor = result[name]
        size = tensor_bytes(tensor)
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_size = 0
        if size > MAX_SHARD_BYTES:
            assert not current
            shards.append({name: tensor})
        else:
            current[name] = tensor
            current_size += size
    if current:
        shards.append(current)

    count = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{count:05d}.safetensors"
        save_file(shard, OUTPUT / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in result.values())},
        "weight_map": weight_map,
    }
    (OUTPUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
