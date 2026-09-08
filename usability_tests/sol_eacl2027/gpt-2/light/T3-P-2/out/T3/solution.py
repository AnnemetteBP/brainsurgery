import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
MAX_SHARD_BYTES = 64 * 1024 * 1024

PROJECTION = re.compile(
    r"^h\.(?:[0-9]|1[01])\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight$"
)
BUFFER = re.compile(r"^h\.(?:[0-9]|1[01])\.attn\.bias$")


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    source = load_file(str(INPUT), device="cpu")
    tensors: dict[str, torch.Tensor] = {}

    for name, tensor in source.items():
        if BUFFER.fullmatch(name):
            continue
        dtype = torch.bfloat16 if PROJECTION.fullmatch(name) else torch.float32
        tensors[name] = tensor.to(dtype).contiguous()

    bf16_names = [name for name, tensor in tensors.items() if tensor.dtype == torch.bfloat16]
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert tensors["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert tensors["wte.weight"].dtype == torch.float32
    assert len(tensors) == 148, f"expected 148 output tensors, got {len(tensors)}"

    # Greedily pack tensors in safetensors key order. An individually oversized
    # tensor is valid, but no other tensor may share its shard.
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name, tensor in tensors.items():
        size = tensor_bytes(tensor)
        if current and (size > MAX_SHARD_BYTES or current_bytes + size > MAX_SHARD_BYTES):
            shards.append(current)
            current = {}
            current_bytes = 0
        if size > MAX_SHARD_BYTES:
            shards.append({name: tensor})
        else:
            current[name] = tensor
            current_bytes += size
    if current:
        shards.append(current)

    for shard in shards:
        shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
        assert shard_size <= MAX_SHARD_BYTES or len(shard) == 1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, str(OUTPUT / filename))
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in tensors.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
