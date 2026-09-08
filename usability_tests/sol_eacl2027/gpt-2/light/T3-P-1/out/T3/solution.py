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


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name, tensor in tensors.items():
        size = tensor_bytes(tensor)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            assert len(current) == 1, f"oversized tensor {name} was not isolated"
            shards.append(current)
            current = {}
            current_bytes = 0

    if current:
        shards.append(current)
    return shards


def main() -> None:
    source = load_file(INPUT, device="cpu")
    result: dict[str, torch.Tensor] = {}
    projection_names: set[str] = set()

    for name, tensor in source.items():
        if BUFFER.fullmatch(name):
            continue
        if PROJECTION.fullmatch(name):
            result[name] = tensor.to(torch.bfloat16)
            projection_names.add(name)
        else:
            result[name] = tensor.to(torch.float32)

    # Required pre-write checks.
    bf16_names = {name for name, tensor in result.items() if tensor.dtype == torch.bfloat16}
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert bf16_names == projection_names, "a non-projection tensor is bfloat16"
    assert result["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert result["wte.weight"].dtype == torch.float32
    assert len(result) == 148, f"expected 148 output tensors, got {len(result)}"
    assert len(projection_names) == 48, f"expected 48 projections, got {len(projection_names)}"
    assert all(t.dtype == torch.float32 for n, t in result.items() if n not in projection_names)

    shards = make_shards(result)
    total = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total:05d}.safetensors"
        save_file(shard, OUTPUT / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in result.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
