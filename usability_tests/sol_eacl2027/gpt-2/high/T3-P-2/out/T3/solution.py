import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 64 * 1024 * 1024


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name, tensor in sorted(tensors.items()):
        size = tensor_bytes(tensor)

        # Oversized tensors are permitted only as singleton shards.
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
            continue

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
    projection_names = {
        f"h.{layer}.{module}.weight"
        for layer in range(12)
        for module in (
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        )
    }
    buffer_names = {f"h.{layer}.attn.bias" for layer in range(12)}

    source = load_file(INPUT_PATH, device="cpu")
    missing_projections = projection_names - source.keys()
    missing_buffers = buffer_names - source.keys()
    assert not missing_projections, f"Missing projection tensors: {sorted(missing_projections)}"
    assert not missing_buffers, f"Missing buffer tensors: {sorted(missing_buffers)}"

    output: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if name in buffer_names:
            continue
        dtype = torch.bfloat16 if name in projection_names else torch.float32
        output[name] = tensor.to(dtype=dtype).contiguous()

    # Required checks, all performed before any checkpoint output is written.
    bfloat16_names = {name for name, tensor in output.items() if tensor.dtype == torch.bfloat16}
    assert len(bfloat16_names) == 48, f"Expected 48 bfloat16 tensors, got {len(bfloat16_names)}"
    assert bfloat16_names == projection_names, (
        f"Incorrect bfloat16 selection; missing={sorted(projection_names - bfloat16_names)}, "
        f"extra={sorted(bfloat16_names - projection_names)}"
    )
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert output["wte.weight"].dtype == torch.float32
    assert len(output) == 148, f"Expected 148 output tensors, got {len(output)}"
    non_float32 = {
        name: tensor.dtype
        for name, tensor in output.items()
        if name not in projection_names and tensor.dtype != torch.float32
    }
    assert not non_float32, f"Non-projection tensors are not float32: {non_float32}"
    assert output.keys() == source.keys() - buffer_names, "Output key set is incorrect"

    shards = make_shards(output)
    for shard in shards:
        size = sum(tensor_bytes(tensor) for tensor in shard.values())
        assert size <= MAX_SHARD_BYTES or (
            len(shard) == 1 and next(iter(shard.values())).numel() * next(iter(shard.values())).element_size() > MAX_SHARD_BYTES
        ), f"Invalid shard of {size} bytes containing {len(shard)} tensors"

    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for shard_number, shard in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    assert weight_map.keys() == output.keys(), "Index weight map does not cover the output exactly"
    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
