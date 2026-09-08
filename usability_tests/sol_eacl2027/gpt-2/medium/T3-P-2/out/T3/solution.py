#!/usr/bin/env python3
"""Export GPT-2 with mixed precision into size-limited safetensors shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 64 * 1024 * 1024


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


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

    source = load_file(str(INPUT), device="cpu")
    missing_projections = projection_names.difference(source)
    missing_buffers = buffer_names.difference(source)
    if missing_projections:
        raise RuntimeError(f"missing projection tensors: {sorted(missing_projections)}")
    if missing_buffers:
        raise RuntimeError(f"missing buffers: {sorted(missing_buffers)}")

    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if name in buffer_names:
            continue
        dtype = torch.bfloat16 if name in projection_names else torch.float32
        tensors[name] = tensor.to(dtype=dtype)

    # All task-required checks happen before any checkpoint output is written.
    bfloat16_names = {
        name for name, tensor in tensors.items() if tensor.dtype == torch.bfloat16
    }
    if len(bfloat16_names) != 48:
        raise RuntimeError(f"expected 48 bfloat16 tensors, found {len(bfloat16_names)}")
    if bfloat16_names != projection_names:
        unexpected = sorted(bfloat16_names.difference(projection_names))
        missing = sorted(projection_names.difference(bfloat16_names))
        raise RuntimeError(
            f"incorrect bfloat16 tensor set; unexpected={unexpected}, missing={missing}"
        )
    if tensors["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        raise RuntimeError("h.0.attn.c_attn.weight is not bfloat16")
    if tensors["wte.weight"].dtype != torch.float32:
        raise RuntimeError("wte.weight is not float32")
    if len(tensors) != 148:
        raise RuntimeError(f"expected 148 output tensors, found {len(tensors)}")
    non_bfloat_dtypes = {
        name: str(tensor.dtype)
        for name, tensor in tensors.items()
        if name not in projection_names and tensor.dtype != torch.float32
    }
    if non_bfloat_dtypes:
        raise RuntimeError(f"non-projection tensors are not float32: {non_bfloat_dtypes}")

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_nbytes(tensor)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
    if current:
        shards.append(current)

    for shard in shards:
        shard_bytes = sum(tensor_nbytes(tensor) for tensor in shard.values())
        if shard_bytes > MAX_SHARD_BYTES and len(shard) != 1:
            raise RuntimeError(
                f"oversized shard has {len(shard)} tensors and {shard_bytes} bytes"
            )

    existing = list(OUTPUT_DIR.glob("model-*-of-*.safetensors"))
    if INDEX_PATH.exists() or existing:
        raise FileExistsError("checkpoint output already exists; refusing to overwrite it")

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, str(OUTPUT_DIR / filename))
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {
            "total_size": sum(tensor_nbytes(tensor) for tensor in tensors.values())
        },
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(tensors)} tensors in {shard_count} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
