import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
INDEX = OUTPUT / "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJECTION_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    projection_names = {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in PROJECTION_SUFFIXES
    }
    buffer_names = {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in BUFFER_SUFFIXES
    }

    source = load_file(INPUT, device="cpu")
    if len(source) != 244:
        raise RuntimeError(f"Expected 244 input tensors, found {len(source)}")
    missing_projections = projection_names.difference(source)
    missing_buffers = buffer_names.difference(source)
    if missing_projections or missing_buffers:
        raise RuntimeError(
            f"Missing expected tensors: projections={sorted(missing_projections)}, "
            f"buffers={sorted(missing_buffers)}"
        )

    output_tensors = {}
    for name, tensor in source.items():
        if name in buffer_names:
            continue
        dtype = torch.bfloat16 if name in projection_names else torch.float32
        output_tensors[name] = tensor.to(dtype=dtype)

    # Required pre-write checks.
    bf16_names = {
        name for name, tensor in output_tensors.items()
        if tensor.dtype == torch.bfloat16
    }
    if len(bf16_names) != 64:
        raise RuntimeError(f"Expected exactly 64 bfloat16 tensors, found {len(bf16_names)}")
    if bf16_names != projection_names:
        raise RuntimeError("The bfloat16 tensor set is not exactly the projection matrix set")
    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if output_tensors[qkv0].dtype != torch.bfloat16:
        raise RuntimeError(f"{qkv0} is not bfloat16")
    if output_tensors["gpt_neox.embed_in.weight"].dtype != torch.float32:
        raise RuntimeError("gpt_neox.embed_in.weight is not float32")
    if len(output_tensors) != 196:
        raise RuntimeError(f"Expected exactly 196 output tensors, found {len(output_tensors)}")
    non_float32 = {
        name: str(tensor.dtype)
        for name, tensor in output_tensors.items()
        if name not in projection_names and tensor.dtype != torch.float32
    }
    if non_float32:
        raise RuntimeError(f"Non-projection tensors are not float32: {non_float32}")

    # Greedily pack tensors in stable name order. A tensor over the limit is
    # permitted only as the sole member of its shard.
    shards = []
    current = {}
    current_bytes = 0
    for name in sorted(output_tensors):
        tensor = output_tensors[name]
        size = tensor_bytes(tensor)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            if len(current) != 1:
                raise RuntimeError(f"Oversized tensor {name} was not isolated")
            shards.append(current)
            current = {}
            current_bytes = 0
    if current:
        shards.append(current)

    total_shards = len(shards)
    weight_map = {}
    for shard_number, shard in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, OUTPUT / filename)
        for name in shard:
            weight_map[name] = filename

    total_size = sum(tensor_bytes(tensor) for tensor in output_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
