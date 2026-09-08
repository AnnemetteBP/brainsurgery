#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection-only bfloat16 sharding."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024


def projection_names() -> set[str]:
    names = set()
    for layer in range(16):
        prefix = f"model.layers.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{prefix}.self_attn.{projection}.weight")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"{prefix}.mlp.{projection}.weight")
    return names


def main() -> None:
    with (BASE / INDEX_NAME).open() as stream:
        source_index = json.load(stream)
    source_map = source_index["weight_map"]
    names = sorted(source_map)
    projections = projection_names()

    if len(projections) != 112 or not projections.issubset(source_map):
        raise RuntimeError("the input does not contain exactly the expected 112 projections")

    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                safe_open(BASE / filename, framework="pt", device="cpu")
            )
            for filename in sorted(set(source_map.values()))
        }

        # Inspect every input and build the complete output plan before writing.
        tensor_sizes = {}
        output_dtypes = {}
        for name in names:
            tensor_slice = readers[source_map[name]].get_slice(name)
            if tensor_slice.get_dtype() != "F32":
                raise RuntimeError(f"input tensor {name} is not float32")
            numel = 1
            for dimension in tensor_slice.get_shape():
                numel *= dimension
            output_dtypes[name] = torch.bfloat16 if name in projections else torch.float32
            tensor_sizes[name] = numel * (2 if name in projections else 4)

        # Required checks, deliberately completed before the first save_file call.
        if sum(dtype == torch.bfloat16 for dtype in output_dtypes.values()) != 112:
            raise RuntimeError("output plan does not contain exactly 112 bfloat16 tensors")
        if output_dtypes.get("model.layers.0.self_attn.q_proj.weight") != torch.bfloat16:
            raise RuntimeError("layer 0 q_proj is not planned as bfloat16")
        if output_dtypes.get("model.embed_tokens.weight") != torch.float32:
            raise RuntimeError("embedding is not planned as float32")
        if len(output_dtypes) != 114:
            raise RuntimeError(f"output plan contains {len(output_dtypes)} tensors, expected 114")

        shards = []
        current = []
        current_size = 0
        for name in names:
            size = tensor_sizes[name]
            if size > MAX_SHARD_BYTES:
                if current:
                    shards.append(current)
                    current, current_size = [], 0
                shards.append([name])
            else:
                if current and current_size + size > MAX_SHARD_BYTES:
                    shards.append(current)
                    current, current_size = [], 0
                current.append(name)
                current_size += size
        if current:
            shards.append(current)

        weight_map = {}
        shard_count = len(shards)
        for number, shard_names in enumerate(shards, start=1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {}
            for name in shard_names:
                tensor = readers[source_map[name]].get_tensor(name)
                tensors[name] = tensor.to(output_dtypes[name])
                weight_map[name] = filename
            save_file(tensors, OUTPUT / filename, metadata={"format": "pt"})

    result_index = {
        "metadata": {"total_size": sum(tensor_sizes.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT / INDEX_NAME).open("w") as stream:
        json.dump(result_index, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"Wrote {len(names)} tensors in {len(shards)} shards to {OUTPUT}")


if __name__ == "__main__":
    main()
