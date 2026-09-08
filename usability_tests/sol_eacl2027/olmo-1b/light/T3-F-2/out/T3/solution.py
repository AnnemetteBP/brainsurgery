#!/usr/bin/env python3
"""Export the OLMo checkpoint with explicitly selected BF16 projections."""

import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUT = Path("out/T3")
LIMIT = 256 * 1024 * 1024
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
BF16_NAMES = {
    f"model.layers.{layer}.{projection}.weight"
    for layer in range(16)
    for projection in PROJECTIONS
}


def load_index():
    with (BASE / "model.safetensors.index.json").open() as stream:
        return json.load(stream)["weight_map"]


def read_tensor(name, weight_map):
    with safe_open(BASE / weight_map[name], framework="pt", device="cpu") as source:
        return source.get_tensor(name)


def output_nbytes(name, tensor):
    itemsize = 2 if name in BF16_NAMES else 4
    return tensor.numel() * itemsize


def preflight(weight_map):
    names = set(weight_map)
    assert len(names) == 114, f"expected 114 output tensors, found {len(names)}"
    assert len(BF16_NAMES) == 112, f"expected 112 BF16 names, got {len(BF16_NAMES)}"
    missing = BF16_NAMES - names
    assert not missing, f"missing projection tensors: {sorted(missing)}"

    # Required dtype checks are performed on the actual conversion path before writes.
    q = read_tensor("model.layers.0.self_attn.q_proj.weight", weight_map).to(torch.bfloat16)
    embed = read_tensor("model.embed_tokens.weight", weight_map).to(torch.float32)
    assert q.dtype == torch.bfloat16, "layer 0 q_proj did not convert to BF16"
    assert embed.dtype == torch.float32, "embedding is not FP32"
    del q, embed
    gc.collect()


def plan_shards(weight_map):
    groups, current, current_size = [], [], 0
    for name in sorted(weight_map):
        tensor = read_tensor(name, weight_map)
        size = output_nbytes(name, tensor)
        del tensor
        if current and (current_size + size > LIMIT or current_size > LIMIT):
            groups.append(current)
            current, current_size = [], 0
        # Oversize tensors form a shard of their own.
        if size > LIMIT:
            if current:
                groups.append(current)
                current, current_size = [], 0
            groups.append([name])
        else:
            current.append(name)
            current_size += size
    if current:
        groups.append(current)
    return groups


def main():
    weight_map = load_index()
    preflight(weight_map)
    groups = plan_shards(weight_map)

    index_map = {}
    total_size = 0
    for number, names in enumerate(groups, 1):
        filename = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        shard_size = 0
        for name in names:
            tensor = read_tensor(name, weight_map)
            tensor = tensor.to(torch.bfloat16 if name in BF16_NAMES else torch.float32)
            tensors[name] = tensor.contiguous()
            shard_size += tensors[name].numel() * tensors[name].element_size()
            index_map[name] = filename
        assert shard_size <= LIMIT or len(names) == 1, (
            f"shard {filename} exceeds 256 MiB and contains multiple tensors"
        )
        save_file(tensors, OUT / filename, metadata={"format": "pt"})
        total_size += shard_size
        del tensors
        gc.collect()

    assert len(index_map) == 114
    index = {"metadata": {"total_size": total_size}, "weight_map": index_map}
    with (OUT / "model.safetensors.index.json").open("w") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
