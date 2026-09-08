#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


BASE_DIR = Path("inputs/base")
LORA_DIR = Path("inputs/lora")
OUT_DIR = Path("out/T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024


def tensor_nbytes(path: Path, name: str) -> tuple[int, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        shape = tuple(view.get_shape())
        dtype = handle.get_tensor(name).dtype
    return int(torch.tensor([], dtype=dtype).element_size()) * int(torch.Size(shape).numel()), shape


def main() -> None:
    with (BASE_DIR / "model.safetensors.index.json").open() as stream:
        base_index = json.load(stream)
    with (LORA_DIR / "adapter_config.json").open() as stream:
        config = json.load(stream)

    weight_map = base_index["weight_map"]
    base_names = list(weight_map)
    adapter = load_file(LORA_DIR / "adapter_model.safetensors", device="cpu")

    suffix_a = ".lora_A.weight"
    prefix = "base_model.model."
    pairs = {}
    for adapter_name in adapter:
        if not adapter_name.endswith(suffix_a):
            continue
        stem = adapter_name[: -len(suffix_a)]
        b_name = stem + ".lora_B.weight"
        assert b_name in adapter, f"missing B factor for {adapter_name}"
        assert stem.startswith(prefix), f"unexpected adapter prefix: {adapter_name}"
        base_name = stem[len(prefix) :] + ".weight"
        assert base_name in weight_map, f"adapter target absent from base: {base_name}"
        pairs[base_name] = (adapter_name, b_name)

    # Required checks are deliberately completed before any checkpoint is written.
    assert len(pairs) == 32, f"expected 32 adapter pairs, found {len(pairs)}"
    assert len(base_names) == 114, f"expected 114 output tensors, found {len(base_names)}"
    assert not any("lora_" in name for name in base_names), "LoRA tensor leaked into output names"

    sizes = {}
    shapes = {}
    source_paths = {}
    for name in base_names:
        source = BASE_DIR / weight_map[name]
        source_paths[name] = source
        sizes[name], shapes[name] = tensor_nbytes(source, name)
    check_name = "model.layers.0.self_attn.q_proj.weight"
    assert shapes[check_name] == (2048, 2048), f"wrong shape for {check_name}: {shapes[check_name]}"

    # Greedy packing in index order, allowing only an individually oversized tensor
    # to exceed the cap (the generic rule, though no supplied tensor exceeds it).
    groups = []
    current = []
    current_bytes = 0
    for name in base_names:
        size = sizes[name]
        if current and current_bytes + size > MAX_SHARD_BYTES:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            assert len(current) == 1
            groups.append(current)
            current = []
            current_bytes = 0
    if current:
        groups.append(current)

    scale = float(config["lora_alpha"]) / int(config["r"])
    transpose = bool(config.get("fan_in_fan_out", False))
    output_map = {}
    total = len(groups)
    for number, names in enumerate(groups, 1):
        shard_name = f"model-{number:05d}-of-{total:05d}.safetensors"
        tensors = {}
        for name in names:
            with safe_open(source_paths[name], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            if name in pairs:
                a_name, b_name = pairs[name]
                a = adapter[a_name].to(torch.float32)
                b = adapter[b_name].to(torch.float32)
                delta = b @ a
                if transpose:
                    delta = delta.T
                assert value.dtype == torch.float32
                assert delta.shape == value.shape
                value = value + scale * delta
            tensors[name] = value.contiguous()
            output_map[name] = shard_name
        shard_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        assert shard_bytes <= MAX_SHARD_BYTES or (
            len(tensors) == 1 and next(iter(tensors.values())).numel()
            * next(iter(tensors.values())).element_size() > MAX_SHARD_BYTES
        ), f"shard {shard_name} exceeds size limit"
        save_file(tensors, OUT_DIR / shard_name)

    output_index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": output_map,
    }
    with (OUT_DIR / "model.safetensors.index.json").open("w") as stream:
        json.dump(output_index, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Merged {len(pairs)} LoRA pairs into {len(base_names)} tensors across {total} shards")


if __name__ == "__main__":
    main()
