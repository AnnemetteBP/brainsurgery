#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into Pythia-1B and export safetensor shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
EXPECTED_TENSOR_COUNT = 244
EXPECTED_PAIR_COUNT = 16
ISOLATED_TENSORS = {"gpt_neox.embed_in.weight", "embed_out.weight"}


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def adapter_base_name(a_name: str) -> str:
    prefix = "base_model.model."
    suffix = ".lora_A.weight"
    if not a_name.startswith(prefix) or not a_name.endswith(suffix):
        raise AssertionError(f"unexpected LoRA A tensor name: {a_name}")
    return a_name[len(prefix) : -len(suffix)] + ".weight"


def plan_shards(state: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedily shard tensor data, while isolating the two embeddings."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in sorted(state):
        size = tensor_bytes(state[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        elif name in ISOLATED_TENSORS:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size

    if current:
        shards.append(current)

    for shard in shards:
        size = sum(tensor_bytes(state[name]) for name in shard)
        assert size <= MAX_SHARD_BYTES or len(shard) == 1, (
            f"shard is too large: {size} bytes for {shard}"
        )
    for name in ISOLATED_TENSORS:
        assert any(shard == [name] for shard in shards), (
            f"required isolated tensor was not placed alone: {name}"
        )
    return shards


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    fan_in_fan_out = bool(config["fan_in_fan_out"])
    assert rank > 0, f"invalid LoRA rank: {rank}"
    scale = alpha / rank

    state = load_file(BASE_PATH, device="cpu")
    adapter = load_file(ADAPTER_PATH, device="cpu")

    a_names = sorted(name for name in adapter if name.endswith(".lora_A.weight"))
    b_names = {name for name in adapter if name.endswith(".lora_B.weight")}
    assert len(adapter) == EXPECTED_PAIR_COUNT * 2, (
        f"expected only {EXPECTED_PAIR_COUNT * 2} adapter tensors, got {len(adapter)}"
    )
    assert len(a_names) == EXPECTED_PAIR_COUNT, (
        f"expected {EXPECTED_PAIR_COUNT} LoRA A tensors, got {len(a_names)}"
    )

    merge_specs: list[tuple[str, str, str]] = []
    expected_b_names: set[str] = set()
    for a_name in a_names:
        b_name = a_name.removesuffix(".lora_A.weight") + ".lora_B.weight"
        expected_b_names.add(b_name)
        assert b_name in adapter, f"missing LoRA B tensor for {a_name}"
        base_name = adapter_base_name(a_name)
        assert base_name in state, f"adapter target is absent from base: {base_name}"

        a = adapter[a_name]
        b = adapter[b_name]
        base = state[base_name]
        assert a.dtype == torch.float32 and b.dtype == torch.float32, (
            f"adapter factors must be float32: {a_name}, {b_name}"
        )
        assert a.ndim == 2 and b.ndim == 2 and a.shape[0] == rank, (
            f"invalid factor shapes: A={tuple(a.shape)}, B={tuple(b.shape)}, r={rank}"
        )
        assert b.shape[1] == rank, f"invalid B shape for rank {rank}: {tuple(b.shape)}"
        delta_shape = (a.shape[1], b.shape[0]) if fan_in_fan_out else (b.shape[0], a.shape[1])
        assert tuple(base.shape) == delta_shape, (
            f"base/delta shape mismatch for {base_name}: {tuple(base.shape)} vs {delta_shape}"
        )
        merge_specs.append((a_name, b_name, base_name))

    assert b_names == expected_b_names, "unpaired or unexpected LoRA B tensor(s) found"

    merged_count = 0
    for a_name, b_name, base_name in merge_specs:
        a = adapter[a_name]
        b = adapter[b_name]
        delta = b @ a
        if fan_in_fan_out:
            delta = delta.T
        base = state[base_name]
        state[base_name] = (base.float() + delta.mul(scale)).to(base.dtype)
        merged_count += 1

    # All task-mandated checks occur before the first output checkpoint write.
    assert merged_count == EXPECTED_PAIR_COUNT, (
        f"expected exactly {EXPECTED_PAIR_COUNT} merged pairs, got {merged_count}"
    )
    assert not any("lora_" in name for name in state), "LoRA tensor leaked into output"
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert tuple(state[probe].shape) == (6144, 2048), (
        f"wrong merged probe shape: {tuple(state[probe].shape)}"
    )
    assert len(state) == EXPECTED_TENSOR_COUNT, (
        f"expected {EXPECTED_TENSOR_COUNT} output tensors, got {len(state)}"
    )

    shards = plan_shards(state)
    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    for shard_number, names in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        for name in names:
            assert name not in weight_map, f"tensor assigned twice: {name}"
            weight_map[name] = filename

    assert set(weight_map) == set(state), "shard plan does not cover the output exactly"

    for shard_number, names in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        save_file({name: state[name].contiguous() for name in names}, OUTPUT_DIR / filename)

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in state.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Merged {merged_count} LoRA pairs; wrote {len(state)} tensors "
        f"across {shard_count} shards to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
