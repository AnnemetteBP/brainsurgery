#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a sharded safetensors checkpoint."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_DIR = Path("inputs/base")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 536_870_912
EXPECTED_TENSORS = 114
EXPECTED_PAIRS = 32
FORCE_SINGLETON = {"model.embed_tokens.weight", "lm_head.weight"}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    config = read_json(CONFIG_PATH)
    rank = config["r"]
    alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    assert isinstance(rank, int) and rank > 0, f"invalid LoRA rank: {rank!r}"
    assert isinstance(alpha, (int, float)), f"invalid LoRA alpha: {alpha!r}"
    assert fan_in_fan_out is False, "this checkpoint requires fan_in_fan_out=false"
    scale = float(alpha) / rank

    base_index = read_json(BASE_DIR / INDEX_NAME)
    base_weight_map = base_index["weight_map"]
    base_names = set(base_weight_map)

    with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter:
        adapter_names = set(adapter.keys())
        suffix_a = ".lora_A.weight"
        suffix_b = ".lora_B.weight"
        stems_a = {name[: -len(suffix_a)] for name in adapter_names if name.endswith(suffix_a)}
        stems_b = {name[: -len(suffix_b)] for name in adapter_names if name.endswith(suffix_b)}
        expected_adapter_names = {
            stem + suffix for stem in stems_a for suffix in (suffix_a, suffix_b)
        }

        assert stems_a == stems_b, "unpaired LoRA A/B tensors"
        assert adapter_names == expected_adapter_names, "unexpected non-pair tensor in adapter"
        assert len(stems_a) == EXPECTED_PAIRS, (
            f"expected {EXPECTED_PAIRS} adapter pairs, found {len(stems_a)}"
        )

        pair_for_base = {}
        prefix = "base_model.model."
        for stem in stems_a:
            assert stem.startswith(prefix), f"unexpected adapter prefix: {stem}"
            base_name = stem[len(prefix) :] + ".weight"
            assert base_name in base_names, f"adapter target absent from base: {base_name}"
            assert base_name not in pair_for_base, f"duplicate adapter target: {base_name}"

            shape_a = adapter.get_slice(stem + suffix_a).get_shape()
            shape_b = adapter.get_slice(stem + suffix_b).get_shape()
            assert shape_a == [rank, 2048], f"unexpected A shape for {stem}: {shape_a}"
            assert shape_b == [2048, rank], f"unexpected B shape for {stem}: {shape_b}"
            pair_for_base[base_name] = (stem + suffix_a, stem + suffix_b)

    # Required output checks are deliberately completed before any shard is written.
    assert len(pair_for_base) == EXPECTED_PAIRS, "did not map exactly 32 adapter pairs"
    assert not any("lora_" in name for name in base_names), "LoRA tensor in output key set"
    assert len(base_names) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} output tensors, found {len(base_names)}"
    )

    shard_paths = sorted(set(base_weight_map.values()))
    handles = {
        filename: safe_open(BASE_DIR / filename, framework="pt", device="cpu")
        for filename in shard_paths
    }
    try:
        q0 = handles[base_weight_map["model.layers.0.self_attn.q_proj.weight"]].get_slice(
            "model.layers.0.self_attn.q_proj.weight"
        )
        assert q0.get_shape() == [2048, 2048], (
            f"unexpected q_proj shape: {q0.get_shape()}"
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stale = list(OUTPUT_DIR.glob("model-*.safetensors")) + [OUTPUT_DIR / INDEX_NAME]
        for path in stale:
            if path.exists():
                path.unlink()

        temporary_shards = []
        temporary_map = {}
        current = {}
        current_bytes = 0

        def flush() -> None:
            nonlocal current, current_bytes
            if not current:
                return
            number = len(temporary_shards) + 1
            filename = f"model-{number:05d}.safetensors"
            save_file(current, OUTPUT_DIR / filename)
            temporary_shards.append(filename)
            for tensor_name in current:
                temporary_map[tensor_name] = filename
            current = {}
            current_bytes = 0

        with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter:
            for name in sorted(base_names):
                tensor = handles[base_weight_map[name]].get_tensor(name)
                assert tensor.dtype == torch.float32, f"base tensor is not float32: {name}"
                if name in pair_for_base:
                    name_a, name_b = pair_for_base[name]
                    a = adapter.get_tensor(name_a)
                    b = adapter.get_tensor(name_b)
                    assert a.dtype == torch.float32 and b.dtype == torch.float32, (
                        f"adapter tensors are not float32 for {name}"
                    )
                    delta = torch.mm(b, a)
                    if fan_in_fan_out:
                        delta = delta.T
                    assert delta.shape == tensor.shape, f"merge shape mismatch for {name}"
                    tensor = tensor + delta.mul(scale)

                size = tensor_nbytes(tensor)
                if name in FORCE_SINGLETON or size > MAX_SHARD_BYTES:
                    flush()
                    current = {name: tensor}
                    current_bytes = size
                    flush()
                    continue
                if current and current_bytes + size > MAX_SHARD_BYTES:
                    flush()
                current[name] = tensor
                current_bytes += size
            flush()

        shard_count = len(temporary_shards)
        final_map = {}
        for number, old_name in enumerate(temporary_shards, start=1):
            new_name = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            (OUTPUT_DIR / old_name).rename(OUTPUT_DIR / new_name)
            for tensor_name, mapped_name in temporary_map.items():
                if mapped_name == old_name:
                    final_map[tensor_name] = new_name

        total_size = sum(
            tensor_nbytes(handles[base_weight_map[name]].get_tensor(name))
            for name in base_names
        )
        output_index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(final_map.items())),
        }
        assert set(final_map) == base_names, "output index does not map every base tensor"
        with (OUTPUT_DIR / INDEX_NAME).open("w", encoding="utf-8") as stream:
            json.dump(output_index, stream, indent=2)
            stream.write("\n")

        print(
            f"Merged {len(pair_for_base)} LoRA pairs into {len(base_names)} tensors "
            f"and wrote {shard_count} shards to {OUTPUT_DIR}"
        )
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)


if __name__ == "__main__":
    main()
