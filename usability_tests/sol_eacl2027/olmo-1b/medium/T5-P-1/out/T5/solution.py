#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a sharded float32 checkpoint."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024
STANDALONE_THRESHOLD = MAX_SHARD_BYTES // 2


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def adapter_pairs(adapter):
    suffix_a = ".lora_A.weight"
    prefix = "base_model.model."
    pairs = {}

    for name in adapter:
        if not name.endswith(suffix_a):
            continue
        stem = name[: -len(suffix_a)]
        b_name = stem + ".lora_B.weight"
        assert b_name in adapter, f"missing LoRA B tensor for {name}"
        assert stem.startswith(prefix), f"unexpected PEFT tensor prefix: {name}"
        base_name = stem[len(prefix) :] + ".weight"
        assert base_name not in pairs, f"duplicate adapter target: {base_name}"
        pairs[base_name] = (name, b_name)

    expected_adapter_names = {
        adapter_name
        for pair in pairs.values()
        for adapter_name in pair
    }
    assert expected_adapter_names == set(adapter), (
        "adapter contains unpaired or unexpected tensors: "
        f"{sorted(set(adapter) - expected_adapter_names)}"
    )
    return pairs


def plan_shards(keys, sizes):
    """Greedily pack normal tensors; keep very large tensors standalone."""
    shards = []
    current = []
    current_size = 0

    for key in keys:
        size = sizes[key]
        assert size <= MAX_SHARD_BYTES, (
            f"tensor {key} is {size} bytes, above the shard limit"
        )
        if size > STANDALONE_THRESHOLD:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([key])
        else:
            if current and current_size + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = []
                current_size = 0
            current.append(key)
            current_size += size

    if current:
        shards.append(current)
    return shards


def main():
    config = load_json(LORA_DIR / "adapter_config.json")
    rank = config["r"]
    alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    assert rank > 0, "LoRA rank must be positive"
    scale = float(alpha) / float(rank)

    adapter = load_file(str(LORA_DIR / "adapter_model.safetensors"), device="cpu")
    pairs = adapter_pairs(adapter)
    assert len(pairs) == 32, f"expected 32 adapter pairs, found {len(pairs)}"

    base_index = load_json(BASE_DIR / "model.safetensors.index.json")
    weight_map = base_index["weight_map"]
    base_keys = sorted(weight_map)
    assert len(base_keys) == 114, f"expected 114 base tensors, found {len(base_keys)}"
    assert set(pairs) <= set(base_keys), (
        f"adapter targets absent from base: {sorted(set(pairs) - set(base_keys))}"
    )

    source_files = sorted(set(weight_map.values()))
    keys_by_source = {
        filename: [key for key in base_keys if weight_map[key] == filename]
        for filename in source_files
    }
    sizes = {}
    shapes = {}
    dtypes = {}
    merged = {}

    # This full preflight computes every merge before any output file is written.
    for filename in source_files:
        with safe_open(str(BASE_DIR / filename), framework="pt", device="cpu") as source:
            actual_keys = set(source.keys())
            expected_keys = set(keys_by_source[filename])
            assert actual_keys == expected_keys, (
                f"index contents disagree with {filename}: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
            for key in keys_by_source[filename]:
                tensor = source.get_tensor(key)
                shapes[key] = tuple(tensor.shape)
                dtypes[key] = tensor.dtype
                sizes[key] = tensor.numel() * tensor.element_size()
                assert tensor.dtype == torch.float32, f"base tensor {key} is not float32"

                if key in pairs:
                    a_name, b_name = pairs[key]
                    a = adapter[a_name]
                    b = adapter[b_name]
                    assert a.dtype == torch.float32 and b.dtype == torch.float32, (
                        f"adapter tensors for {key} are not float32"
                    )
                    assert a.ndim == 2 and b.ndim == 2, f"non-matrix LoRA factors for {key}"
                    assert a.shape[0] == rank, f"LoRA A rank mismatch for {key}: {a.shape}"
                    assert b.shape[1] == rank, f"LoRA B rank mismatch for {key}: {b.shape}"
                    delta = torch.matmul(b, a)
                    if fan_in_fan_out:
                        delta = delta.transpose(0, 1)
                    assert delta.shape == tensor.shape, (
                        f"LoRA delta/base shape mismatch for {key}: "
                        f"{tuple(delta.shape)} vs {tuple(tensor.shape)}"
                    )
                    merged[key] = (tensor + delta * scale).contiguous()

    # Required checks, deliberately all before writing checkpoint files.
    assert len(merged) == 32, f"expected 32 merged tensors, produced {len(merged)}"
    assert not any("lora_" in key for key in base_keys), "LoRA tensor leaked into output names"
    probe = "model.layers.0.self_attn.q_proj.weight"
    assert shapes.get(probe) == (2048, 2048), (
        f"{probe} has shape {shapes.get(probe)}, expected (2048, 2048)"
    )
    assert len(base_keys) == 114, f"output would contain {len(base_keys)} tensors, expected 114"
    assert set(shapes) == set(base_keys), "not every indexed base tensor was inspected"
    assert all(dtypes[key] == torch.float32 for key in base_keys), "output dtype would not be float32"

    shards = plan_shards(base_keys, sizes)
    assert sum(len(shard) for shard in shards) == 114, "shard plan lost tensors"
    assert all(sum(sizes[key] for key in shard) <= MAX_SHARD_BYTES for shard in shards), (
        "shard plan exceeds 512 MiB"
    )
    for large_name in ("model.embed_tokens.weight", "lm_head.weight"):
        assert any(shard == [large_name] for shard in shards), (
            f"required large tensor is not standalone: {large_name}"
        )

    # Remove only checkpoint artifacts from an earlier invocation, if present.
    for old_shard in OUT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    old_index = OUT_DIR / "model.safetensors.index.json"
    if old_index.exists():
        old_index.unlink()

    output_weight_map = {}
    shard_count = len(shards)
    with ExitStack() as stack:
        sources = {
            filename: stack.enter_context(
                safe_open(str(BASE_DIR / filename), framework="pt", device="cpu")
            )
            for filename in source_files
        }
        for number, shard_keys in enumerate(shards, start=1):
            shard_name = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {}
            for key in shard_keys:
                tensors[key] = (
                    merged[key]
                    if key in merged
                    else sources[weight_map[key]].get_tensor(key)
                )
                output_weight_map[key] = shard_name
            save_file(tensors, str(OUT_DIR / shard_name), metadata={"format": "pt"})

    output_index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": output_weight_map,
    }
    with old_index.open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Merged {len(merged)} LoRA pairs into {len(base_keys)} tensors; "
        f"wrote {shard_count} shards to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
