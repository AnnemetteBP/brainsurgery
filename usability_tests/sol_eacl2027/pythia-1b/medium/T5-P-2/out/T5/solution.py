#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a base checkpoint and export shards."""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = ROOT / "out/T5"
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
ISOLATED_TENSORS = {
    "gpt_neox.embed_in.weight",
    "embed_out.weight",
}

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def tensor_nbytes(tensor_slice) -> int:
    dtype = tensor_slice.get_dtype()
    if dtype not in DTYPE_BYTES:
        raise AssertionError(f"Unsupported safetensors dtype: {dtype}")
    return math.prod(tensor_slice.get_shape()) * DTYPE_BYTES[dtype]


def adapter_stem(key: str, suffix: str) -> str:
    prefix = "base_model.model."
    if not key.startswith(prefix) or not key.endswith(suffix):
        raise AssertionError(f"Unexpected adapter tensor name: {key}")
    return key[len(prefix) : -len(suffix)]


def preflight():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    alpha = config["lora_alpha"]
    if not isinstance(rank, int) or rank <= 0:
        raise AssertionError(f"Invalid LoRA rank: {rank!r}")
    if config.get("fan_in_fan_out") is not False:
        raise AssertionError("This script expects fan_in_fan_out=false")
    scale = float(alpha) / rank

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_file, safe_open(
        ADAPTER_PATH, framework="pt", device="cpu"
    ) as adapter_file:
        base_keys = sorted(base_file.keys())
        adapter_keys = set(adapter_file.keys())

        a_by_stem = {
            adapter_stem(key, ".lora_A.weight"): key
            for key in adapter_keys
            if key.endswith(".lora_A.weight")
        }
        b_by_stem = {
            adapter_stem(key, ".lora_B.weight"): key
            for key in adapter_keys
            if key.endswith(".lora_B.weight")
        }
        recognized = set(a_by_stem.values()) | set(b_by_stem.values())
        if recognized != adapter_keys:
            unexpected = sorted(adapter_keys - recognized)
            raise AssertionError(f"Unexpected adapter tensors: {unexpected}")
        if set(a_by_stem) != set(b_by_stem):
            missing_a = sorted(set(b_by_stem) - set(a_by_stem))
            missing_b = sorted(set(a_by_stem) - set(b_by_stem))
            raise AssertionError(
                f"Unpaired adapters; missing A={missing_a}, missing B={missing_b}"
            )
        if len(a_by_stem) != 16:
            raise AssertionError(
                f"Expected exactly 16 adapter pairs, found {len(a_by_stem)}"
            )

        pairs = {}
        for stem in sorted(a_by_stem):
            base_key = stem + ".weight"
            if base_key not in base_keys:
                raise AssertionError(f"Adapter target absent from base: {base_key}")
            a_shape = list(adapter_file.get_slice(a_by_stem[stem]).get_shape())
            b_shape = list(adapter_file.get_slice(b_by_stem[stem]).get_shape())
            base_shape = list(base_file.get_slice(base_key).get_shape())
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise AssertionError(f"LoRA factors for {stem} must be matrices")
            if a_shape[0] != rank or b_shape[1] != rank:
                raise AssertionError(
                    f"Rank mismatch for {stem}: A={a_shape}, B={b_shape}, r={rank}"
                )
            product_shape = [b_shape[0], a_shape[1]]
            if product_shape != base_shape:
                raise AssertionError(
                    f"Shape mismatch for {stem}: B@A={product_shape}, base={base_shape}"
                )
            pairs[base_key] = (a_by_stem[stem], b_by_stem[stem])

        # Required checks, all completed before creating checkpoint artifacts.
        if any("lora_" in name for name in base_keys):
            raise AssertionError("Output key set contains an adapter tensor")
        required_key = "gpt_neox.layers.0.attention.query_key_value.weight"
        required_shape = list(base_file.get_slice(required_key).get_shape())
        if required_shape != [6144, 2048]:
            raise AssertionError(
                f"Unexpected layer-0 query/key/value shape: {required_shape}"
            )
        if len(base_keys) != 244:
            raise AssertionError(f"Expected 244 output tensors, found {len(base_keys)}")

        sizes = {
            key: tensor_nbytes(base_file.get_slice(key)) for key in base_keys
        }

    return base_keys, pairs, scale, sizes


def plan_shards(base_keys, sizes):
    shards = []
    current = []
    current_size = 0

    def flush():
        nonlocal current, current_size
        if current:
            shards.append(current)
            current = []
            current_size = 0

    for key in base_keys:
        size = sizes[key]
        if key in ISOLATED_TENSORS or size > MAX_SHARD_BYTES:
            flush()
            shards.append([key])
            continue
        if current and current_size + size > MAX_SHARD_BYTES:
            flush()
        current.append(key)
        current_size += size
    flush()

    assigned = [key for shard in shards for key in shard]
    if assigned != base_keys or len(set(assigned)) != len(base_keys):
        raise AssertionError("Shard plan does not contain every base tensor exactly once")
    for shard in shards:
        size = sum(sizes[key] for key in shard)
        if size > MAX_SHARD_BYTES and len(shard) != 1:
            raise AssertionError(f"Oversized multi-tensor shard planned: {size} bytes")
    return shards


def export(base_keys, pairs, scale, sizes):
    shards = plan_shards(base_keys, sizes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_path.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    weight_map = {}
    total_shards = len(shards)
    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_file, safe_open(
        ADAPTER_PATH, framework="pt", device="cpu"
    ) as adapter_file:
        for shard_number, keys in enumerate(shards, start=1):
            filename = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
            tensors = {}
            for key in keys:
                base_tensor = base_file.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = adapter_file.get_tensor(a_key).to(torch.float32)
                    b = adapter_file.get_tensor(b_key).to(torch.float32)
                    update = torch.mm(b, a)
                    update.mul_(scale)
                    update.add_(base_tensor.to(torch.float32))
                    tensors[key] = update.to(base_tensor.dtype).contiguous()
                else:
                    tensors[key] = base_tensor
                weight_map[key] = filename
            save_file(tensors, OUTPUT_DIR / filename)

    if set(weight_map) != set(base_keys) or len(weight_map) != 244:
        raise AssertionError("Written weight map does not match the validated output keys")
    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    base_keys, pairs, scale, sizes = preflight()
    export(base_keys, pairs, scale, sizes)
    print(
        f"Merged {len(pairs)} LoRA pairs and exported {len(base_keys)} tensors "
        f"to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
