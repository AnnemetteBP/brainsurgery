#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a base safetensors checkpoint and shard it."""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
DEDICATED_SHARD_TENSORS = {
    "gpt_neox.embed_in.weight",
    "embed_out.weight",
}


def tensor_nbytes(tensor_slice) -> int:
    """Return tensor-data bytes from safetensors metadata without loading it."""
    bytes_per_element = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
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
    dtype = tensor_slice.get_dtype()
    if dtype not in bytes_per_element:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}")
    return math.prod(tensor_slice.get_shape()) * bytes_per_element[dtype]


def adapter_module_name(adapter_key: str, suffix: str) -> str:
    prefix = "base_model.model."
    if not adapter_key.startswith(prefix) or not adapter_key.endswith(suffix):
        raise ValueError(f"Unexpected adapter tensor name: {adapter_key}")
    return adapter_key[len(prefix) : -len(suffix)]


def build_adapter_pairs(adapter_keys: list[str]) -> dict[str, tuple[str, str]]:
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}

    for key in adapter_keys:
        if key.endswith(a_suffix):
            module = adapter_module_name(key, a_suffix)
            if module in a_keys:
                raise ValueError(f"Duplicate LoRA A tensor for {module}")
            a_keys[module] = key
        elif key.endswith(b_suffix):
            module = adapter_module_name(key, b_suffix)
            if module in b_keys:
                raise ValueError(f"Duplicate LoRA B tensor for {module}")
            b_keys[module] = key
        else:
            raise ValueError(f"Unexpected non-LoRA tensor in adapter: {key}")

    if set(a_keys) != set(b_keys):
        missing_a = sorted(set(b_keys) - set(a_keys))
        missing_b = sorted(set(a_keys) - set(b_keys))
        raise ValueError(f"Unpaired LoRA tensors: missing A={missing_a}, missing B={missing_b}")

    return {
        f"{module}.weight": (a_keys[module], b_keys[module])
        for module in sorted(a_keys)
    }


def make_shard_plan(base_file, base_keys: list[str]) -> tuple[list[list[str]], int]:
    """Greedily pack tensors, while placing designated embeddings alone."""
    sizes = {key: tensor_nbytes(base_file.get_slice(key)) for key in base_keys}
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for key in base_keys:
        size = sizes[key]
        if key in DEDICATED_SHARD_TENSORS:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([key])
            continue

        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([key])
        elif current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = [key]
            current_bytes = size
        else:
            current.append(key)
            current_bytes += size

    if current:
        shards.append(current)

    for shard in shards:
        shard_bytes = sum(sizes[key] for key in shard)
        if shard_bytes > MAX_SHARD_BYTES and len(shard) != 1:
            raise AssertionError(f"Packed shard exceeds size limit: {shard_bytes} bytes")
    return shards, sum(sizes.values())


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    if config.get("fan_in_fan_out") is not False:
        raise ValueError("This checkpoint requires fan_in_fan_out=false")
    scale = alpha / rank

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_file, safe_open(
        ADAPTER_PATH, framework="pt", device="cpu"
    ) as adapter_file:
        base_keys = sorted(base_file.keys())
        adapter_keys = sorted(adapter_file.keys())
        pairs = build_adapter_pairs(adapter_keys)

        # Required checks and all structural validation happen before any write.
        if len(pairs) != 16:
            raise AssertionError(f"Expected exactly 16 adapter pairs, found {len(pairs)}")
        if len(base_keys) != 244:
            raise AssertionError(f"Expected exactly 244 output tensors, found {len(base_keys)}")
        if any("lora_" in key for key in base_keys):
            raise AssertionError("Output key set contains a LoRA adapter tensor")

        shape_key = "gpt_neox.layers.0.attention.query_key_value.weight"
        shape = list(base_file.get_slice(shape_key).get_shape())
        if shape != [6144, 2048]:
            raise AssertionError(f"Unexpected shape for {shape_key}: {shape}")

        for base_key, (a_key, b_key) in pairs.items():
            if base_key not in base_keys:
                raise KeyError(f"Adapter target is absent from base checkpoint: {base_key}")
            base_shape = list(base_file.get_slice(base_key).get_shape())
            a_shape = list(adapter_file.get_slice(a_key).get_shape())
            b_shape = list(adapter_file.get_slice(b_key).get_shape())
            if a_shape[0] != rank or b_shape[1] != rank:
                raise AssertionError(
                    f"Rank mismatch for {base_key}: base={base_shape}, A={a_shape}, B={b_shape}"
                )
            expected_shape = [b_shape[0], a_shape[1]]
            if base_shape != expected_shape:
                raise AssertionError(
                    f"Shape mismatch for {base_key}: base={base_shape}, B@A={expected_shape}"
                )

        missing_dedicated = DEDICATED_SHARD_TENSORS - set(base_keys)
        if missing_dedicated:
            raise KeyError(f"Missing requested dedicated-shard tensors: {sorted(missing_dedicated)}")

        shard_plan, total_size = make_shard_plan(base_file, base_keys)
        shard_count = len(shard_plan)
        weight_map: dict[str, str] = {}

        # Remove only stale files owned by this script, after validation succeeds.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
            old_shard.unlink()
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()

        for shard_number, shard_keys in enumerate(shard_plan, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for key in shard_keys:
                base_tensor = base_file.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = adapter_file.get_tensor(a_key).to(dtype=torch.float32)
                    b = adapter_file.get_tensor(b_key).to(dtype=torch.float32)
                    merged = torch.addmm(
                        base_tensor.to(dtype=torch.float32), b, a, beta=1.0, alpha=scale
                    )
                    tensors[key] = merged.to(dtype=base_tensor.dtype).contiguous()
                else:
                    tensors[key] = base_tensor.contiguous()
                weight_map[key] = filename

            save_file(tensors, OUTPUT_DIR / filename, metadata={"format": "pt"})

    if len(weight_map) != 244 or set(weight_map) != set(base_keys):
        raise AssertionError("Internal error: index weight map does not match output tensors")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as index_file:
        json.dump(index, index_file, indent=2, sort_keys=True)
        index_file.write("\n")

    print(
        f"Merged {len(pairs)} LoRA pairs into {len(base_keys)} tensors; "
        f"wrote {shard_count} shards and {INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
