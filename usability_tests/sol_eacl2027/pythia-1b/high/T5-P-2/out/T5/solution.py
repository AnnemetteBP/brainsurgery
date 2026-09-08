#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA weights and export a dense sharded checkpoint."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_SIZE = 512 * 1024 * 1024
EXPECTED_TENSORS = 244
EXPECTED_PAIRS = 16
SPECIAL_SINGLETONS = {
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
        raise AssertionError(f"Unsupported safetensors dtype in base: {dtype}")
    elements = 1
    for dimension in tensor_slice.get_shape():
        elements *= dimension
    return elements * DTYPE_BYTES[dtype]


def base_name_from_adapter_a(adapter_name: str) -> str:
    prefix = "base_model.model."
    suffix = ".lora_A.weight"
    if not adapter_name.startswith(prefix) or not adapter_name.endswith(suffix):
        raise AssertionError(f"Unexpected LoRA A tensor name: {adapter_name}")
    return adapter_name[len(prefix) : -len(suffix)] + ".weight"


def preflight():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config.get("r")
    alpha = config.get("lora_alpha")
    fan_in_fan_out = config.get("fan_in_fan_out")
    if rank != 16 or alpha != 32:
        raise AssertionError(f"Unexpected LoRA scaling configuration: r={rank}, alpha={alpha}")
    if fan_in_fan_out is not False:
        raise AssertionError("This script expects fan_in_fan_out=false")
    scale = float(alpha) / float(rank)

    with safe_open(str(BASE_PATH), framework="pt", device="cpu") as base_file:
        base_names = list(base_file.keys())
        base_name_set = set(base_names)
        sizes = {name: tensor_nbytes(base_file.get_slice(name)) for name in base_names}
        probe_shape = list(
            base_file.get_slice(
                "gpt_neox.layers.0.attention.query_key_value.weight"
            ).get_shape()
        )

    with safe_open(str(ADAPTER_PATH), framework="pt", device="cpu") as adapter_file:
        adapter_names = set(adapter_file.keys())
        a_names = sorted(name for name in adapter_names if name.endswith(".lora_A.weight"))
        b_names = sorted(name for name in adapter_names if name.endswith(".lora_B.weight"))

        pairs = {}
        for a_name in a_names:
            b_name = a_name[: -len(".lora_A.weight")] + ".lora_B.weight"
            if b_name not in adapter_names:
                raise AssertionError(f"Missing LoRA B tensor for {a_name}")
            base_name = base_name_from_adapter_a(a_name)
            if base_name not in base_name_set:
                raise AssertionError(f"LoRA target is absent from base checkpoint: {base_name}")

            a_shape = list(adapter_file.get_slice(a_name).get_shape())
            b_shape = list(adapter_file.get_slice(b_name).get_shape())
            with safe_open(str(BASE_PATH), framework="pt", device="cpu") as base_file:
                base_shape = list(base_file.get_slice(base_name).get_shape())
            if a_shape != [rank, base_shape[1]] or b_shape != [base_shape[0], rank]:
                raise AssertionError(
                    f"Incompatible shapes for {base_name}: base={base_shape}, "
                    f"A={a_shape}, B={b_shape}"
                )
            pairs[base_name] = (a_name, b_name)

    if len(a_names) != EXPECTED_PAIRS or len(b_names) != EXPECTED_PAIRS:
        raise AssertionError(
            f"Expected exactly {EXPECTED_PAIRS} adapter pairs, found "
            f"{len(a_names)} A tensors and {len(b_names)} B tensors"
        )
    if len(pairs) != EXPECTED_PAIRS:
        raise AssertionError(f"Expected exactly {EXPECTED_PAIRS} unique merge targets, found {len(pairs)}")
    if adapter_names != set(a_names) | set(b_names):
        extras = sorted(adapter_names - set(a_names) - set(b_names))
        raise AssertionError(f"Unexpected non-pair adapter tensors: {extras}")

    # Required output checks, evaluated before any checkpoint files are written.
    if len(base_names) != EXPECTED_TENSORS:
        raise AssertionError(f"Expected {EXPECTED_TENSORS} output tensors, found {len(base_names)}")
    if any("lora_" in name for name in base_names):
        raise AssertionError("A base/output tensor name contains 'lora_'")
    if probe_shape != [6144, 2048]:
        raise AssertionError(f"Layer 0 query_key_value shape changed/unexpected: {probe_shape}")
    missing_singletons = SPECIAL_SINGLETONS - base_name_set
    if missing_singletons:
        raise AssertionError(f"Required singleton tensors are missing: {sorted(missing_singletons)}")

    return base_names, sizes, pairs, scale


def plan_shards(base_names, sizes):
    shards = []
    current = []
    current_size = 0

    def flush():
        nonlocal current, current_size
        if current:
            shards.append(current)
            current = []
            current_size = 0

    for name in base_names:
        size = sizes[name]
        if size > MAX_SHARD_SIZE:
            flush()
            shards.append([name])
        elif name in SPECIAL_SINGLETONS:
            flush()
            shards.append([name])
        else:
            if current and current_size + size > MAX_SHARD_SIZE:
                flush()
            current.append(name)
            current_size += size
    flush()

    flattened = [name for shard in shards for name in shard]
    if flattened != base_names:
        raise AssertionError("Shard plan did not preserve every output tensor exactly once")
    for shard in shards:
        shard_size = sum(sizes[name] for name in shard)
        if shard_size > MAX_SHARD_SIZE and len(shard) != 1:
            raise AssertionError(f"Planned shard exceeds limit: {shard_size} bytes")
        if SPECIAL_SINGLETONS.intersection(shard) and len(shard) != 1:
            raise AssertionError(f"Embedding tensor was not planned as a singleton: {shard}")
    return shards


def main() -> None:
    base_names, sizes, pairs, scale = preflight()
    shards = plan_shards(base_names, sizes)

    # Only stale generated checkpoint artifacts are replaced; solution.py/REPORT.md remain.
    for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    shard_count = len(shards)
    weight_map = {}
    with safe_open(str(BASE_PATH), framework="pt", device="cpu") as base_file, safe_open(
        str(ADAPTER_PATH), framework="pt", device="cpu"
    ) as adapter_file:
        for shard_number, names in enumerate(shards, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {}
            for name in names:
                base_tensor = base_file.get_tensor(name)
                if name in pairs:
                    a_name, b_name = pairs[name]
                    a = adapter_file.get_tensor(a_name).float()
                    b = adapter_file.get_tensor(b_name).float()
                    merged = base_tensor.float().addmm(b, a, beta=1.0, alpha=scale)
                    base_tensor = merged.to(dtype=base_tensor.dtype)
                tensors[name] = base_tensor.contiguous()
                weight_map[name] = filename
            save_file(tensors, str(OUTPUT_DIR / filename))

    if len(weight_map) != EXPECTED_TENSORS or set(weight_map) != set(base_names):
        raise AssertionError("Written weight map does not contain the exact base key set")
    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Merged {len(pairs)} LoRA pairs into {len(base_names)} tensors; "
        f"wrote {shard_count} shards and {INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
