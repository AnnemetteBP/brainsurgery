#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into Pythia-1B and export safetensors shards."""

import json
import re
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
EXPECTED_TENSORS = 244
EXPECTED_PAIRS = 16
SHAPE_CHECK_KEY = "gpt_neox.layers.0.attention.query_key_value.weight"
# The task explicitly requests these large embedding tensors in solo shards.
SOLO_SHARD_KEYS = {"gpt_neox.embed_in.weight", "embed_out.weight"}
LORA_RE = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")


def tensor_nbytes(shape: list[int], dtype: str) -> int:
    sizes = {
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
    assert dtype in sizes, f"Unsupported safetensors dtype: {dtype}"
    return sizes[dtype] * int(torch.tensor(shape).prod().item())


def make_shard_plan(keys: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """Greedily pack regular tensors while putting requested large tensors alone."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(current)
            current = []
            current_bytes = 0

    for key in keys:
        size = sizes[key]
        assert size <= MAX_SHARD_BYTES, (
            f"Tensor {key} is {size} bytes, exceeding the shard limit"
        )
        if key in SOLO_SHARD_KEYS:
            flush()
            shards.append([key])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                flush()
            current.append(key)
            current_bytes += size
    flush()
    assert all(sum(sizes[k] for k in shard) <= MAX_SHARD_BYTES for shard in shards)
    assert all([key] in shards for key in SOLO_SHARD_KEYS)
    return shards


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    assert rank > 0, "LoRA rank must be positive"
    assert config.get("fan_in_fan_out") is False, (
        "This merge expects nn.Linear layout (fan_in_fan_out=false)"
    )
    scale = alpha / rank

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_file:
        base_keys = list(base_file.keys())
        base_shapes = {key: list(base_file.get_slice(key).get_shape()) for key in base_keys}
        base_dtypes = {key: base_file.get_slice(key).get_dtype() for key in base_keys}

    pairs: dict[str, dict[str, str]] = {}
    with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter_file:
        adapter_keys = list(adapter_file.keys())
        for adapter_key in adapter_keys:
            match = LORA_RE.fullmatch(adapter_key)
            assert match, f"Unexpected non-LoRA adapter tensor: {adapter_key}"
            module, factor = match.groups()
            base_key = f"{module}.weight"
            assert base_key in base_shapes, f"Adapter target absent from base: {base_key}"
            assert factor not in pairs.setdefault(base_key, {}), (
                f"Duplicate LoRA factor {factor} for {base_key}"
            )
            pairs[base_key][factor] = adapter_key

    # All required checks are performed on the intended output before writing.
    assert len(pairs) == EXPECTED_PAIRS, (
        f"Expected exactly {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}"
    )
    assert len(adapter_keys) == 2 * EXPECTED_PAIRS, (
        f"Expected {2 * EXPECTED_PAIRS} adapter tensors, found {len(adapter_keys)}"
    )
    assert all(set(pair) == {"A", "B"} for pair in pairs.values()), (
        "Every adapter target must have exactly one A and one B factor"
    )
    assert len(base_keys) == EXPECTED_TENSORS, (
        f"Expected exactly {EXPECTED_TENSORS} output tensors, found {len(base_keys)}"
    )
    assert not any("lora_" in key for key in base_keys), "LoRA key leaked into output names"
    assert base_shapes.get(SHAPE_CHECK_KEY) == [6144, 2048], (
        f"Unexpected shape for {SHAPE_CHECK_KEY}: {base_shapes.get(SHAPE_CHECK_KEY)}"
    )
    assert all(base_dtypes[key] == "F16" for key in pairs), (
        "All adapted base weights must be float16"
    )

    # Compute every merge before writing, so shape/layout failures cannot leave a
    # superficially complete checkpoint on disk.
    merged: dict[str, torch.Tensor] = {}
    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter_file,
    ):
        for base_key, pair in sorted(pairs.items()):
            base = base_file.get_tensor(base_key)
            a = adapter_file.get_tensor(pair["A"])
            b = adapter_file.get_tensor(pair["B"])
            assert list(a.shape) == [rank, base.shape[1]], (
                f"Bad A shape for {base_key}: {list(a.shape)}"
            )
            assert list(b.shape) == [base.shape[0], rank], (
                f"Bad B shape for {base_key}: {list(b.shape)}"
            )
            update = torch.matmul(b.float(), a.float())
            assert update.shape == base.shape
            merged[base_key] = (base.float() + scale * update).to(base.dtype).contiguous()

    sizes = {
        key: tensor_nbytes(base_shapes[key], base_dtypes[key])
        for key in base_keys
    }
    shards = make_shard_plan(sorted(base_keys), sizes)
    shard_names = [
        f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        for number in range(1, len(shards) + 1)
    ]
    weight_map = {
        key: shard_name
        for shard_name, shard_keys in zip(shard_names, shards)
        for key in shard_keys
    }
    assert set(weight_map) == set(base_keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_path.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_file:
        for shard_name, shard_keys in zip(shard_names, shards):
            tensors = {
                key: merged[key] if key in merged else base_file.get_tensor(key)
                for key in shard_keys
            }
            save_file(tensors, OUTPUT_DIR / shard_name, metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    # Verify the persisted artifact, including the tensor-data size of each shard.
    persisted_keys: set[str] = set()
    for shard_name, shard_keys in zip(shard_names, shards):
        shard_bytes = sum(sizes[key] for key in shard_keys)
        assert shard_bytes <= MAX_SHARD_BYTES
        with safe_open(OUTPUT_DIR / shard_name, framework="pt", device="cpu") as shard_file:
            actual = set(shard_file.keys())
        assert actual == set(shard_keys), f"Wrong keys in {shard_name}"
        assert persisted_keys.isdisjoint(actual), f"Duplicate tensor in {shard_name}"
        persisted_keys.update(actual)
    assert persisted_keys == set(base_keys)
    print(
        f"Merged {len(pairs)} LoRA pairs into {len(base_keys)} tensors; "
        f"wrote {len(shards)} shards (scale={scale:g})."
    )


if __name__ == "__main__":
    main()
