#!/usr/bin/env python3
"""Merge the supplied GPT-2 LoRA adapter and export bounded safetensor shards."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = Path(__file__).resolve().parent
MAX_SHARD_BYTES = 100 * 1024 * 1024
ADAPTER_RE = re.compile(
    r"^base_model\.model\.(.+)\.lora_([AB])\.weight$"
)
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def tensor_data_bytes(slice_) -> int:
    shape = slice_.get_shape()
    elements = 1
    for dimension in shape:
        elements *= dimension
    dtype = slice_.get_dtype()
    if dtype not in DTYPE_BYTES:
        raise AssertionError(f"unsupported safetensors dtype: {dtype}")
    return elements * DTYPE_BYTES[dtype]


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["r"] > 0, "LoRA rank must be positive"
    assert config.get("fan_in_fan_out") is True, "expected fan_in_fan_out=true"
    scale = float(config["lora_alpha"]) / int(config["r"])

    with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter:
        adapter_keys = list(adapter.keys())
        pairs: dict[str, dict[str, str]] = {}
        for key in adapter_keys:
            match = ADAPTER_RE.fullmatch(key)
            assert match, f"unexpected adapter tensor: {key}"
            base_stem, factor = match.groups()
            base_key = base_stem + ".weight"
            pairs.setdefault(base_key, {})[factor] = key

        assert len(pairs) == 12, f"expected 12 adapter pairs, found {len(pairs)}"
        assert all(set(pair) == {"A", "B"} for pair in pairs.values()), \
            "every adapted module must have exactly one A and one B factor"

        with safe_open(BASE_PATH, framework="pt", device="cpu") as base:
            base_keys = list(base.keys())
            base_key_set = set(base_keys)

            # All required checks happen before any shard is written.
            assert len(base_keys) == 160, f"expected 160 output tensors, found {len(base_keys)}"
            assert not any("lora_" in key for key in base_keys), "LoRA tensor leaked into output keys"
            probe = base.get_slice("h.0.attn.c_attn.weight")
            assert probe.get_shape() == [768, 2304], \
                f"unexpected h.0.attn.c_attn.weight shape: {probe.get_shape()}"
            assert set(pairs).issubset(base_key_set), "an adapted base tensor is missing"

            sizes = {key: tensor_data_bytes(base.get_slice(key)) for key in base_keys}
            groups: list[list[str]] = []
            current: list[str] = []
            current_size = 0
            for key in base_keys:
                size = sizes[key]
                if current and current_size + size > MAX_SHARD_BYTES:
                    groups.append(current)
                    current, current_size = [], 0
                current.append(key)
                current_size += size
                if size > MAX_SHARD_BYTES:
                    assert len(current) == 1, "oversized tensor must be alone"
                    groups.append(current)
                    current, current_size = [], 0
            if current:
                groups.append(current)

            # Remove only stale checkpoint products from a prior invocation.
            for old in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
                old.unlink()
            index_path = OUTPUT_DIR / "model.safetensors.index.json"
            if index_path.exists():
                index_path.unlink()

            weight_map: dict[str, str] = {}
            shard_count = len(groups)
            for number, keys in enumerate(groups, 1):
                filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
                tensors = {}
                for key in keys:
                    value = base.get_tensor(key)
                    if key in pairs:
                        pair = pairs[key]
                        a = adapter.get_tensor(pair["A"])
                        b = adapter.get_tensor(pair["B"])
                        assert value.dtype == a.dtype == b.dtype == torch.float32
                        delta = torch.matmul(b, a).transpose(0, 1)
                        assert delta.shape == value.shape, f"shape mismatch while merging {key}"
                        value = value + delta.mul(scale)
                    tensors[key] = value.contiguous()
                    weight_map[key] = filename
                save_file(tensors, OUTPUT_DIR / filename, metadata={"format": "pt"})

    assert len(weight_map) == 160
    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Merged {len(pairs)} LoRA pairs into {len(weight_map)} tensors across {shard_count} shards")


if __name__ == "__main__":
    main()
