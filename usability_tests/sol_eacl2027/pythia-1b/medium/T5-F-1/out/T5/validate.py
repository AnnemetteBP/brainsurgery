#!/usr/bin/env python3
"""Independent content validation for the generated T5 checkpoint."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    index = json.loads((OUT / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((ROOT / "inputs/lora/adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / int(config["r"])

    with ExitStack() as stack:
        shards = {
            filename: stack.enter_context(safe_open(OUT / filename, framework="pt", device="cpu"))
            for filename in sorted(set(weight_map.values()))
        }
        adapter = stack.enter_context(
            safe_open(ROOT / "inputs/lora/adapter_model.safetensors", framework="pt", device="cpu")
        )
        a_names = sorted(key for key in adapter.keys() if key.endswith(".lora_A.weight"))
        adapted = {
            key.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight": key
            for key in a_names
        }
        base = stack.enter_context(safe_open(ROOT / "inputs/base/model.safetensors", framework="pt", device="cpu"))
        assert set(weight_map) == set(base.keys())
        unchanged = 0
        checked_merged = 0
        max_relative_error = 0.0
        for key in base.keys():
            original = base.get_tensor(key)
            actual = shards[weight_map[key]].get_tensor(key)
            if key not in adapted:
                assert torch.equal(actual, original), f"unchanged tensor differs: {key}"
                unchanged += 1
                continue
            a_name = adapted[key]
            b_name = a_name.removesuffix(".lora_A.weight") + ".lora_B.weight"
            expected = (original.float() + scale * (adapter.get_tensor(b_name).float() @ adapter.get_tensor(a_name).float())).to(original.dtype)
            relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
            max_relative_error = max(max_relative_error, relative_error.item())
            assert relative_error <= 1e-3, f"merged tensor exceeds tolerance: {key}: {relative_error}"
            checked_merged += 1

    assert unchanged == 228, f"checked {unchanged} unchanged tensors, expected 228"
    assert checked_merged == 16, f"checked {checked_merged} merged tensors, expected 16"
    print(
        f"Validated {unchanged} bit-exact unchanged tensors and {checked_merged} merged tensors; "
        f"maximum relative error={max_relative_error:.3g}."
    )


if __name__ == "__main__":
    main()
