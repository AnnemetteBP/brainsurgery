#!/usr/bin/env python3
"""Independent full-value verification of the generated T5 checkpoint."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base"
ADAPTER_PATH = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
OUT = ROOT / "out" / "T5"
LIMIT = 536_870_912
SINGLETONS = {"model.embed_tokens.weight", "lm_head.weight"}

base_map = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
out_index = json.loads((OUT / "model.safetensors.index.json").read_text())
out_map = out_index["weight_map"]
assert set(base_map) == set(out_map) and len(out_map) == 114
assert not any("lora_" in name for name in out_map)

pairs = {}
with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter:
    for key in adapter.keys():
        short = key.removeprefix("base_model.model.")
        if short.endswith(".lora_A.weight"):
            pairs.setdefault(short.removesuffix(".lora_A.weight") + ".weight", {})["A"] = key
        elif short.endswith(".lora_B.weight"):
            pairs.setdefault(short.removesuffix(".lora_B.weight") + ".weight", {})["B"] = key
assert len(pairs) == 32 and all(set(pair) == {"A", "B"} for pair in pairs.values())

with ExitStack() as stack:
    base_readers = {
        filename: stack.enter_context(safe_open(BASE / filename, framework="pt", device="cpu"))
        for filename in set(base_map.values())
    }
    out_readers = {
        filename: stack.enter_context(safe_open(OUT / filename, framework="pt", device="cpu"))
        for filename in set(out_map.values())
    }
    adapter = stack.enter_context(safe_open(ADAPTER_PATH, framework="pt", device="cpu"))
    max_relative_error = 0.0
    unchanged = merged = 0
    for name in sorted(base_map):
        base = base_readers[base_map[name]].get_tensor(name)
        actual = out_readers[out_map[name]].get_tensor(name)
        assert actual.shape == base.shape and actual.dtype == base.dtype == torch.float32
        if name in pairs:
            a = adapter.get_tensor(pairs[name]["A"])
            b = adapter.get_tensor(pairs[name]["B"])
            expected = base + 2.0 * (b @ a)
            relative_error = ((actual - expected).norm() / expected.norm()).item()
            max_relative_error = max(max_relative_error, relative_error)
            assert relative_error <= 1e-5, f"merge error {relative_error} for {name}"
            merged += 1
        else:
            assert torch.equal(actual, base), f"unchanged tensor differs: {name}"
            unchanged += 1

for filename in set(out_map.values()):
    assigned = {name for name, shard in out_map.items() if shard == filename}
    with safe_open(OUT / filename, framework="pt", device="cpu") as reader:
        actual_names = set(reader.keys())
        shard_bytes = sum(
            reader.get_tensor(name).numel() * reader.get_tensor(name).element_size()
            for name in actual_names
        )
    assert actual_names == assigned
    assert shard_bytes <= LIMIT or len(actual_names) == 1
    if actual_names & SINGLETONS:
        assert len(actual_names) == 1

assert unchanged == 82 and merged == 32
print(
    f"PASS: {unchanged} unchanged tensors exact, {merged} merged tensors verified; "
    f"maximum relative error {max_relative_error:.3g}"
)
