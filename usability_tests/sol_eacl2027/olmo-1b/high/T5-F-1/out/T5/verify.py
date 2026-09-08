#!/usr/bin/env python3
"""Independent read-back verification for the generated T5 checkpoint."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "inputs" / "base"
LORA = ROOT / "inputs" / "lora"
LIMIT = 512 * 1024 * 1024
DTYPE_BYTES = {"F32": 4}


def main() -> None:
    base_index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    out_index = json.loads((HERE / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(out_index) == set(base_index) and len(out_index) == 114
    assert not any("lora_" in name for name in out_index)

    adapter = load_file(LORA / "adapter_model.safetensors", device="cpu")
    config = json.loads((LORA / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    merged = set()

    with ExitStack() as stack:
        base_handles = {
            filename: stack.enter_context(safe_open(BASE / filename, framework="pt", device="cpu"))
            for filename in set(base_index.values())
        }
        out_handles = {
            filename: stack.enter_context(safe_open(HERE / filename, framework="pt", device="cpu"))
            for filename in set(out_index.values())
        }
        actual_names = set()
        for filename, handle in out_handles.items():
            names = set(handle.keys())
            assert names == {name for name, shard in out_index.items() if shard == filename}
            actual_names.update(names)
            shard_bytes = 0
            for name in names:
                view = handle.get_slice(name)
                numel = 1
                for dimension in view.get_shape():
                    numel *= dimension
                shard_bytes += numel * DTYPE_BYTES[view.get_dtype()]
            assert shard_bytes <= LIMIT, (filename, shard_bytes)
        assert actual_names == set(out_index)

        for large in ("model.embed_tokens.weight", "lm_head.weight"):
            assert list(out_index.values()).count(out_index[large]) == 1

        for name in sorted(base_index):
            base_view = base_handles[base_index[name]].get_slice(name)
            out_view = out_handles[out_index[name]].get_slice(name)
            assert base_view.get_shape() == out_view.get_shape()
            assert base_view.get_dtype() == out_view.get_dtype() == "F32"
            adapter_prefix = "base_model.model." + name.removesuffix(".weight")
            a_name = adapter_prefix + ".lora_A.weight"
            b_name = adapter_prefix + ".lora_B.weight"
            if a_name in adapter:
                expected = base_view[:] + scale * torch.matmul(adapter[b_name], adapter[a_name])
                actual = out_view[:]
                assert torch.equal(actual, expected), f"merged value mismatch: {name}"
                merged.add(name)
            else:
                rows = base_view.get_shape()[0] if base_view.get_shape() else 1
                for start in range(0, rows, 1024):
                    stop = min(start + 1024, rows)
                    assert torch.equal(base_view[start:stop], out_view[start:stop]), (
                        f"unchanged value mismatch: {name}"
                    )

    assert len(merged) == 32
    print("Verified 114 tensors: 32 merged exactly, 82 unchanged bit-for-bit; all shards valid.")


if __name__ == "__main__":
    main()
