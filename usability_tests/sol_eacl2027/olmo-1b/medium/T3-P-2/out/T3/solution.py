#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection-only BF16 and 256 MiB shards."""

import json
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path("inputs/base")
OUTPUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024
PROJECTION_RE = re.compile(
    r"^model\.layers\.(?:[0-9]|1[0-5])\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dtype).element_size()) * int(
        torch.tensor(shape).prod().item()
    )


def main() -> None:
    with (INPUT_DIR / INDEX_NAME).open(encoding="utf-8") as handle:
        source_index = json.load(handle)
    source_map = source_index["weight_map"]

    # Read only metadata first. This creates the entire proposed output plan,
    # allowing all required checks to run before any checkpoint file is written.
    source_info: dict[str, tuple[str, tuple[int, ...], torch.dtype]] = {}
    ordered_names: list[str] = []
    for shard_name in sorted(set(source_map.values())):
        names = sorted(name for name, shard in source_map.items() if shard == shard_name)
        with safe_open(INPUT_DIR / shard_name, framework="pt", device="cpu") as source:
            assert set(names) == set(source.keys()), f"Index mismatch in {shard_name}"
            for name in names:
                view = source.get_slice(name)
                dtype = view.get_dtype()
                assert dtype == "F32", f"Unexpected input dtype for {name}: {dtype}"
                source_info[name] = (shard_name, tuple(view.get_shape()), torch.float32)
                ordered_names.append(name)

    assert set(source_info) == set(source_map), "Input index and shard keys differ"
    output_dtypes = {
        name: torch.bfloat16 if PROJECTION_RE.fullmatch(name) else torch.float32
        for name in ordered_names
    }

    # Required pre-write checks.
    assert len(output_dtypes) == 114, f"Expected 114 tensors, got {len(output_dtypes)}"
    bf16_count = sum(dtype == torch.bfloat16 for dtype in output_dtypes.values())
    assert bf16_count == 112, f"Expected 112 bfloat16 tensors, got {bf16_count}"
    assert output_dtypes["model.layers.0.self_attn.q_proj.weight"] == torch.bfloat16
    assert output_dtypes["model.embed_tokens.weight"] == torch.float32

    # First-fit sequential packing, except that an oversized tensor occupies a
    # shard by itself as required. Sizes count tensor payload only.
    shard_groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in ordered_names:
        size = tensor_bytes(source_info[name][1], output_dtypes[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shard_groups.append(current)
                current, current_bytes = [], 0
            shard_groups.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shard_groups.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        shard_groups.append(current)

    output_map: dict[str, str] = {}
    total_size = 0
    shard_count = len(shard_groups)
    with ExitStack() as stack:
        sources = {
            shard_name: stack.enter_context(
                safe_open(INPUT_DIR / shard_name, framework="pt", device="cpu")
            )
            for shard_name in sorted(set(source_map.values()))
        }
        for number, names in enumerate(shard_groups, start=1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            payload_size = 0
            for name in names:
                source_name = source_info[name][0]
                tensor = sources[source_name].get_tensor(name)
                if output_dtypes[name] == torch.bfloat16:
                    tensor = tensor.to(torch.bfloat16)
                else:
                    assert tensor.dtype == torch.float32
                tensors[name] = tensor.contiguous()
                payload_size += tensor.numel() * tensor.element_size()
                output_map[name] = filename
            assert payload_size <= MAX_SHARD_BYTES or len(names) == 1
            save_file(tensors, OUTPUT_DIR / filename, metadata={"format": "pt"})
            total_size += payload_size

    assert len(output_map) == 114 and set(output_map) == set(source_map)
    index = {"metadata": {"total_size": total_size}, "weight_map": output_map}
    with (OUTPUT_DIR / INDEX_NAME).open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(output_map)} tensors in {shard_count} shards ({total_size} bytes)")


if __name__ == "__main__":
    main()
