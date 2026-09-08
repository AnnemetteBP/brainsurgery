#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into the base and export bounded shards."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs/base"
LORA = ROOT / "inputs/lora"
OUT = Path(__file__).resolve().parent
MAX_SHARD_BYTES = 512 * 1024 * 1024
SINGLETONS = {"model.embed_tokens.weight", "lm_head.weight"}


def tensor_info(path: Path, name: str) -> tuple[tuple[int, ...], torch.dtype]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
        return tuple(tensor.shape), tensor.dtype


def main() -> None:
    config = json.loads((LORA / "adapter_config.json").read_text())
    rank = config["r"]
    scale = config["lora_alpha"] / rank
    fan_in_fan_out = config.get("fan_in_fan_out", False)
    assert rank == 16 and scale == 2, (rank, scale)
    assert fan_in_fan_out is False, "This input requires nn.Linear (B @ A) layout"

    base_index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map = base_index["weight_map"]
    adapter_path = LORA / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        adapter_names = set(adapter.keys())

    suffix_a = ".lora_A.weight"
    prefix = "base_model.model."
    pairs: dict[str, tuple[str, str]] = {}
    for a_name in sorted(name for name in adapter_names if name.endswith(suffix_a)):
        b_name = a_name[: -len(suffix_a)] + ".lora_B.weight"
        assert b_name in adapter_names, f"Missing B factor for {a_name}"
        assert a_name.startswith(prefix), f"Unexpected adapter prefix: {a_name}"
        base_name = a_name[len(prefix) : -len(suffix_a)] + ".weight"
        assert base_name in weight_map, f"Missing base tensor: {base_name}"
        pairs[base_name] = (a_name, b_name)

    # All required checks happen before the first output shard is written.
    assert len(pairs) == 32, f"Expected 32 adapter pairs, found {len(pairs)}"
    assert len(adapter_names) == 2 * len(pairs), "Unexpected unpaired adapter tensors"
    assert len(weight_map) == 114, f"Expected 114 output tensors, found {len(weight_map)}"
    assert not any("lora_" in name for name in weight_map), "LoRA name in output keys"
    q0 = "model.layers.0.self_attn.q_proj.weight"
    q0_path = BASE / weight_map[q0]
    assert tensor_info(q0_path, q0)[0] == (2048, 2048), "q_proj shape changed"

    # Plan shards from tensor metadata before loading their contents.
    sizes: dict[str, int] = {}
    by_source: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_source.setdefault(filename, []).append(name)
    for source, names in by_source.items():
        with safe_open(BASE / source, framework="pt", device="cpu") as base:
            for name in names:
                view = base.get_slice(name)
                assert view.get_dtype() == "F32", f"Non-float32 base tensor: {name}"
                sizes[name] = 4 * int(torch.tensor(view.get_shape()).prod().item())

    pending: list[str] = []
    pending_bytes = 0
    groups: list[list[str]] = []

    def flush() -> None:
        nonlocal pending, pending_bytes
        if pending:
            groups.append(pending)
            pending = []
            pending_bytes = 0

    for name in weight_map:
        size = sizes[name]
        if name in SINGLETONS:
            flush()
            groups.append([name])
        else:
            if pending and pending_bytes + size > MAX_SHARD_BYTES:
                flush()
            assert size <= MAX_SHARD_BYTES, f"Oversize non-singleton tensor: {name}"
            pending.append(name)
            pending_bytes += size
    flush()

    output_map: dict[str, str] = {}
    shard_count = len(groups)
    with ExitStack() as stack:
        bases = {
            source: stack.enter_context(safe_open(BASE / source, framework="pt", device="cpu"))
            for source in by_source
        }
        adapter = stack.enter_context(safe_open(adapter_path, framework="pt", device="cpu"))
        for number, group in enumerate(groups, 1):
            tensors: dict[str, torch.Tensor] = {}
            for name in group:
                value = bases[weight_map[name]].get_tensor(name)
                if name in pairs:
                    a_name, b_name = pairs[name]
                    a, b = adapter.get_tensor(a_name), adapter.get_tensor(b_name)
                    assert a.dtype == b.dtype == torch.float32
                    delta = b @ a
                    value = value + scale * (delta.T if fan_in_fan_out else delta)
                tensors[name] = value
            assert sum(sizes[name] for name in group) <= MAX_SHARD_BYTES
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            save_file(tensors, OUT / filename, metadata={"format": "pt"})
            output_map.update({name: filename for name in group})

    assert set(output_map) == set(weight_map)
    total_size = sum(sizes.values())
    index = {"metadata": {"total_size": int(total_size)}, "weight_map": output_map}
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"Merged {len(pairs)} LoRA pairs into {len(output_map)} tensors across {shard_count} shards")


if __name__ == "__main__":
    main()
