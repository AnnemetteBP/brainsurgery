#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into GPT-2 and export safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


MAX_SHARD_BYTES = 100 * 1024 * 1024
EXPECTED_TENSORS = 160
EXPECTED_PAIRS = 12
ADAPTER_PREFIX = "base_model.model."
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(
    state: dict[str, torch.Tensor], max_bytes: int
) -> list[dict[str, torch.Tensor]]:
    """Greedily pack tensors without splitting any individual tensor."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(state):
        tensor = state[name]
        size = tensor_nbytes(tensor)
        if current and current_bytes + size > max_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0

        current[name] = tensor
        current_bytes += size

        # A tensor larger than the limit is permitted only in a shard by itself.
        if size > max_bytes:
            assert len(current) == 1, f"oversized tensor {name} was not isolated"
            shards.append(current)
            current = {}
            current_bytes = 0

    if current:
        shards.append(current)
    return shards


def main() -> None:
    sandbox = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).resolve().parent
    base_path = sandbox / "inputs/base/model.safetensors"
    adapter_path = sandbox / "inputs/lora/adapter_model.safetensors"
    config_path = sandbox / "inputs/lora/adapter_config.json"

    config = json.loads(config_path.read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    assert rank > 0, f"invalid LoRA rank: {rank}"
    assert config.get("fan_in_fan_out") is True, "expected Conv1D fan_in_fan_out=true"
    scale = alpha / rank

    base = load_file(base_path, device="cpu")
    adapter = load_file(adapter_path, device="cpu")

    a_names = {name for name in adapter if name.endswith(A_SUFFIX)}
    b_names = {name for name in adapter if name.endswith(B_SUFFIX)}
    assert len(a_names) == EXPECTED_PAIRS, (
        f"expected {EXPECTED_PAIRS} lora_A tensors, found {len(a_names)}"
    )
    assert len(b_names) == EXPECTED_PAIRS, (
        f"expected {EXPECTED_PAIRS} lora_B tensors, found {len(b_names)}"
    )

    consumed: set[str] = set()
    merged_keys: set[str] = set()
    for a_name in sorted(a_names):
        stem = a_name[: -len(A_SUFFIX)]
        b_name = stem + B_SUFFIX
        assert b_name in b_names, f"missing adapter pair for {a_name}"
        assert stem.startswith(ADAPTER_PREFIX), f"unexpected adapter prefix: {stem}"
        base_key = stem.removeprefix(ADAPTER_PREFIX) + ".weight"
        assert base_key in base, f"adapter target not found in base: {base_key}"
        assert base_key not in merged_keys, f"duplicate adapter target: {base_key}"

        a = adapter[a_name]
        b = adapter[b_name]
        weight = base[base_key]
        assert a.dtype == b.dtype == weight.dtype == torch.float32, (
            f"{base_key}: merge operands must all be float32"
        )
        assert tuple(a.shape) == (rank, weight.shape[0]), (
            f"{a_name}: incompatible shape {tuple(a.shape)}"
        )
        assert tuple(b.shape) == (weight.shape[1], rank), (
            f"{b_name}: incompatible shape {tuple(b.shape)}"
        )

        # B @ A uses Linear [out, in] layout; GPT-2 Conv1D is [in, out].
        base[base_key] = weight + (b @ a).T * scale
        consumed.update((a_name, b_name))
        merged_keys.add(base_key)

    # Required pre-write checks, plus full adapter-consumption validation.
    assert len(merged_keys) == EXPECTED_PAIRS, (
        f"expected exactly {EXPECTED_PAIRS} merged pairs, got {len(merged_keys)}"
    )
    assert consumed == set(adapter), "unpaired or unexpected adapter tensors remain"
    assert not any("lora_" in name for name in base), "LoRA tensor leaked into output"
    assert tuple(base["h.0.attn.c_attn.weight"].shape) == (768, 2304), (
        "h.0.attn.c_attn.weight has the wrong shape"
    )
    assert len(base) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} output tensors, got {len(base)}"
    )

    shards = make_shards(base, MAX_SHARD_BYTES)
    for shard in shards:
        sizes = [tensor_nbytes(tensor) for tensor in shard.values()]
        assert sum(sizes) <= MAX_SHARD_BYTES or (
            len(shard) == 1 and sizes[0] > MAX_SHARD_BYTES
        ), "invalid shard exceeds 100 MiB without being a single oversized tensor"

    # Make reruns safe: remove only checkpoint files produced by this script.
    for old_shard in output_dir.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    for old_file in (
        output_dir / "model.safetensors",
        output_dir / "model.safetensors.index.json",
    ):
        if old_file.exists():
            old_file.unlink()

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_nbytes(t) for t in base.values())},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Merged {len(merged_keys)} LoRA pairs and wrote "
        f"{len(base)} tensors across {shard_count} shards to {output_dir}"
    )


if __name__ == "__main__":
    main()
