#!/usr/bin/env python3
"""Export the T3 checkpoint with exact dtype selection and size-bounded shards."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
INDEX = OUTPUT / "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJECTION_RE = re.compile(
    r"^gpt_neox\.layers\.(?:[0-9]|1[0-5])\."
    r"(?:attention\.(?:query_key_value|dense)|mlp\.(?:dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.(?:[0-9]|1[0-5])\.attention\."
    r"(?:bias|masked_bias|rotary_emb\.inv_freq)$"
)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def preflight(keys: list[str]) -> tuple[list[str], set[str]]:
    """Validate the complete planned key set and dtypes before writing files."""
    projections = {key for key in keys if PROJECTION_RE.fullmatch(key)}
    buffers = {key for key in keys if BUFFER_RE.fullmatch(key)}
    output_keys = [key for key in keys if key not in buffers]

    expected_projections = {
        f"gpt_neox.layers.{layer}.{module}.weight"
        for layer in range(16)
        for module in (
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        )
    }
    expected_buffers = {
        f"gpt_neox.layers.{layer}.attention.{name}"
        for layer in range(16)
        for name in ("bias", "masked_bias", "rotary_emb.inv_freq")
    }

    assert projections == expected_projections, (
        f"projection key mismatch: missing={sorted(expected_projections - projections)}, "
        f"extra={sorted(projections - expected_projections)}"
    )
    assert buffers == expected_buffers, (
        f"buffer key mismatch: missing={sorted(expected_buffers - buffers)}, "
        f"extra={sorted(buffers - expected_buffers)}"
    )

    # Required pre-write checks. Target dtype is defined solely by exact membership.
    planned_bfloat16 = sum(key in projections for key in output_keys)
    assert planned_bfloat16 == 64, f"expected 64 bfloat16 tensors, got {planned_bfloat16}"
    assert "gpt_neox.layers.0.attention.query_key_value.weight" in projections
    assert "gpt_neox.embed_in.weight" in output_keys
    assert "gpt_neox.embed_in.weight" not in projections  # therefore planned float32
    assert len(output_keys) == 196, f"expected 196 output tensors, got {len(output_keys)}"
    return output_keys, projections


def flush_shard(
    tensors: dict[str, torch.Tensor],
    shard_number: int,
    weight_map: dict[str, str],
) -> None:
    name = f"model-{shard_number:05d}-of-PLACEHOLDER.safetensors"
    save_file(tensors, OUTPUT / name, metadata={"format": "pt"})
    weight_map.update({key: name for key in tensors})


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if INDEX.exists() or list(OUTPUT.glob("model-*-of-*.safetensors")):
        raise FileExistsError("checkpoint output already exists; refusing to mix or overwrite shards")

    with safe_open(INPUT, framework="pt", device="cpu") as source:
        output_keys, projections = preflight(list(source.keys()))

        weight_map: dict[str, str] = {}
        shard: dict[str, torch.Tensor] = {}
        shard_bytes = 0
        shard_number = 0
        total_size = 0

        for key in output_keys:
            tensor = source.get_tensor(key)
            tensor = tensor.to(torch.bfloat16 if key in projections else torch.float32)
            expected_dtype = torch.bfloat16 if key in projections else torch.float32
            assert tensor.dtype == expected_dtype, f"wrong converted dtype for {key}: {tensor.dtype}"
            size = tensor_bytes(tensor)

            if shard and (shard_bytes + size > MAX_SHARD_BYTES or size > MAX_SHARD_BYTES):
                shard_number += 1
                flush_shard(shard, shard_number, weight_map)
                shard = {}
                shard_bytes = 0

            # Oversize tensors are valid only as singleton shards.
            if size > MAX_SHARD_BYTES:
                shard_number += 1
                flush_shard({key: tensor}, shard_number, weight_map)
            else:
                shard[key] = tensor
                shard_bytes += size
            total_size += size

        if shard:
            shard_number += 1
            flush_shard(shard, shard_number, weight_map)

    # Give every shard its final HuggingFace-style name and update the map.
    final_map: dict[str, str] = {}
    for key, temporary_name in weight_map.items():
        number = int(temporary_name.split("-")[1])
        final_map[key] = f"model-{number:05d}-of-{shard_number:05d}.safetensors"
    for number in range(1, shard_number + 1):
        temporary = OUTPUT / f"model-{number:05d}-of-PLACEHOLDER.safetensors"
        final = OUTPUT / f"model-{number:05d}-of-{shard_number:05d}.safetensors"
        temporary.rename(final)

    INDEX.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": final_map},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(final_map)} tensors in {shard_number} shards ({total_size} tensor bytes)")


if __name__ == "__main__":
    main()
