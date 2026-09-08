#!/usr/bin/env python3
"""Export the supplied Pythia checkpoint with exact mixed dtypes and sharding."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("inputs/base/model.safetensors")
DESTINATION = Path("out/T3")
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJECTION = re.compile(
    r"^gpt_neox\.layers\.(?:[0-9]|1[0-5])\."
    r"(?:attention\.(?:query_key_value|dense)|"
    r"mlp\.(?:dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER = re.compile(
    r"^gpt_neox\.layers\.(?:[0-9]|1[0-5])\.attention\."
    r"(?:bias|masked_bias|rotary_emb\.inv_freq)$"
)


def transformed_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if PROJECTION.fullmatch(name) else torch.float32


def tensor_bytes(shape: list[int], dtype: torch.dtype) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * torch.empty((), dtype=dtype).element_size()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    old_outputs = list(DESTINATION.glob("model-*-of-*.safetensors"))
    old_outputs += list(DESTINATION.glob("model.safetensors.index.json"))
    if old_outputs:
        raise FileExistsError(
            "checkpoint output already exists; refusing to mix or overwrite files: "
            + ", ".join(map(str, old_outputs))
        )

    # Construct and validate the entire intended output manifest before writing.
    with safe_open(SOURCE, framework="pt", device="cpu") as source:
        source_names = sorted(source.keys())
        buffer_names = [name for name in source_names if BUFFER.fullmatch(name)]
        output_names = [name for name in source_names if name not in buffer_names]
        projection_names = [name for name in output_names if PROJECTION.fullmatch(name)]

        assert len(buffer_names) == 48, f"expected 48 buffers, found {len(buffer_names)}"
        assert len(projection_names) == 64, (
            f"expected exactly 64 bfloat16 tensors, found {len(projection_names)}"
        )
        assert (
            transformed_dtype("gpt_neox.layers.0.attention.query_key_value.weight")
            == torch.bfloat16
        ), "layer 0 query/key/value weight is not planned as bfloat16"
        assert transformed_dtype("gpt_neox.embed_in.weight") == torch.float32, (
            "input embedding is not planned as float32"
        )
        assert len(output_names) == 196, (
            f"expected exactly 196 output tensors, found {len(output_names)}"
        )

        sizes = {
            name: tensor_bytes(
                list(source.get_slice(name).get_shape()), transformed_dtype(name)
            )
            for name in output_names
        }

    # Greedy packing; an over-limit tensor is necessarily alone in its shard.
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in output_names:
        size = sizes[name]
        if current and (size > MAX_SHARD_BYTES or current_bytes + size > MAX_SHARD_BYTES):
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
    if current:
        shards.append(current)

    weight_map: dict[str, str] = {}
    total_size = sum(sizes.values())
    shard_count = len(shards)
    with safe_open(SOURCE, framework="pt", device="cpu") as source:
        for shard_number, names in enumerate(shards, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {
                name: source.get_tensor(name).to(transformed_dtype(name)) for name in names
            }
            shard_size = sum(sizes[name] for name in names)
            assert shard_size <= MAX_SHARD_BYTES or len(names) == 1, (
                f"shard {filename} exceeds limit and contains multiple tensors"
            )
            save_file(tensors, DESTINATION / filename, metadata={"format": "pt"})
            weight_map.update({name: filename for name in names})
            del tensors

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = DESTINATION / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(output_names)} tensors in {shard_count} shards "
        f"({len(projection_names)} bfloat16, {len(output_names)-len(projection_names)} float32)"
    )


if __name__ == "__main__":
    main()
