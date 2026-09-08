#!/usr/bin/env python3
"""Create the mixed-precision, buffer-free, sharded T3 checkpoint."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
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


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def preflight() -> tuple[list[str], dict[str, int]]:
    """Validate the intended output completely before any output is written."""
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        output_keys = [key for key in source_keys if not BUFFER.fullmatch(key)]
        projection_keys = [key for key in output_keys if PROJECTION.fullmatch(key)]

        assert len(source_keys) == 244, f"expected 244 input tensors, got {len(source_keys)}"
        assert len(output_keys) == 196, f"expected 196 output tensors, got {len(output_keys)}"
        assert len(projection_keys) == 64, (
            f"expected exactly 64 bfloat16 tensors, got {len(projection_keys)}"
        )
        assert "gpt_neox.layers.0.attention.query_key_value.weight" in projection_keys
        assert "gpt_neox.embed_in.weight" in output_keys
        assert not PROJECTION.fullmatch("gpt_neox.embed_in.weight")
        assert sum(bool(BUFFER.fullmatch(key)) for key in source_keys) == 48
        output_sizes = {
            key: torch.Size(source.get_slice(key).get_shape()).numel() * 4
            if key not in projection_keys
            else torch.Size(source.get_slice(key).get_shape()).numel() * 2
            for key in output_keys
        }

    # The conversion below is exclusively determined by membership in these
    # validated sets: projections become bfloat16 and all other keys float32.
    intended_dtype = {
        key: torch.bfloat16 if key in set(projection_keys) else torch.float32
        for key in output_keys
    }
    assert sum(dtype == torch.bfloat16 for dtype in intended_dtype.values()) == 64
    assert intended_dtype["gpt_neox.layers.0.attention.query_key_value.weight"] == torch.bfloat16
    assert intended_dtype["gpt_neox.embed_in.weight"] == torch.float32
    return output_keys, output_sizes


def main() -> None:
    output_keys, output_sizes = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Plan greedily in safetensors key order. Oversize individual tensors are
    # permitted, but are always isolated in their own shard.
    shard_keys: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in output_keys:
        size = output_sizes[key]
        if current and (current_bytes + size > MAX_SHARD_BYTES or size > MAX_SHARD_BYTES):
            shard_keys.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            shard_keys.append(current)
            current = []
            current_bytes = 0
    if current:
        shard_keys.append(current)

    count = len(shard_keys)
    weight_map: dict[str, str] = {}
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        for number, keys in enumerate(shard_keys, start=1):
            shard = {
                key: source.get_tensor(key).to(
                    torch.bfloat16 if PROJECTION.fullmatch(key) else torch.float32
                )
                for key in keys
            }
            shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
            assert shard_size <= MAX_SHARD_BYTES or len(shard) == 1, (
                f"shard {number} exceeds limit and contains multiple tensors"
            )
            filename = f"model-{number:05d}-of-{count:05d}.safetensors"
            save_file(shard, OUTPUT / filename, metadata={"format": "pt"})
            weight_map.update({key: filename for key in shard})

    assert set(weight_map) == set(output_keys)
    total_bytes = sum(output_sizes.values())
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (OUTPUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(output_keys)} tensors in {count} shards ({total_bytes} data bytes)")


if __name__ == "__main__":
    main()
