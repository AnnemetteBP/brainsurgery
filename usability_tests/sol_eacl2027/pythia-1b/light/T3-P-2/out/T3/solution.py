import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJECTION_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)


def projection_names():
    return {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in PROJECTION_SUFFIXES
    }


def buffer_names():
    suffixes = (
        "attention.bias",
        "attention.masked_bias",
        "attention.rotary_emb.inv_freq",
    )
    return {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in suffixes
    }


def main():
    projections = projection_names()
    buffers = buffer_names()

    with safe_open(SOURCE, framework="pt", device="cpu") as source:
        source_names = set(source.keys())
        assert len(source_names) == 244, f"expected 244 inputs, got {len(source_names)}"
        assert projections <= source_names, "one or more projection matrices are missing"
        assert buffers <= source_names, "one or more expected buffers are missing"

        output_names = sorted(source_names - buffers)
        output_dtypes = {
            name: torch.bfloat16 if name in projections else torch.float32
            for name in output_names
        }

        # Required pre-write checks.
        assert sum(dtype == torch.bfloat16 for dtype in output_dtypes.values()) == 64
        assert output_dtypes["gpt_neox.layers.0.attention.query_key_value.weight"] == torch.bfloat16
        assert output_dtypes["gpt_neox.embed_in.weight"] == torch.float32
        assert len(output_names) == 196, f"expected 196 outputs, got {len(output_names)}"

        sizes = {}
        for name in output_names:
            shape = source.get_slice(name).get_shape()
            sizes[name] = torch.tensor([], dtype=output_dtypes[name]).element_size() * int(
                torch.tensor(shape).prod().item()
            )

        # Greedy, deterministic packing. Oversized tensors occupy a shard alone.
        shards = []
        current = []
        current_size = 0
        for name in output_names:
            size = sizes[name]
            if current and (size > MAX_SHARD_BYTES or current_size + size > MAX_SHARD_BYTES):
                shards.append(current)
                current = []
                current_size = 0
            current.append(name)
            current_size += size
            if size > MAX_SHARD_BYTES:
                shards.append(current)
                current = []
                current_size = 0
        if current:
            shards.append(current)

        shard_count = len(shards)
        weight_map = {}
        for number, names in enumerate(shards, 1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {
                name: source.get_tensor(name).to(output_dtypes[name]).contiguous()
                for name in names
            }
            save_file(tensors, OUTPUT / filename, metadata={"format": "pt"})
            weight_map.update({name: filename for name in names})
            del tensors

    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
