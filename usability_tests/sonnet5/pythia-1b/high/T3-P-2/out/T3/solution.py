"""
T3: Mixed-precision export with sharding (Pythia-1B).

Reads inputs/base/model.safetensors, casts the 64 large projection matrices
to bfloat16, upcasts everything else to float32, drops the 48 non-parameter
buffers, and writes a sharded safetensors checkpoint (<=256MiB tensor data
per shard, oversized tensors alone in their own shard) plus an index file.
"""

import json
import os

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUT_PATH = os.path.join(REPO_ROOT, "inputs", "base", "model.safetensors")
OUT_DIR = os.path.join(REPO_ROOT, "out", "T3")

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes
NUM_LAYERS = 16

# The 4 projection-matrix name templates that must become bfloat16.
BF16_TEMPLATES = [
    "gpt_neox.layers.{i}.attention.query_key_value.weight",
    "gpt_neox.layers.{i}.attention.dense.weight",
    "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
    "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
]
BF16_NAMES = {t.format(i=i) for i in range(NUM_LAYERS) for t in BF16_TEMPLATES}

# The 3 non-parameter buffer templates per layer that must be dropped.
BUFFER_TEMPLATES = [
    "gpt_neox.layers.{i}.attention.bias",
    "gpt_neox.layers.{i}.attention.masked_bias",
    "gpt_neox.layers.{i}.attention.rotary_emb.inv_freq",
]
BUFFER_NAMES = {t.format(i=i) for i in range(NUM_LAYERS) for t in BUFFER_TEMPLATES}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with safe_open(INPUT_PATH, framework="pt") as f:
        all_keys = list(f.keys())
        assert len(all_keys) == 244, f"expected 244 input tensors, got {len(all_keys)}"

        tensors: dict[str, torch.Tensor] = {}
        for key in all_keys:
            if key in BUFFER_NAMES:
                continue  # drop non-parameter buffers
            t = f.get_tensor(key)
            if key in BF16_NAMES:
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            tensors[key] = t.contiguous()

    # --- Required checks (fail loudly before writing) ---
    bf16_count = sum(1 for t in tensors.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 64, f"expected exactly 64 bfloat16 tensors, got {bf16_count}"

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert tensors[qkv0].dtype == torch.bfloat16, f"{qkv0} must be bfloat16"

    assert tensors["gpt_neox.embed_in.weight"].dtype == torch.float32, (
        "gpt_neox.embed_in.weight must be float32"
    )

    assert len(tensors) == 196, f"expected exactly 196 output tensors, got {len(tensors)}"

    for name in BUFFER_NAMES:
        assert name not in tensors, f"buffer {name} should have been dropped"

    # --- Sharding: greedy bin-pack, oversized tensors get their own shard ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names = list(tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in names:
        nbytes = tensor_nbytes(tensors[name])
        if nbytes > MAX_SHARD_BYTES:
            # Oversized tensor: alone in its own shard.
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes

    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(tensors)} tensors across {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
