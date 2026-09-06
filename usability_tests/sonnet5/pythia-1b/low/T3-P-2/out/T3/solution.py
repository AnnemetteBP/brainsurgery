"""
T3: Mixed-precision export with sharding (Pythia-1B).

Loads inputs/base/model.safetensors, casts the 64 large projection matrices
to bfloat16, upcasts everything else to float32, drops the 48 non-parameter
buffers, and writes a sharded safetensors checkpoint (<=256MiB per shard,
oversized tensors get their own shard) plus an index file.
"""

import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUT_PATH = os.path.join(REPO_ROOT, "inputs", "base", "model.safetensors")
OUT_DIR = HERE
SHARD_LIMIT = 256 * 1024 * 1024  # 268,435,456 bytes

BF16_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def main():
    tensors = load_file(INPUT_PATH)

    out = {}
    for name, tensor in tensors.items():
        if BUFFER_PATTERN.match(name):
            continue  # drop non-parameter buffers
        if BF16_PATTERN.match(name):
            out[name] = tensor.to(torch.bfloat16)
        else:
            out[name] = tensor.to(torch.float32)

    # --- Required checks ---
    bf16_names = [n for n, t in out.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 64, f"expected 64 bfloat16 tensors, got {len(bf16_names)}"
    assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
    assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
    assert len(out) == 196, f"expected 196 tensors, got {len(out)}"
    for n in bf16_names:
        assert BF16_PATTERN.match(n), f"unexpected bfloat16 tensor {n}"
    for n, t in out.items():
        if n not in bf16_names:
            assert t.dtype == torch.float32, f"{n} is not float32"
    for name in tensors:
        if BUFFER_PATTERN.match(name):
            assert name not in out, f"buffer {name} was not dropped"

    def nbytes(t):
        return t.numel() * t.element_size()

    # Bin-pack tensors into shards, oversized tensors alone in their own shard.
    names_sorted = sorted(out.keys())
    shards = []  # list of dict name->tensor
    current = {}
    current_size = 0
    for name in names_sorted:
        t = out[name]
        size = nbytes(t)
        if size > SHARD_LIMIT:
            if current:
                shards.append(current)
                current = {}
                current_size = 0
            shards.append({name: t})
            continue
        if current_size + size > SHARD_LIMIT:
            shards.append(current)
            current = {}
            current_size = 0
        current[name] = t
        current_size += size
    if current:
        shards.append(current)

    os.makedirs(OUT_DIR, exist_ok=True)
    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    shard_filenames = []
    for idx, shard in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_filenames.append(filename)
        for name, t in shard.items():
            weight_map[name] = filename
            total_size += nbytes(t)

    for filename, shard in zip(shard_filenames, shards):
        save_file(shard, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {num_shards} shards, {len(out)} tensors, total_size={total_size}")


if __name__ == "__main__":
    main()
