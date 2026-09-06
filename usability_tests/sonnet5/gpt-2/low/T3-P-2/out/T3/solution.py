"""
T3: Mixed-precision export with sharding (GPT-2 124M).

Cast the 48 large projection matrices to bfloat16, keep everything else
(embeddings, layer norms, biases) in float32, drop the non-parameter causal
mask buffers (h.<i>.attn.bias), and write a sharded safetensors checkpoint
with an index file, each shard <= 64 MiB of tensor data (a single oversized
tensor gets its own shard).
"""

import json
import os
import re

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
OUTPUT_DIR = os.path.join(HERE, "..", "..", "out", "T3")

SHARD_LIMIT_BYTES = 64 * 1024 * 1024

# Exactly the 4 projection matrices per layer, per the task spec.
PROJ_RE = re.compile(
    r"^h\.\d+\.(attn\.c_attn\.weight|attn\.c_proj\.weight|mlp\.c_fc\.weight|mlp\.c_proj\.weight)$"
)
# Non-parameter causal-mask buffer to drop.
BUFFER_RE = re.compile(r"^h\.\d+\.attn\.bias$")


def load_state_dict(path):
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    state_dict = load_state_dict(INPUT_PATH)
    assert len(state_dict) == 160, f"expected 160 input tensors, got {len(state_dict)}"

    output = {}
    proj_matches = 0
    buffer_matches = 0

    for name, tensor in state_dict.items():
        if BUFFER_RE.match(name):
            buffer_matches += 1
            continue  # drop non-parameter buffer
        if PROJ_RE.match(name):
            proj_matches += 1
            output[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            output[name] = tensor.to(torch.float32).contiguous()

    assert proj_matches == 48, f"expected 48 projection matrices, matched {proj_matches}"
    assert buffer_matches == 12, f"expected 12 dropped buffers, matched {buffer_matches}"

    # Required checks (fail loudly before writing).
    bf16_count = sum(1 for t in output.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 48, f"expected exactly 48 bfloat16 tensors, got {bf16_count}"
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert output["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(output) == 148, f"expected exactly 148 output tensors, got {len(output)}"

    # Build shards: greedy bin-packing in a stable (sorted) order, each shard
    # capped at SHARD_LIMIT_BYTES of tensor data; an oversized single tensor
    # gets its own shard.
    def tensor_nbytes(t):
        return t.numel() * t.element_size()

    names_sorted = sorted(output.keys())

    shards = []  # list of list of names
    current_shard = []
    current_size = 0

    for name in names_sorted:
        size = tensor_nbytes(output[name])
        if size > SHARD_LIMIT_BYTES:
            # Oversized tensor: goes alone in its own shard.
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_size = 0
            shards.append([name])
            continue
        if current_size + size > SHARD_LIMIT_BYTES and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(name)
        current_size += size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0

    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: output[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUTPUT_DIR, shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(output[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(output)} tensors across {num_shards} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
