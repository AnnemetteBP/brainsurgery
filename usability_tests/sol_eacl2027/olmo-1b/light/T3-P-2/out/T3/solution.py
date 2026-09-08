import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path("inputs/base")
OUTPUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def main():
    with (INPUT_DIR / INDEX_NAME).open() as handle:
        input_index = json.load(handle)
    input_map = input_index["weight_map"]

    projection_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    projection_names = {
        f"model.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in projection_suffixes
    }
    assert len(projection_names) == 112
    assert projection_names <= set(input_map), "missing expected projection tensors"
    assert len(input_map) == 114, f"expected 114 input tensors, got {len(input_map)}"

    tensors = {}
    for source_name in sorted(set(input_map.values())):
        names = sorted(name for name, shard in input_map.items() if shard == source_name)
        with safe_open(INPUT_DIR / source_name, framework="pt", device="cpu") as source:
            for name in names:
                tensor = source.get_tensor(name)
                tensors[name] = tensor.to(torch.bfloat16) if name in projection_names else tensor.to(torch.float32)

    # Required pre-write checks.
    assert len(tensors) == 114, f"expected 114 output tensors, got {len(tensors)}"
    bf16_count = sum(t.dtype == torch.bfloat16 for t in tensors.values())
    assert bf16_count == 112, f"expected 112 bfloat16 tensors, got {bf16_count}"
    assert tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert tensors["model.embed_tokens.weight"].dtype == torch.float32
    assert all(
        tensor.dtype == (torch.bfloat16 if name in projection_names else torch.float32)
        for name, tensor in tensors.items()
    ), "unexpected output dtype"

    shards = []
    current = {}
    current_bytes = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_bytes(tensor)
        if current and (size > MAX_SHARD_BYTES or current_bytes + size > MAX_SHARD_BYTES):
            shards.append(current)
            current = {}
            current_bytes = 0
        if size > MAX_SHARD_BYTES:
            shards.append({name: tensor})
        else:
            current[name] = tensor
            current_bytes += size
    if current:
        shards.append(current)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, 1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    output_index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in tensors.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT_DIR / INDEX_NAME).open("w") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(tensors)} tensors in {shard_count} shards ({bf16_count} bfloat16).")


if __name__ == "__main__":
    main()
