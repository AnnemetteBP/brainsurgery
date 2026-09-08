import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJECTION_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)


def is_layer_item(name: str, suffixes: tuple[str, ...]) -> bool:
    prefix = "gpt_neox.layers."
    if not name.startswith(prefix):
        return False
    remainder = name[len(prefix):]
    layer, separator, tail = remainder.partition(".")
    return separator == "." and layer.isdigit() and 0 <= int(layer) < 16 and tail in suffixes


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    source = load_file(str(INPUT), device="cpu")
    output = {}
    removed = []

    for name, tensor in source.items():
        if is_layer_item(name, BUFFER_SUFFIXES):
            removed.append(name)
        elif is_layer_item(name, PROJECTION_SUFFIXES):
            output[name] = tensor.to(torch.bfloat16)
        else:
            output[name] = tensor.to(torch.float32)

    # Required pre-write checks, plus exact buffer-count validation.
    bf16_names = [name for name, tensor in output.items() if tensor.dtype == torch.bfloat16]
    assert len(bf16_names) == 64, f"expected 64 bfloat16 tensors, got {len(bf16_names)}"
    assert output["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
    assert output["gpt_neox.embed_in.weight"].dtype == torch.float32
    assert len(output) == 196, f"expected 196 output tensors, got {len(output)}"
    assert len(removed) == 48, f"expected to remove 48 buffers, removed {len(removed)}"
    assert all(t.dtype in (torch.bfloat16, torch.float32) for t in output.values())

    # Greedily pack tensors in deterministic name order. Oversized tensors form
    # singleton shards; all ordinary shards stay below the tensor-data limit.
    shards = []
    current = {}
    current_bytes = 0
    for name in sorted(output):
        tensor = output[name]
        size = tensor_bytes(tensor)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        if size > MAX_SHARD_BYTES:
            assert not current
            shards.append({name: tensor})
        else:
            current[name] = tensor
            current_bytes += size
    if current:
        shards.append(current)

    total = len(shards)
    weight_map = {}
    for number, shard in enumerate(shards, 1):
        filename = f"model-{number:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(OUTPUT / filename), metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in output.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
