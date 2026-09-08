import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 64 * 1024 * 1024

PROJECTION_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
PROJECTION_NAMES = {
    f"h.{layer}.{suffix}"
    for layer in range(12)
    for suffix in PROJECTION_SUFFIXES
}
BUFFER_NAMES = {f"h.{layer}.attn.bias" for layer in range(12)}


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_nbytes(tensor)

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
    return shards


def main() -> None:
    source = load_file(INPUT_PATH, device="cpu")

    missing_projections = PROJECTION_NAMES - source.keys()
    missing_buffers = BUFFER_NAMES - source.keys()
    if missing_projections:
        raise RuntimeError(f"missing projection tensors: {sorted(missing_projections)}")
    if missing_buffers:
        raise RuntimeError(f"missing causal-mask buffers: {sorted(missing_buffers)}")

    output: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if name in BUFFER_NAMES:
            continue
        target_dtype = torch.bfloat16 if name in PROJECTION_NAMES else torch.float32
        output[name] = tensor.to(dtype=target_dtype).contiguous()

    # Required pre-write checks.
    bf16_names = {name for name, tensor in output.items() if tensor.dtype == torch.bfloat16}
    if bf16_names != PROJECTION_NAMES:
        raise RuntimeError(
            f"expected exactly the 48 projection tensors in bfloat16; got {len(bf16_names)}"
        )
    if output["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        raise RuntimeError("h.0.attn.c_attn.weight is not bfloat16")
    if output["wte.weight"].dtype != torch.float32:
        raise RuntimeError("wte.weight is not float32")
    if len(output) != 148:
        raise RuntimeError(f"expected 148 output tensors, got {len(output)}")
    non_float32 = {
        name
        for name, tensor in output.items()
        if name not in PROJECTION_NAMES and tensor.dtype != torch.float32
    }
    if non_float32:
        raise RuntimeError(f"non-projection tensors are not float32: {sorted(non_float32)}")

    shards = make_shards(output)
    shard_count = len(shards)
    weight_map: dict[str, str] = {}

    for shard_number, shard in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_nbytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(output)} tensors across {shard_count} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
