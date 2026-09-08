from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024
NUM_LAYERS = 16

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


def layer_names(suffixes: tuple[str, ...]) -> set[str]:
    return {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(NUM_LAYERS)
        for suffix in suffixes
    }


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    projection_names = layer_names(PROJECTION_SUFFIXES)
    buffer_names = layer_names(BUFFER_SUFFIXES)

    # Establish the complete output plan and perform all required checks before
    # creating any checkpoint files.
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        input_names = set(source.keys())

    missing_projections = projection_names - input_names
    missing_buffers = buffer_names - input_names
    if missing_projections:
        raise RuntimeError(f"missing projection tensors: {sorted(missing_projections)}")
    if missing_buffers:
        raise RuntimeError(f"missing buffers: {sorted(missing_buffers)}")

    output_names = input_names - buffer_names
    bf16_names = output_names & projection_names
    if len(bf16_names) != 64:
        raise RuntimeError(f"expected exactly 64 bfloat16 tensors, got {len(bf16_names)}")
    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if qkv0 not in bf16_names:
        raise RuntimeError(f"{qkv0} is not planned as bfloat16")
    if "gpt_neox.embed_in.weight" not in output_names:
        raise RuntimeError("gpt_neox.embed_in.weight is absent from the output")
    if len(output_names) != 196:
        raise RuntimeError(f"expected exactly 196 output tensors, got {len(output_names)}")

    # Refuse to mix a newly generated checkpoint with stale shards from a
    # prior invocation.
    existing = list(OUTPUT_DIR.glob("model-*-of-*.safetensors"))
    existing += list(OUTPUT_DIR.glob(".tmp-shard-*.safetensors"))
    if existing or INDEX_PATH.exists():
        raise RuntimeError("output checkpoint files already exist; remove them before rerunning")

    temporary_shards: list[Path] = []
    shard_keys: list[list[str]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    total_size = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_number = len(temporary_shards) + 1
        path = OUTPUT_DIR / f".tmp-shard-{shard_number:05d}.safetensors"
        save_file(current, path)
        temporary_shards.append(path)
        shard_keys.append(list(current))
        current = {}
        current_bytes = 0

    with safe_open(INPUT, framework="pt", device="cpu") as source:
        for name in sorted(output_names):
            tensor = source.get_tensor(name)
            target_dtype = torch.bfloat16 if name in projection_names else torch.float32
            tensor = tensor.to(dtype=target_dtype)
            size = tensor_nbytes(tensor)

            if current and current_bytes + size > MAX_SHARD_BYTES:
                flush()
            current[name] = tensor
            current_bytes += size
            total_size += size

            # Oversized tensors are valid only as single-tensor shards.
            if size > MAX_SHARD_BYTES:
                if len(current) != 1:
                    raise RuntimeError(f"oversized tensor {name} was not isolated")
                flush()
        flush()

    shard_count = len(temporary_shards)
    weight_map: dict[str, str] = {}
    for number, (temporary, names) in enumerate(zip(temporary_shards, shard_keys), start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        temporary.rename(OUTPUT_DIR / filename)
        for name in names:
            weight_map[name] = filename

    if set(weight_map) != output_names:
        raise RuntimeError("internal error: index key set does not match output key set")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_names)} tensors in {shard_count} shards ({total_size} bytes)")


if __name__ == "__main__":
    main()
