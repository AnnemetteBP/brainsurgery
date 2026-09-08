#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a sharded safetensors checkpoint."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import struct

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


BASE_DIR = Path("inputs/base")
ADAPTER_DIR = Path("inputs/lora")
OUTPUT_DIR = Path("out/T5")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
DEDICATED_TENSORS = {"model.embed_tokens.weight", "lm_head.weight"}


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read tensor metadata without materializing checkpoint tensors."""
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise RuntimeError(f"Invalid safetensors header in {path}")
        (header_length,) = struct.unpack("<Q", length_bytes)
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise RuntimeError(f"Truncated safetensors header in {path}")
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    return header


def load_base_metadata(weight_map: dict[str, str]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = BASE_DIR / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing base shard: {shard_path}")
        for tensor_name, info in read_safetensors_header(shard_path).items():
            if tensor_name in metadata:
                raise RuntimeError(f"Duplicate base tensor: {tensor_name}")
            metadata[tensor_name] = {
                "shape": list(info["shape"]),
                "dtype": info["dtype"],
                "nbytes": info["data_offsets"][1] - info["data_offsets"][0],
                "shard": shard_name,
            }

    if set(metadata) != set(weight_map):
        missing = sorted(set(weight_map) - set(metadata))
        extra = sorted(set(metadata) - set(weight_map))
        raise RuntimeError(
            f"Base index/header mismatch; missing={missing}, extra={extra}"
        )
    for tensor_name, shard_name in weight_map.items():
        if metadata[tensor_name]["shard"] != shard_name:
            raise RuntimeError(f"Wrong source shard for {tensor_name}")
    return metadata


def adapter_pairs(
    adapter: dict[str, torch.Tensor], base_metadata: dict[str, dict]
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefix = "base_model.model."
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    consumed: set[str] = set()

    for a_name in sorted(name for name in adapter if name.endswith(a_suffix)):
        stem = a_name[: -len(a_suffix)]
        b_name = stem + b_suffix
        if b_name not in adapter:
            raise RuntimeError(f"Missing LoRA B tensor for {a_name}")
        if not stem.startswith(prefix):
            raise RuntimeError(f"Unexpected PEFT tensor prefix: {a_name}")
        base_name = stem[len(prefix) :] + ".weight"
        if base_name not in base_metadata:
            raise RuntimeError(f"Adapter target absent from base: {base_name}")

        a = adapter[a_name]
        b = adapter[b_name]
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise RuntimeError(f"LoRA factors for {base_name} must be float32")
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise RuntimeError(f"Incompatible LoRA factor shapes for {base_name}")

        pairs.append((base_name, a, b))
        consumed.update((a_name, b_name))

    if consumed != set(adapter):
        unexpected = sorted(set(adapter) - consumed)
        raise RuntimeError(f"Unpaired or unexpected adapter tensors: {unexpected}")
    return pairs


def plan_shards(base_metadata: dict[str, dict]) -> list[list[str]]:
    """Greedily pack tensors, isolating the task's two large matrices."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(current)
            current = []
            current_bytes = 0

    for name in sorted(base_metadata):
        nbytes = base_metadata[name]["nbytes"]
        if name in DEDICATED_TENSORS or nbytes > MAX_SHARD_BYTES:
            flush()
            shards.append([name])
        else:
            if current and current_bytes + nbytes > MAX_SHARD_BYTES:
                flush()
            current.append(name)
            current_bytes += nbytes
    flush()

    flattened = [name for shard in shards for name in shard]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(base_metadata):
        raise RuntimeError("Shard plan does not contain every base tensor exactly once")
    for shard in shards:
        size = sum(base_metadata[name]["nbytes"] for name in shard)
        if size > MAX_SHARD_BYTES and len(shard) != 1:
            raise RuntimeError(f"Planned shard exceeds 512 MiB: {size} bytes")
    for name in DEDICATED_TENSORS:
        if [name] not in shards:
            raise RuntimeError(f"Large tensor was not isolated: {name}")
    return shards


def main() -> None:
    with (BASE_DIR / INDEX_NAME).open() as handle:
        base_index = json.load(handle)
    weight_map = base_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("Base index has no weight_map object")
    base_metadata = load_base_metadata(weight_map)

    with (ADAPTER_DIR / "adapter_config.json").open() as handle:
        config = json.load(handle)
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    fan_in_fan_out = config.get("fan_in_fan_out")
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)):
        raise RuntimeError("Invalid LoRA rank or alpha")
    if not isinstance(fan_in_fan_out, bool):
        raise RuntimeError("fan_in_fan_out must be boolean")
    scale = float(alpha) / rank

    adapter = load_file(ADAPTER_DIR / "adapter_model.safetensors", device="cpu")
    pairs = adapter_pairs(adapter, base_metadata)
    pair_by_base = {base_name: (a, b) for base_name, a, b in pairs}

    # Required checks, all performed before creating any output checkpoint file.
    if len(pairs) != 32 or len(pair_by_base) != 32:
        raise RuntimeError(f"Expected exactly 32 adapter pairs, found {len(pairs)}")
    output_names = set(base_metadata)
    if any("lora_" in name for name in output_names):
        raise RuntimeError("Output tensor names contain an adapter tensor")
    q0 = "model.layers.0.self_attn.q_proj.weight"
    if base_metadata.get(q0, {}).get("shape") != [2048, 2048]:
        raise RuntimeError(f"{q0} does not have shape [2048, 2048]")
    if len(output_names) != 114:
        raise RuntimeError(f"Expected exactly 114 output tensors, found {len(output_names)}")
    if any(info["dtype"] != "F32" for info in base_metadata.values()):
        raise RuntimeError("Every base tensor must be float32")

    for base_name, a, b in pairs:
        update_shape = [b.shape[0], a.shape[1]]
        if fan_in_fan_out:
            update_shape.reverse()
        if update_shape != base_metadata[base_name]["shape"]:
            raise RuntimeError(f"LoRA update shape mismatch for {base_name}")

    shard_plan = plan_shards(base_metadata)
    shard_count = len(shard_plan)
    output_weight_map: dict[str, str] = {}
    total_size = sum(info["nbytes"] for info in base_metadata.values())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    index_path = OUTPUT_DIR / INDEX_NAME
    if index_path.exists():
        index_path.unlink()

    for shard_number, names in enumerate(shard_plan, start=1):
        output_name = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        source_shards = sorted({base_metadata[name]["shard"] for name in names})
        with ExitStack() as stack:
            readers = {
                shard: stack.enter_context(
                    safe_open(BASE_DIR / shard, framework="pt", device="cpu")
                )
                for shard in source_shards
            }
            for name in names:
                tensor = readers[base_metadata[name]["shard"]].get_tensor(name)
                if name in pair_by_base:
                    a, b = pair_by_base[name]
                    update = torch.matmul(b, a)
                    if fan_in_fan_out:
                        update = update.transpose(0, 1)
                    tensor = tensor + update * scale
                    if tensor.dtype != torch.float32:
                        raise RuntimeError(f"Merged tensor is not float32: {name}")
                tensors[name] = tensor.contiguous()
                output_weight_map[name] = output_name
            save_file(tensors, OUTPUT_DIR / output_name, metadata={"format": "pt"})

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    with index_path.open("w") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Merged {len(pairs)} LoRA pairs and wrote {len(output_names)} tensors "
        f"to {shard_count} shards in {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
