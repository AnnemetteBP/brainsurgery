#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUT_DIR = Path(__file__).resolve().parent
MAX_SHARD_BYTES = 512 * 1024 * 1024
FORCE_SINGLE = {"gpt_neox.embed_in.weight", "embed_out.weight"}


def tensor_nbytes(handle, name):
    view = handle.get_slice(name)
    shape = view.get_shape()
    dtype = view.get_dtype()
    bytes_per_element = {
        "BOOL": 1, "U8": 1, "I8": 1,
        "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
        "U32": 4, "I32": 4, "F32": 4,
        "U64": 8, "I64": 8, "F64": 8,
    }[dtype]
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * bytes_per_element


def make_shards(base, names):
    """Return ordered key lists whose tensor payload is within the cap."""
    shards = []
    current = []
    current_bytes = 0
    for name in names:
        size = tensor_nbytes(base, name)
        if size > MAX_SHARD_BYTES:
            raise AssertionError(f"single tensor exceeds shard cap: {name} ({size} bytes)")
        if name in FORCE_SINGLE:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        shards.append(current)
    return shards


def main():
    config = json.loads(CONFIG_PATH.read_text())
    rank = config["r"]
    scale = config["lora_alpha"] / rank
    if config.get("fan_in_fan_out") is not False:
        raise AssertionError("this solution expects fan_in_fan_out=false")

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base, \
         safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter:
        base_names = list(base.keys())
        adapter_names = set(adapter.keys())
        a_names = sorted(name for name in adapter_names if name.endswith(".lora_A.weight"))
        b_names = {name for name in adapter_names if name.endswith(".lora_B.weight")}
        pairs = {}
        for a_name in a_names:
            b_name = a_name.removesuffix(".lora_A.weight") + ".lora_B.weight"
            if b_name not in b_names:
                raise AssertionError(f"missing B factor for {a_name}")
            base_name = a_name.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight"
            if base_name not in base_names:
                raise AssertionError(f"mapped base tensor not found: {base_name}")
            pairs[base_name] = (a_name, b_name)

        # All required validation happens before creating any checkpoint file.
        if len(pairs) != 16 or len(a_names) != 16 or len(b_names) != 16:
            raise AssertionError(
                f"expected exactly 16 complete adapter pairs, got "
                f"{len(a_names)} A, {len(b_names)} B, {len(pairs)} mapped"
            )
        if adapter_names != set(a_names) | b_names:
            raise AssertionError("adapter contains tensors other than the 16 A/B pairs")
        if any("lora_" in name for name in base_names):
            raise AssertionError("output key set would contain a LoRA tensor")
        probe = "gpt_neox.layers.0.attention.query_key_value.weight"
        if base.get_slice(probe).get_shape() != [6144, 2048]:
            raise AssertionError(f"unexpected probe shape: {base.get_slice(probe).get_shape()}")
        if len(base_names) != 244:
            raise AssertionError(f"expected 244 output tensors, got {len(base_names)}")

        shards = make_shards(base, base_names)
        total = len(shards)
        weight_map = {}
        total_size = sum(tensor_nbytes(base, name) for name in base_names)
        for number, shard_names in enumerate(shards, 1):
            filename = f"model-{number:05d}-of-{total:05d}.safetensors"
            tensors = {}
            for name in shard_names:
                original = base.get_tensor(name)
                if name in pairs:
                    a_name, b_name = pairs[name]
                    a = adapter.get_tensor(a_name)
                    b = adapter.get_tensor(b_name)
                    if list(a.shape) != [rank, original.shape[1]]:
                        raise AssertionError(f"bad A shape for {name}: {list(a.shape)}")
                    if list(b.shape) != [original.shape[0], rank]:
                        raise AssertionError(f"bad B shape for {name}: {list(b.shape)}")
                    merged = original.float().add(torch.matmul(b.float(), a.float()), alpha=scale)
                    tensors[name] = merged.to(original.dtype).contiguous()
                else:
                    tensors[name] = original
                weight_map[name] = filename
            payload = sum(t.numel() * t.element_size() for t in tensors.values())
            if payload > MAX_SHARD_BYTES:
                raise AssertionError(f"shard payload exceeds cap: {filename} ({payload} bytes)")
            save_file(tensors, OUT_DIR / filename, metadata={"format": "pt"})

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"Merged {len(pairs)} LoRA pairs and wrote {len(weight_map)} tensors in {total} shards")


if __name__ == "__main__":
    main()
