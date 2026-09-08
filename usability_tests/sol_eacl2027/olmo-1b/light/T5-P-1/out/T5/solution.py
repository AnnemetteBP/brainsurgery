#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024
SINGLETON_TENSORS = {"model.embed_tokens.weight", "lm_head.weight"}


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def main():
    with (BASE_DIR / "model.safetensors.index.json").open() as f:
        base_index = json.load(f)
    with (LORA_DIR / "adapter_config.json").open() as f:
        config = json.load(f)

    weight_map = base_index["weight_map"]
    base_names = list(weight_map)
    adapter_path = LORA_DIR / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter_file:
        adapter_names = set(adapter_file.keys())

    pairs = {}
    prefix = "base_model.model."
    a_suffix = ".lora_A.weight"
    for a_name in sorted(name for name in adapter_names if name.endswith(a_suffix)):
        stem = a_name[: -len(a_suffix)]
        b_name = stem + ".lora_B.weight"
        if b_name not in adapter_names:
            raise RuntimeError(f"missing LoRA B tensor for {a_name}")
        if not stem.startswith(prefix):
            raise RuntimeError(f"unexpected adapter tensor name: {a_name}")
        base_name = stem[len(prefix):] + ".weight"
        if base_name not in weight_map:
            raise RuntimeError(f"adapter target is absent from base: {base_name}")
        pairs[base_name] = (a_name, b_name)

    paired_adapter_names = {name for pair in pairs.values() for name in pair}
    if paired_adapter_names != adapter_names:
        extra = sorted(adapter_names - paired_adapter_names)
        raise RuntimeError(f"unpaired or unexpected adapter tensors: {extra}")
    if len(pairs) != 32:
        raise RuntimeError(f"expected exactly 32 adapter pairs, found {len(pairs)}")
    if len(base_names) != 114:
        raise RuntimeError(f"expected exactly 114 output tensors, found {len(base_names)}")
    if any("lora_" in name for name in base_names):
        raise RuntimeError("LoRA tensor name would appear in output")

    probe_name = "model.layers.0.self_attn.q_proj.weight"
    probe_shard = BASE_DIR / weight_map[probe_name]
    with safe_open(probe_shard, framework="pt", device="cpu") as f:
        if list(f.get_slice(probe_name).get_shape()) != [2048, 2048]:
            raise RuntimeError(f"{probe_name} does not have shape [2048, 2048]")

    rank = config["r"]
    scale = float(config["lora_alpha"]) / rank
    transpose_delta = bool(config.get("fan_in_fan_out", False))
    adapter = load_file(adapter_path, device="cpu")

    # Form shard groups from metadata before writing. The named large tensors
    # are deliberately singleton shards, as required by the task.
    groups = []
    current = []
    current_bytes = 0
    shard_handles = {}
    try:
        for name in base_names:
            shard_name = weight_map[name]
            if shard_name not in shard_handles:
                shard_handles[shard_name] = safe_open(
                    BASE_DIR / shard_name, framework="pt", device="cpu"
                )
            size = tensor_bytes(shard_handles[shard_name].get_tensor(name))
            if size > MAX_SHARD_BYTES:
                raise RuntimeError(f"single tensor exceeds shard limit: {name}")
            if name in SINGLETON_TENSORS:
                if current:
                    groups.append(current)
                    current, current_bytes = [], 0
                groups.append([name])
            else:
                if current and current_bytes + size > MAX_SHARD_BYTES:
                    groups.append(current)
                    current, current_bytes = [], 0
                current.append(name)
                current_bytes += size
        if current:
            groups.append(current)

        output_map = {}
        total_size = 0
        shard_count = len(groups)
        for shard_number, names in enumerate(groups, 1):
            tensors = {}
            for name in names:
                tensor = shard_handles[weight_map[name]].get_tensor(name)
                if name in pairs:
                    a_name, b_name = pairs[name]
                    a = adapter[a_name].float()
                    b = adapter[b_name].float()
                    if a.shape != (rank, tensor.shape[1]) or b.shape != (tensor.shape[0], rank):
                        raise RuntimeError(f"incompatible LoRA shapes for {name}")
                    delta = torch.matmul(b, a)
                    if transpose_delta:
                        delta = delta.T
                    tensor = tensor.float() + scale * delta
                tensors[name] = tensor.contiguous()
                total_size += tensor_bytes(tensors[name])

            actual_bytes = sum(tensor_bytes(t) for t in tensors.values())
            if actual_bytes > MAX_SHARD_BYTES:
                raise RuntimeError(f"constructed shard exceeds size limit: {actual_bytes}")
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            save_file(tensors, OUT_DIR / filename, metadata={"format": "pt"})
            output_map.update({name: filename for name in names})

        if set(output_map) != set(base_names):
            raise RuntimeError("output index key set differs from base key set")
        output_index = {"metadata": {"total_size": total_size}, "weight_map": output_map}
        with (OUT_DIR / "model.safetensors.index.json").open("w") as f:
            json.dump(output_index, f, indent=2, sort_keys=True)
            f.write("\n")
    finally:
        shard_handles.clear()

    print(f"Merged {len(pairs)} LoRA pairs into {len(base_names)} tensors across {len(groups)} shards")


if __name__ == "__main__":
    main()
