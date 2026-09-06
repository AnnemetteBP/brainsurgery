"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf)

Merges a PEFT-style LoRA adapter into the base checkpoint's q_proj/v_proj
weights ("merge and unload"), then writes the result as a sharded
safetensors checkpoint under out/T5/, with each shard holding at most
512 MiB of tensor data (a single tensor bigger than that limit is stored
alone in its own shard).
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(SANDBOX, "inputs", "base")
LORA_DIR = os.path.join(SANDBOX, "inputs", "lora")
OUT_DIR = os.path.join(SANDBOX, "out", "T5")

SHARD_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MiB


def load_base_index():
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    return index["weight_map"]  # tensor name -> shard filename, in file order


def load_lora_config():
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base_weight_map = load_base_index()
    base_names = list(base_weight_map.keys())
    if len(base_names) != 114:
        raise AssertionError(f"expected 114 base tensors, found {len(base_names)}")

    lora_config = load_lora_config()
    r = lora_config["r"]
    alpha = lora_config["lora_alpha"]
    fan_in_fan_out = lora_config["fan_in_fan_out"]
    if fan_in_fan_out:
        raise AssertionError("this solution only handles fan_in_fan_out = false")
    scale = alpha / r

    # Open all base shard files (lazily) keyed by filename.
    base_shard_files = sorted(set(base_weight_map.values()))
    base_handles = {
        fname: safe_open(os.path.join(BASE_DIR, fname), framework="pt", device="cpu")
        for fname in base_shard_files
    }

    # Collect LoRA A/B pairs, mapping adapter names -> base module name.
    adapter_path = os.path.join(LORA_DIR, "adapter_model.safetensors")
    pairs = {}  # base_module_name -> {"A": tensor, "B": tensor}
    with safe_open(adapter_path, framework="pt", device="cpu") as lora_handle:
        for key in lora_handle.keys():
            if "lora_" not in key:
                raise AssertionError(f"unexpected non-lora tensor in adapter file: {key}")
            prefix = "base_model.model."
            if not key.startswith(prefix):
                raise AssertionError(f"unexpected adapter key prefix: {key}")
            rest = key[len(prefix):]
            if rest.endswith(".lora_A.weight"):
                module = rest[: -len(".lora_A.weight")]
                slot = "A"
            elif rest.endswith(".lora_B.weight"):
                module = rest[: -len(".lora_B.weight")]
                slot = "B"
            else:
                raise AssertionError(f"unrecognized adapter tensor name: {key}")
            base_name = module + ".weight"
            pairs.setdefault(base_name, {})[slot] = lora_handle.get_tensor(key)

    # Required check: exactly 32 adapter pairs.
    complete_pairs = {
        name: t for name, t in pairs.items() if "A" in t and "B" in t
    }
    if len(complete_pairs) != 32:
        raise AssertionError(
            f"expected exactly 32 adapter pairs, found {len(complete_pairs)}"
        )
    for name, t in pairs.items():
        if name not in complete_pairs:
            raise AssertionError(f"incomplete adapter pair for {name}: {sorted(t.keys())}")
    for name in complete_pairs:
        if name not in base_weight_map:
            raise AssertionError(f"adapter targets unknown base tensor: {name}")

    # Build merged output tensors in original base order.
    merged = {}
    for name in base_names:
        fname = base_weight_map[name]
        tensor = base_handles[fname].get_tensor(name)
        if name in complete_pairs:
            a = complete_pairs[name]["A"].to(torch.float32)
            b = complete_pairs[name]["B"].to(torch.float32)
            delta = scale * (b @ a)
            tensor = tensor.to(torch.float32) + delta
        merged[name] = tensor.contiguous()

    del base_handles

    # Required checks before writing.
    if any("lora_" in name for name in merged):
        raise AssertionError("output contains a tensor with 'lora_' in its name")
    q0_shape = tuple(merged["model.layers.0.self_attn.q_proj.weight"].shape)
    if q0_shape != (2048, 2048):
        raise AssertionError(f"model.layers.0.self_attn.q_proj.weight has shape {q0_shape}")
    if len(merged) != 114:
        raise AssertionError(f"expected 114 output tensors, found {len(merged)}")

    # Greedy shard packing, preserving base tensor order. A tensor whose own
    # size exceeds the shard limit is stored alone in its own shard.
    sizes = {name: t.numel() * t.element_size() for name, t in merged.items()}
    shards = []  # list[list[str]]
    current = []
    current_size = 0
    for name in base_names:
        size = sizes[name]
        if size > SHARD_LIMIT_BYTES:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map = {}
    shard_filenames = []
    for i, names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_filenames.append(shard_filename)
        shard_tensors = {name: merged[name] for name in names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))
        for name in names:
            weight_map[name] = shard_filename

    total_size = sum(sizes.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {len(complete_pairs)} adapter pairs into {len(merged)} tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}: {shard_filenames}")


if __name__ == "__main__":
    main()
