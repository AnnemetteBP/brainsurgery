"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Standalone script: no brainsurgery. Reads a sharded safetensors base
checkpoint plus a PEFT-style LoRA adapter, merges the adapter into the
adapted base weights (scale * B @ A), and writes a plain dense sharded
safetensors checkpoint with no adapter tensors.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(REPO_ROOT, "inputs", "base")
LORA_DIR = os.path.join(REPO_ROOT, "inputs", "lora")
OUT_DIR = os.path.join(REPO_ROOT, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes


def load_base_state_dict():
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    state_dict = {}
    dtype_map = {}
    for shard_file in shard_files:
        path = os.path.join(BASE_DIR, shard_file)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
                dtype_map[key] = f.get_slice(key).get_dtype()

    assert set(state_dict.keys()) == set(weight_map.keys()), (
        "base tensor names do not match model.safetensors.index.json weight_map"
    )
    return state_dict


def load_lora_state_dict():
    path = os.path.join(LORA_DIR, "adapter_model.safetensors")
    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def find_adapter_pairs(lora_state_dict):
    """Map each lora_A tensor to its lora_B counterpart and the base tensor name.

    Adapter keys look like:
      base_model.model.model.layers.<i>.<module>.lora_A.weight
      base_model.model.model.layers.<i>.<module>.lora_B.weight
    and map onto base tensor:
      model.layers.<i>.<module>.weight
    """
    pattern = re.compile(r"^base_model\.model\.(model\..+)\.lora_A\.weight$")
    pairs = []  # (base_key, a_key, b_key)
    for key in lora_state_dict:
        m = pattern.match(key)
        if m is None:
            continue
        base_key = m.group(1) + ".weight"
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        assert b_key in lora_state_dict, f"missing matching lora_B for {key}"
        pairs.append((base_key, key, b_key))
    return pairs


def pack_shards(names_and_sizes, max_bytes):
    """Greedily bin-pack tensor names into shards under a byte budget.

    Any tensor larger than half the shard budget is placed alone in its own
    shard (this is how the two ~412 MB embedding tensors, each well under
    the 512 MiB budget on their own but too large to share a shard with
    much else, end up isolated instead of being packed tightly with small
    tensors up to the budget).
    """
    shards = []
    current = []
    current_size = 0
    for name, size in names_and_sizes:
        if size > max_bytes // 2:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue
        if current and current_size + size > max_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)
    return shards


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main():
    base_state_dict = load_base_state_dict()
    lora_state_dict = load_lora_state_dict()

    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    assert not fan_in_fan_out, (
        "this script assumes fan_in_fan_out=False (nn.Linear [out, in] layout); "
        "adapter_config.json says otherwise"
    )
    scale = lora_alpha / r

    pairs = find_adapter_pairs(lora_state_dict)

    # --- Required checks (fail loudly before writing anything) ---
    assert len(pairs) == 32, f"expected exactly 32 adapter pairs, found {len(pairs)}"

    merged_count = 0
    for base_key, a_key, b_key in pairs:
        assert base_key in base_state_dict, f"base tensor {base_key} not found"
        base_weight = base_state_dict[base_key]
        a = lora_state_dict[a_key].to(torch.float32)
        b = lora_state_dict[b_key].to(torch.float32)

        delta = scale * (b @ a)  # [out, in], same layout as base_weight
        assert delta.shape == base_weight.shape, (
            f"shape mismatch merging {base_key}: delta {tuple(delta.shape)} "
            f"vs base {tuple(base_weight.shape)}"
        )

        merged = (base_weight.to(torch.float32) + delta).contiguous()
        assert merged.dtype == torch.float32
        base_state_dict[base_key] = merged
        merged_count += 1

    assert merged_count == 32, f"expected to merge 32 adapters, merged {merged_count}"

    output_state_dict = base_state_dict  # same 114 keys, 32 values replaced in place

    assert not any("lora_" in name for name in output_state_dict), (
        "adapter tensor leaked into output"
    )
    assert tuple(output_state_dict["model.layers.0.self_attn.q_proj.weight"].shape) == (
        2048,
        2048,
    ), "model.layers.0.self_attn.q_proj.weight has unexpected shape after merge"
    assert len(output_state_dict) == 114, (
        f"expected exactly 114 tensors in output, got {len(output_state_dict)}"
    )

    # --- Shard and write ---
    names_and_sizes = [
        (name, tensor_nbytes(tensor)) for name, tensor in output_state_dict.items()
    ]
    names_and_sizes.sort(key=lambda kv: kv[0])
    shards = pack_shards(names_and_sizes, MAX_SHARD_BYTES)

    os.makedirs(OUT_DIR, exist_ok=True)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    for shard_idx, names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {}
        for name in names:
            tensor = output_state_dict[name]
            shard_tensors[name] = tensor.contiguous()
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensor)
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Merged {merged_count} adapter pairs into {len(output_state_dict)} tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}")
    for shard_idx, names in enumerate(shards, start=1):
        shard_bytes = sum(tensor_nbytes(output_state_dict[n]) for n in names)
        print(f"  shard {shard_idx}: {len(names)} tensors, {shard_bytes} bytes")


if __name__ == "__main__":
    main()
