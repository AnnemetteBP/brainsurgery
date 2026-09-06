"""
T5: LoRA adapter merge with sharded export (GPT-2 124M).

Merges a PEFT-style LoRA adapter into the base GPT-2 checkpoint's
attn.c_attn weights, then writes a sharded safetensors checkpoint.
"""

import json
import os
import re

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_PATH = os.path.join(REPO_ROOT, "inputs", "base", "model.safetensors")
LORA_PATH = os.path.join(REPO_ROOT, "inputs", "lora", "adapter_model.safetensors")
LORA_CONFIG_PATH = os.path.join(REPO_ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(HERE)

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 104,857,600 bytes


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    with open(LORA_CONFIG_PATH) as f:
        lora_config = json.load(f)

    assert lora_config["fan_in_fan_out"] is True, "expected fan_in_fan_out=True"
    r = lora_config["r"]
    lora_alpha = lora_config["lora_alpha"]
    scale = lora_alpha / r

    base_tensors = load_safetensors(BASE_PATH)
    lora_tensors = load_safetensors(LORA_PATH)

    # Map lora_A names to their base module name.
    # e.g. "base_model.model.h.0.attn.c_attn.lora_A.weight"
    #   -> base name "h.0.attn.c_attn.weight"
    lora_a_pattern = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")

    merged_pairs = 0
    for key in list(lora_tensors.keys()):
        m = lora_a_pattern.match(key)
        if m is None:
            continue
        module_name = m.group(1)
        base_key = f"{module_name}.weight"
        b_key = key.replace("lora_A", "lora_B")

        assert base_key in base_tensors, f"missing base tensor {base_key}"
        assert b_key in lora_tensors, f"missing paired lora_B tensor {b_key}"

        A = lora_tensors[key].to(torch.float32)
        B = lora_tensors[b_key].to(torch.float32)

        delta = scale * (B @ A).T
        base = base_tensors[base_key].to(torch.float32)
        assert delta.shape == base.shape, (
            f"shape mismatch for {base_key}: delta {delta.shape} vs base {base.shape}"
        )
        base_tensors[base_key] = (base + delta).contiguous()
        merged_pairs += 1

    # Required checks.
    if merged_pairs != 12:
        raise RuntimeError(f"expected exactly 12 adapter pairs merged, got {merged_pairs}")

    for name in base_tensors:
        if "lora_" in name:
            raise RuntimeError(f"adapter tensor leaked into output: {name}")

    expected_shape = (768, 2304)
    actual_shape = tuple(base_tensors["h.0.attn.c_attn.weight"].shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"h.0.attn.c_attn.weight has shape {actual_shape}, expected {expected_shape}"
        )

    if len(base_tensors) != 160:
        raise RuntimeError(f"expected exactly 160 tensors in output, got {len(base_tensors)}")

    # --- Shard the output ---
    os.makedirs(OUT_DIR, exist_ok=True)

    def tensor_nbytes(t):
        return t.numel() * t.element_size()

    # Deterministic ordering.
    names = sorted(base_tensors.keys())

    shards = []  # list of dict name->tensor
    current_shard = {}
    current_bytes = 0
    for name in names:
        t = base_tensors[name]
        nbytes = tensor_nbytes(t)
        if nbytes > MAX_SHARD_BYTES:
            # goes alone in its own shard
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_bytes = 0
            shards.append({name: t})
            continue
        if current_bytes + nbytes > MAX_SHARD_BYTES and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_bytes = 0
        current_shard[name] = t
        current_bytes += nbytes
    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    if num_shards == 1:
        shard_filename = "model.safetensors"
        save_file(shards[0], os.path.join(OUT_DIR, shard_filename), metadata={"format": "pt"})
        for name, t in shards[0].items():
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(t)
    else:
        for idx, shard in enumerate(shards, start=1):
            shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
            save_file(shard, os.path.join(OUT_DIR, shard_filename), metadata={"format": "pt"})
            for name, t in shard.items():
                weight_map[name] = shard_filename
                total_size += tensor_nbytes(t)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {merged_pairs} LoRA pairs into {len(base_tensors)} tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
