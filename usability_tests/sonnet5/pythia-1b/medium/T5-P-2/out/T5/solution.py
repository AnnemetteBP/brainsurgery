"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Merges a PEFT-style LoRA adapter into the base gpt_neox checkpoint
("merge and unload") and writes the result as a sharded safetensors
checkpoint under out/T5/, with shards capped at 512 MiB of tensor data
(oversized single tensors get their own shard).
"""

import json
import os
import re

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.dirname(os.path.dirname(HERE))  # out/T5 -> out -> task root
INPUTS = os.path.join(TASK_ROOT, "inputs")
OUT_DIR = os.path.join(TASK_ROOT, "out", "T5")

BASE_PATH = os.path.join(INPUTS, "base", "model.safetensors")
LORA_PATH = os.path.join(INPUTS, "lora", "adapter_model.safetensors")
LORA_CONFIG_PATH = os.path.join(INPUTS, "lora", "adapter_config.json")

SHARD_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB, tensor data only

LORA_A_RE = re.compile(
    r"^base_model\.model\.gpt_neox\.layers\.(\d+)\.(.+)\.lora_A\.weight$"
)
LORA_B_RE = re.compile(
    r"^base_model\.model\.gpt_neox\.layers\.(\d+)\.(.+)\.lora_B\.weight$"
)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(LORA_CONFIG_PATH) as f:
        lora_config = json.load(f)

    r = lora_config["r"]
    lora_alpha = lora_config["lora_alpha"]
    fan_in_fan_out = lora_config["fan_in_fan_out"]
    scale = lora_alpha / r

    if fan_in_fan_out:
        raise RuntimeError(
            "This script only handles fan_in_fan_out=false; "
            f"got {fan_in_fan_out!r}"
        )

    # Load base tensors (float16), keeping original dtype/shape metadata.
    base_tensors: dict[str, torch.Tensor] = {}
    with safe_open(BASE_PATH, framework="pt") as f:
        for key in f.keys():
            base_tensors[key] = f.get_tensor(key)

    # Load adapter tensors and pair up lora_A / lora_B by (layer, module).
    lora_tensors: dict[str, torch.Tensor] = {}
    with safe_open(LORA_PATH, framework="pt") as f:
        for key in f.keys():
            lora_tensors[key] = f.get_tensor(key)

    pairs: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in lora_tensors.items():
        m = LORA_A_RE.match(key)
        if m:
            layer, module = m.group(1), m.group(2)
            pairs.setdefault((layer, module), {})["A"] = tensor
            continue
        m = LORA_B_RE.match(key)
        if m:
            layer, module = m.group(1), m.group(2)
            pairs.setdefault((layer, module), {})["B"] = tensor
            continue
        raise RuntimeError(f"Unrecognized adapter tensor name: {key}")

    if len(pairs) != 16:
        raise RuntimeError(
            f"Expected exactly 16 adapter pairs, found {len(pairs)}"
        )

    merged_count = 0
    for (layer, module), factors in pairs.items():
        if "A" not in factors or "B" not in factors:
            raise RuntimeError(
                f"Incomplete adapter pair for layer {layer} module {module}: "
                f"found {list(factors.keys())}"
            )
        A = factors["A"]
        B = factors["B"]

        base_key = f"gpt_neox.layers.{layer}.{module}.weight"
        if base_key not in base_tensors:
            raise RuntimeError(f"Base tensor not found for adapter target: {base_key}")

        base_weight = base_tensors[base_key]
        base_dtype = base_weight.dtype
        expected_shape = tuple(base_weight.shape)

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        if tuple(delta.shape) != expected_shape:
            raise RuntimeError(
                f"Shape mismatch merging {base_key}: delta {tuple(delta.shape)} "
                f"vs base {expected_shape}"
            )

        merged = base_weight.to(torch.float32) + delta
        base_tensors[base_key] = merged.to(base_dtype).contiguous()
        merged_count += 1

    if merged_count != 16:
        raise RuntimeError(f"Expected to merge 16 tensors, merged {merged_count}")

    # --- Required checks ---
    if any("lora_" in k for k in base_tensors):
        raise RuntimeError("Adapter tensor(s) leaked into output")

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base_tensors[qkv0].shape) != (6144, 2048):
        raise RuntimeError(
            f"{qkv0} has unexpected shape {tuple(base_tensors[qkv0].shape)}"
        )

    if len(base_tensors) != 244:
        raise RuntimeError(f"Expected 244 tensors in output, got {len(base_tensors)}")

    # --- Shard and write ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Deterministic ordering.
    names = sorted(base_tensors.keys())

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in names:
        nbytes = tensor_nbytes(base_tensors[name])
        if nbytes > SHARD_MAX_BYTES:
            # Oversized tensor gets its own shard.
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > SHARD_MAX_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes

    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    if num_shards == 1:
        shard_filename = "model.safetensors"
        shard_tensors = {name: base_tensors[name] for name in shards[0]}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename), metadata={"format": "pt"})
        for name in shards[0]:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(base_tensors[name])
    else:
        width = len(str(num_shards))
        for idx, shard_names in enumerate(shards, start=1):
            shard_filename = (
                f"model-{idx:0{width}d}-of-{num_shards:0{width}d}.safetensors"
            )
            shard_tensors = {name: base_tensors[name] for name in shard_names}
            save_file(
                shard_tensors, os.path.join(OUT_DIR, shard_filename), metadata={"format": "pt"}
            )
            for name in shard_names:
                weight_map[name] = shard_filename
                total_size += tensor_nbytes(base_tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Merged {merged_count} LoRA pairs into base weights.")
    print(f"Wrote {len(names)} tensors across {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
