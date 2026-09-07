"""LoRA merge-and-unload for Pythia-1B, with sharded safetensors export."""

import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
INPUTS = os.path.join(ROOT, "inputs")
OUT_DIR = BASE_DIR

MAX_SHARD_BYTES = 512 * 1024 * 1024


def main():
    base_path = os.path.join(INPUTS, "base", "model.safetensors")
    lora_path = os.path.join(INPUTS, "lora", "adapter_model.safetensors")
    config_path = os.path.join(INPUTS, "lora", "adapter_config.json")

    with open(config_path) as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    if fan_in_fan_out:
        raise NotImplementedError("fan_in_fan_out=True is not supported by this script")
    scale = lora_alpha / r

    base_tensors = load_file(base_path)
    lora_tensors = load_file(lora_path)

    n_base = len(base_tensors)

    # Find lora_A/lora_B pairs and map to base tensor names.
    a_pattern = re.compile(r"^(.*)\.lora_A\.weight$")
    pairs = {}  # base_name -> (A, B)
    for name, tensor in lora_tensors.items():
        m = a_pattern.match(name)
        if not m:
            continue
        prefix = m.group(1)
        b_name = f"{prefix}.lora_B.weight"
        if b_name not in lora_tensors:
            raise ValueError(f"missing matching lora_B for {name}")
        # Map adapter name to base checkpoint name.
        # e.g. base_model.model.gpt_neox.layers.0.attention.query_key_value.lora_A.weight
        # -> gpt_neox.layers.0.attention.query_key_value.weight
        base_name_match = re.search(r"(gpt_neox\..*)$", prefix)
        if base_name_match is None:
            raise ValueError(f"cannot map adapter tensor name to base name: {name}")
        base_name = base_name_match.group(1) + ".weight"
        pairs[base_name] = (lora_tensors[name], lora_tensors[b_name])

    if len(pairs) != 16:
        raise AssertionError(f"expected exactly 16 adapter pairs, found {len(pairs)}")

    for base_name, (A, B) in pairs.items():
        if base_name not in base_tensors:
            raise ValueError(f"base checkpoint has no tensor named {base_name}")
        base_w = base_tensors[base_name]
        base_dtype = base_w.dtype
        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        if delta.shape != base_w.shape:
            raise ValueError(
                f"shape mismatch merging {base_name}: base {tuple(base_w.shape)} "
                f"vs delta {tuple(delta.shape)}"
            )
        merged = (base_w.to(torch.float32) + delta).to(base_dtype)
        base_tensors[base_name] = merged

    # Required checks.
    if any("lora_" in name for name in base_tensors):
        raise AssertionError("adapter tensor name (containing 'lora_') found in output")

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base_tensors[qkv0].shape) != (6144, 2048):
        raise AssertionError(
            f"{qkv0} has shape {tuple(base_tensors[qkv0].shape)}, expected (6144, 2048)"
        )

    if len(base_tensors) != n_base or len(base_tensors) != 244:
        raise AssertionError(f"expected exactly 244 tensors in output, got {len(base_tensors)}")

    # Shard: bin-pack by descending size, splitting off any tensor bigger than
    # the shard limit into its own shard.
    def tensor_nbytes(t):
        return t.numel() * t.element_size()

    names_sorted = sorted(base_tensors.keys(), key=lambda n: tensor_nbytes(base_tensors[n]), reverse=True)

    shard_name_lists = []
    totals = []
    for name in names_sorted:
        size = tensor_nbytes(base_tensors[name])
        placed_idx = None
        if size <= MAX_SHARD_BYTES:
            for i, total in enumerate(totals):
                if total + size <= MAX_SHARD_BYTES:
                    placed_idx = i
                    break
        if placed_idx is None:
            shard_name_lists.append([name])
            totals.append(size)
        else:
            shard_name_lists[placed_idx].append(name)
            totals[placed_idx] += size

    weight_map = {}
    total_out_tensors = 0
    for i, names in enumerate(shard_name_lists):
        shard_file = f"model-{i+1:05d}-of-{len(shard_name_lists):05d}.safetensors"
        shard_tensors = {name: base_tensors[name] for name in names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_file), metadata={"format": "pt"})
        for name in names:
            weight_map[name] = shard_file
        total_out_tensors += len(names)

    if total_out_tensors != 244:
        raise AssertionError(f"wrote {total_out_tensors} tensors across shards, expected 244")

    total_size = sum(tensor_nbytes(t) for t in base_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {len(pairs)} LoRA pairs into base checkpoint.")
    print(f"Wrote {len(shard_name_lists)} shards, {total_out_tensors} tensors total, to {OUT_DIR}")


if __name__ == "__main__":
    main()
