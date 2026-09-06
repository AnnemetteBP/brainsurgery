"""T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Merges the PEFT LoRA adapter directly into the base checkpoint tensors and
writes a sharded safetensors output, using only plain safetensors/torch I/O
(no model instantiation).
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = Path("inputs/base")
LORA_PATH = Path("inputs/lora/adapter_model.safetensors")
LORA_CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUT_DIR = Path("out/T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024

LORA_A_RE = re.compile(
    r"^base_model\.model\.(model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj))\.lora_A\.weight$"
)


def load_base_state_dict() -> dict[str, torch.Tensor]:
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    state_dict = {}
    for shard_name in shard_names:
        with safe_open(BASE_DIR / shard_name, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    assert set(state_dict.keys()) == set(weight_map.keys()), "base tensor set mismatch"
    return state_dict


def load_lora_state_dict() -> dict[str, torch.Tensor]:
    state_dict = {}
    with safe_open(LORA_PATH, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def merge(base_sd: dict[str, torch.Tensor], lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with open(LORA_CONFIG_PATH) as f:
        config = json.load(f)
    assert config["fan_in_fan_out"] is False, "fan_in_fan_out=True is not handled by this script"
    scale = config["lora_alpha"] / config["r"]

    merged = dict(base_sd)
    pairs_merged = 0
    for key in lora_sd:
        match = LORA_A_RE.match(key)
        if match is None:
            continue
        base_module = match.group(1)
        a_key = key
        b_key = key.replace("lora_A", "lora_B")
        assert b_key in lora_sd, f"missing paired lora_B for {a_key}"

        base_key = f"{base_module}.weight"
        assert base_key in merged, f"base tensor {base_key} not found"

        a = lora_sd[a_key].to(torch.float32)
        b = lora_sd[b_key].to(torch.float32)
        base_weight = merged[base_key]
        assert base_weight.dtype == torch.float32, f"{base_key} is not float32"

        delta = scale * (b @ a)
        assert delta.shape == base_weight.shape, (
            f"shape mismatch for {base_key}: delta {delta.shape} vs base {base_weight.shape}"
        )
        merged[base_key] = (base_weight + delta).contiguous()
        pairs_merged += 1

    assert pairs_merged == 32, f"expected 32 adapter pairs merged, got {pairs_merged}"
    assert not any("lora_" in name for name in merged), "adapter tensor leaked into output"
    assert len(merged) == 114, f"expected 114 tensors in output, got {len(merged)}"
    assert merged["model.layers.0.self_attn.q_proj.weight"].shape == (2048, 2048)

    return merged


def shard_and_save(state_dict: dict[str, torch.Tensor]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Greedy bin-packing in a stable (sorted) key order: oversized tensors get
    # their own shard, others are packed up to MAX_SHARD_BYTES.
    names = sorted(state_dict.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_bytes(state_dict[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    for i, names_in_shard in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: state_dict[name] for name in names_in_shard}
        save_file(shard_tensors, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in names_in_shard:
            weight_map[name] = shard_filename
            total_size += tensor_bytes(state_dict[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


def main() -> None:
    base_sd = load_base_state_dict()
    lora_sd = load_lora_state_dict()
    merged = merge(base_sd, lora_sd)
    shard_and_save(merged)
    print(f"Wrote {len(merged)} tensors to {OUT_DIR}")


if __name__ == "__main__":
    main()
