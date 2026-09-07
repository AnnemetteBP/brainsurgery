"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf)

Loads the base OLMo-1B checkpoint (sharded safetensors) and a PEFT-style
LoRA adapter, merges the adapter into the base q_proj/v_proj weights for
every layer, and writes out a plain dense sharded safetensors checkpoint
with no adapter tensors, respecting a 512 MiB per-shard tensor-data budget.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # out/T5/solution.py -> out/T5 -> out -> sandbox root
BASE_DIR = REPO_ROOT / "inputs" / "base"
LORA_PATH = REPO_ROOT / "inputs" / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = REPO_ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

LORA_A_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.(?:q_proj|v_proj))\.lora_A\.weight$"
)
LORA_B_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.(?:q_proj|v_proj))\.lora_B\.weight$"
)


def load_base_state_dict():
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    state_dict = {}
    for shard_file in shard_files:
        with safe_open(BASE_DIR / shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    assert set(state_dict.keys()) == set(weight_map.keys()), (
        "Base tensor keys do not match the index's weight_map"
    )
    return state_dict


def load_adapter_config():
    with open(ADAPTER_CONFIG_PATH) as f:
        cfg = json.load(f)
    assert cfg["fan_in_fan_out"] is False, (
        "This solution only handles fan_in_fan_out=False (no transposition)"
    )
    return cfg


def load_lora_pairs():
    """Returns {(layer_idx, module): {"A": tensor, "B": tensor}}."""
    pairs = {}
    with safe_open(LORA_PATH, framework="pt") as f:
        for key in f.keys():
            m_a = LORA_A_RE.match(key)
            m_b = LORA_B_RE.match(key)
            if m_a:
                layer_idx, module = int(m_a.group(1)), m_a.group(2)
                pairs.setdefault((layer_idx, module), {})["A"] = f.get_tensor(key)
            elif m_b:
                layer_idx, module = int(m_b.group(1)), m_b.group(2)
                pairs.setdefault((layer_idx, module), {})["B"] = f.get_tensor(key)
            else:
                raise ValueError(f"Unexpected tensor name in adapter: {key}")
    return pairs


def main():
    cfg = load_adapter_config()
    scale = cfg["lora_alpha"] / cfg["r"]

    base_sd = load_base_state_dict()
    lora_pairs = load_lora_pairs()

    # Required check: exactly 32 adapter pairs were found and merged.
    for (layer_idx, module), parts in lora_pairs.items():
        assert "A" in parts and "B" in parts, (
            f"Missing lora_A or lora_B for layer {layer_idx} module {module}"
        )
    assert len(lora_pairs) == 32, f"Expected 32 adapter pairs, found {len(lora_pairs)}"

    merged_sd = dict(base_sd)  # shallow copy; unmerged tensors kept as-is (unchanged)

    for (layer_idx, module), parts in lora_pairs.items():
        base_key = f"model.layers.{layer_idx}.{module}.weight"
        assert base_key in merged_sd, f"Base tensor {base_key} not found"

        A = parts["A"].to(torch.float32)  # [r, in]
        B = parts["B"].to(torch.float32)  # [out, r]
        base_weight = merged_sd[base_key].to(torch.float32)

        delta = scale * (B @ A)  # [out, in]
        assert delta.shape == base_weight.shape, (
            f"Shape mismatch merging {base_key}: delta {delta.shape} vs base {base_weight.shape}"
        )

        merged_sd[base_key] = (base_weight + delta).to(torch.float32)

    # Required checks -----------------------------------------------------
    assert not any("lora_" in k for k in merged_sd), "Adapter tensor leaked into output"
    q0 = merged_sd["model.layers.0.self_attn.q_proj.weight"]
    assert tuple(q0.shape) == (2048, 2048), f"Unexpected shape for layer 0 q_proj: {q0.shape}"
    assert len(merged_sd) == 114, f"Expected 114 tensors in output, found {len(merged_sd)}"

    # Sharding: greedy-pack tensors into shards under the byte budget,
    # oversized tensors get their own shard.
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names_in_order = list(merged_sd.keys())
    shards: list[list[str]] = []
    current_shard: list[str] = []
    current_bytes = 0

    for name in names_in_order:
        nbytes = tensor_nbytes(merged_sd[name])
        if nbytes > MAX_SHARD_BYTES:
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_bytes = 0
            shards.append([name])
            continue
        if current_bytes + nbytes > MAX_SHARD_BYTES and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_bytes = 0
        current_shard.append(name)
        current_bytes += nbytes

    if current_shard:
        shards.append(current_shard)

    for shard_names in shards:
        total = sum(tensor_nbytes(merged_sd[n]) for n in shard_names)
        assert total <= MAX_SHARD_BYTES or len(shard_names) == 1, (
            f"Shard exceeds byte budget: {total} bytes across {len(shard_names)} tensors"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    for shard_idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: merged_sd[name].contiguous() for name in shard_names}
        save_file(shard_tensors, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(merged_sd[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {len(lora_pairs)} adapter pairs into {len(merged_sd)} tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}.")


if __name__ == "__main__":
    main()
