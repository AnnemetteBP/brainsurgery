"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Loads the base checkpoint and the PEFT-style LoRA adapter directly as
safetensors state dicts (no model instantiation needed), merges
`weight += (alpha/r) * B @ A` for every adapted q_proj/v_proj, and writes
a sharded safetensors checkpoint under out/T5/ with a max shard size of
512 MiB of tensor data and large tensors (embed_tokens, lm_head) isolated
in their own shard.

Usage: python solution.py
Reads from ../../inputs/{base,lora}, writes into this directory.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
BASE_DIR = INPUTS / "base"
LORA_DIR = INPUTS / "lora"
OUT_DIR = HERE  # out/T5/

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data

# Tensors at least this big get their own shard even though they fit under
# MAX_SHARD_BYTES alone (spec calls out embed_tokens/lm_head, 412 MB each,
# as always-alone; anything past the mid-point of a shard is bin-packed
# alone too, since packing it with siblings would immediately overflow the
# same shard for almost any other tensor in this checkpoint).
ALONE_THRESHOLD_BYTES = MAX_SHARD_BYTES // 2

LORA_A_RE = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.(.+)\.lora_B\.weight$")


def load_base_state_dict() -> dict[str, torch.Tensor]:
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    state: dict[str, torch.Tensor] = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        with safe_open(BASE_DIR / shard_file, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)

    assert set(state.keys()) == set(weight_map.keys()), (
        "base state dict keys do not match index.json weight_map"
    )
    return state


def load_lora_state_dict() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(LORA_DIR / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def main() -> None:
    with open(LORA_DIR / "adapter_config.json") as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    assert not fan_in_fan_out, (
        "this script only implements the fan_in_fan_out=False (nn.Linear [out, in]) layout"
    )
    scale = lora_alpha / r

    base_state = load_base_state_dict()
    lora_state = load_lora_state_dict()

    # Pair up lora_A / lora_B tensors by their base module path.
    a_keys = {}
    b_keys = {}
    for key in lora_state:
        m = LORA_A_RE.match(key)
        if m:
            a_keys[m.group(1)] = key
            continue
        m = LORA_B_RE.match(key)
        if m:
            b_keys[m.group(1)] = key
            continue
        raise AssertionError(f"unrecognized adapter tensor name: {key}")

    assert a_keys.keys() == b_keys.keys(), "lora_A/lora_B module sets do not match"
    module_paths = sorted(a_keys.keys())

    # Required check: exactly 32 adapter pairs found.
    assert len(module_paths) == 32, (
        f"expected exactly 32 adapter pairs, found {len(module_paths)}"
    )

    merged_count = 0
    for module_path in module_paths:
        base_key = f"{module_path}.weight"
        assert base_key in base_state, f"base checkpoint is missing {base_key}"

        A = lora_state[a_keys[module_path]].to(torch.float32)  # [r, in]
        B = lora_state[b_keys[module_path]].to(torch.float32)  # [out, r]
        delta = scale * (B @ A)  # [out, in], nn.Linear layout, no transpose needed

        base_tensor = base_state[base_key]
        assert base_tensor.dtype == torch.float32, f"{base_key} is not float32"
        assert delta.shape == base_tensor.shape, (
            f"shape mismatch for {base_key}: base {tuple(base_tensor.shape)} "
            f"vs delta {tuple(delta.shape)}"
        )

        base_state[base_key] = (base_tensor + delta).contiguous()
        merged_count += 1

    assert merged_count == 32, f"expected to merge 32 weights, merged {merged_count}"

    # Required check: no adapter tensor leaks into the output.
    assert all("lora_" not in k for k in base_state), "an adapter tensor leaked into the output"

    # Required check: q_proj shape preserved.
    q0 = base_state["model.layers.0.self_attn.q_proj.weight"]
    assert tuple(q0.shape) == (2048, 2048), f"unexpected shape for layer 0 q_proj: {tuple(q0.shape)}"

    # Required check: tensor count unchanged.
    assert len(base_state) == 114, f"expected 114 tensors in output, got {len(base_state)}"

    # --- Shard and write ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.element_size() * t.nelement()

    names = list(base_state.keys())  # preserves base checkpoint's key order
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in names:
        nbytes = tensor_nbytes(base_state[name])
        if nbytes >= ALONE_THRESHOLD_BYTES:
            # Oversized (or ">half a shard") tensor gets its own shard.
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes

    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: base_state[name] for name in shard_names}
        save_file(shard_tensors, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(base_state[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {merged_count} LoRA pairs (scale={scale}).")
    print(f"Wrote {len(names)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
