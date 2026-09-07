#!/usr/bin/env python3
"""T5: LoRA merge-and-unload with sharded safetensors export (OLMo-1B).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). The
merge math is simple enough (scale * B @ A, no transpose since
fan_in_fan_out=False) that doing it directly on the state dicts, with our own
checks and our own shard packer, gives full control over the "Required
checks" and the exact shard-size rule -- rather than trusting
`peft.merge_and_unload()` + `transformers.save_pretrained(max_shard_size=...)`
to hit a byte-exact 512 MiB (=536,870,912 B) budget and to leave literally
nothing but the 114 base tensor names behind.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"

SHARD_BUDGET = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

ADAPTER_KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.(?:q_proj|v_proj))\.lora_([AB])\.weight$"
)


def load_base_state_dict() -> dict[str, torch.Tensor]:
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        with safe_open(BASE_DIR / shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    assert set(state_dict.keys()) == set(weight_map.keys()), (
        "base state dict keys do not match index.json weight_map"
    )
    return state_dict


def load_adapter_pairs() -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    """Group adapter tensors into {(layer, module): {"A": ..., "B": ...}}."""
    pairs: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    with safe_open(LORA_DIR / "adapter_model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            m = ADAPTER_KEY_RE.match(key)
            if m is None:
                raise ValueError(f"unrecognized adapter tensor name: {key!r}")
            layer, module, ab = int(m.group(1)), m.group(2), m.group(3)
            pairs.setdefault((layer, module), {})[ab] = f.get_tensor(key)
    for loc, tensors in pairs.items():
        if set(tensors) != {"A", "B"}:
            raise ValueError(f"incomplete lora pair at {loc}: got {sorted(tensors)}")
    return pairs


def main() -> None:
    with open(LORA_DIR / "adapter_config.json") as f:
        adapter_config = json.load(f)

    if adapter_config.get("fan_in_fan_out", False):
        raise NotImplementedError(
            "this script only handles fan_in_fan_out=false (nn.Linear [out,in] layout)"
        )

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scale = lora_alpha / r

    state_dict = load_base_state_dict()
    n_base_tensors_before = len(state_dict)

    pairs = load_adapter_pairs()

    # --- Required check: exactly 32 adapter pairs found -------------------
    n_pairs = len(pairs)
    if n_pairs != 32:
        raise AssertionError(f"expected exactly 32 adapter pairs, found {n_pairs}")

    merged_targets: set[str] = set()
    for (layer, module), tensors in pairs.items():
        A = tensors["A"].to(torch.float32)  # [r, in]
        B = tensors["B"].to(torch.float32)  # [out, r]
        target_name = f"model.layers.{layer}.{module}.weight"
        if target_name not in state_dict:
            raise KeyError(f"adapter targets missing base tensor {target_name!r}")
        base_tensor = state_dict[target_name]
        if base_tensor.dtype != torch.float32:
            raise AssertionError(
                f"{target_name}: expected float32 base tensor, got {base_tensor.dtype}"
            )
        delta = scale * (B @ A)  # [out, in], no transpose (fan_in_fan_out=False)
        if delta.shape != base_tensor.shape:
            raise AssertionError(
                f"{target_name}: delta shape {tuple(delta.shape)} != "
                f"base shape {tuple(base_tensor.shape)}"
            )
        state_dict[target_name] = (base_tensor + delta).contiguous()
        merged_targets.add(target_name)

    if len(merged_targets) != 32:
        raise AssertionError(
            f"expected 32 distinct merged base tensors, got {len(merged_targets)}"
        )

    # --- Required checks ----------------------------------------------------
    if any("lora_" in name for name in state_dict):
        raise AssertionError("a tensor named with 'lora_' leaked into the output")

    q0 = "model.layers.0.self_attn.q_proj.weight"
    if tuple(state_dict[q0].shape) != (2048, 2048):
        raise AssertionError(
            f"{q0}: expected shape (2048, 2048), got {tuple(state_dict[q0].shape)}"
        )

    if len(state_dict) != 114:
        raise AssertionError(f"expected exactly 114 tensors in output, got {len(state_dict)}")

    if len(state_dict) != n_base_tensors_before:
        raise AssertionError(
            "tensor count changed during merge "
            f"({n_base_tensors_before} -> {len(state_dict)}); merge must be in place"
        )

    # --- Shard packing --------------------------------------------------
    # Greedy bin-packing in base weight_map order: a tensor whose own size
    # exceeds the budget gets its own shard; otherwise tensors are appended
    # to the current shard until adding the next one would exceed the budget.
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        base_order = list(json.load(f)["weight_map"].keys())
    assert set(base_order) == set(state_dict.keys())

    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in base_order:
        size = nbytes(state_dict[name])
        if size > SHARD_BUDGET:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_BUDGET:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    for shard_names in shards:
        total = sum(nbytes(state_dict[n]) for n in shard_names)
        if len(shard_names) > 1 and total > SHARD_BUDGET:
            raise AssertionError("shard packer produced an oversized multi-tensor shard")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_file = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        tensors = {name: state_dict[name] for name in shard_names}
        save_file(tensors, OUT_DIR / shard_file, metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_file
            total_size += nbytes(state_dict[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"OK: merged {n_pairs} lora pairs, wrote {len(weight_map)} tensors in {n_shards} shards")
    print(f"    scale = lora_alpha/r = {lora_alpha}/{r} = {scale}")
    print(f"    output: {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fail loudly, don't swallow
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
