"""
T5: Fold a LoRA adapter into Pythia-1B's base weights ("merge and unload"),
then write the result as a sharded safetensors checkpoint.

Standalone script: torch + safetensors only.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T5/solution.py -> sandbox)
BASE_PATH = ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = ROOT / "out" / "T5"

SHARD_BUDGET_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard

# Adapter tensor names look like:
#   base_model.model.gpt_neox.layers.<i>.attention.query_key_value.lora_A.weight
#   base_model.model.gpt_neox.layers.<i>.attention.query_key_value.lora_B.weight
# The corresponding base tensor is:
#   gpt_neox.layers.<i>.attention.query_key_value.weight
LORA_A_RE = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.(.+)\.lora_B\.weight$")


def load_adapter_config() -> dict:
    with open(ADAPTER_CONFIG_PATH) as f:
        return json.load(f)


def load_all_tensors(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    cfg = load_adapter_config()
    r = cfg["r"]
    lora_alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg["fan_in_fan_out"]
    scale = lora_alpha / r

    if fan_in_fan_out:
        raise NotImplementedError(
            "fan_in_fan_out=True is not handled by this script; "
            "the task's adapter has fan_in_fan_out=False."
        )

    base_tensors = load_all_tensors(BASE_PATH)
    lora_tensors = load_all_tensors(LORA_PATH)

    # Pair up lora_A / lora_B tensors by their base module name.
    a_by_base: dict[str, torch.Tensor] = {}
    b_by_base: dict[str, torch.Tensor] = {}
    for name, tensor in lora_tensors.items():
        m = LORA_A_RE.match(name)
        if m:
            a_by_base[m.group(1) + ".weight"] = tensor
            continue
        m = LORA_B_RE.match(name)
        if m:
            b_by_base[m.group(1) + ".weight"] = tensor
            continue
        raise ValueError(f"Unexpected tensor in adapter checkpoint: {name!r}")

    if set(a_by_base.keys()) != set(b_by_base.keys()):
        raise AssertionError(
            "lora_A and lora_B module sets do not match: "
            f"A-only={set(a_by_base) - set(b_by_base)}, "
            f"B-only={set(b_by_base) - set(a_by_base)}"
        )

    pairs = sorted(a_by_base.keys())

    # Required check: exactly 16 adapter pairs.
    if len(pairs) != 16:
        raise AssertionError(f"Expected exactly 16 adapter pairs, found {len(pairs)}")

    merged_count = 0
    for base_name in pairs:
        if base_name not in base_tensors:
            raise KeyError(f"Base checkpoint has no tensor named {base_name!r}")

        A = a_by_base[base_name]  # [r, in]
        B = b_by_base[base_name]  # [out, r]
        base_w = base_tensors[base_name]  # [out, in], float16

        if A.shape[0] != r or B.shape[1] != r:
            raise AssertionError(
                f"{base_name}: lora_A/lora_B rank mismatch with r={r} "
                f"(A={tuple(A.shape)}, B={tuple(B.shape)})"
            )

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))  # [out, in], fp32
        if delta.shape != base_w.shape:
            raise AssertionError(
                f"{base_name}: delta shape {tuple(delta.shape)} != "
                f"base shape {tuple(base_w.shape)}"
            )

        merged = (base_w.to(torch.float32) + delta).to(base_w.dtype)
        base_tensors[base_name] = merged
        merged_count += 1

    if merged_count != 16:
        raise AssertionError(f"Expected to merge 16 tensors, merged {merged_count}")

    # Required checks on the merged, pre-write state.
    lora_leftover = [n for n in base_tensors if "lora_" in n]
    if lora_leftover:
        raise AssertionError(f"Adapter tensor names leaked into output: {lora_leftover}")

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base_tensors[qkv0].shape) != (6144, 2048):
        raise AssertionError(
            f"{qkv0} has shape {tuple(base_tensors[qkv0].shape)}, expected (6144, 2048)"
        )

    if len(base_tensors) != 244:
        raise AssertionError(f"Expected 244 tensors in output, found {len(base_tensors)}")

    # --- Shard and write ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Deterministic ordering for shard assignment.
    names = sorted(base_tensors.keys())

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        nbytes = tensor_nbytes(base_tensors[name])
        if nbytes > SHARD_BUDGET_BYTES:
            # Oversized tensor goes alone in its own shard.
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > SHARD_BUDGET_BYTES:
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

    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {}
        for name in shard_names:
            t = base_tensors[name]
            # safetensors needs contiguous tensors.
            shard_tensors[name] = t.contiguous()
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(t)
        save_file(shard_tensors, str(OUT_DIR / shard_filename))

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Final sanity re-check of what actually got written.
    written_names = set(weight_map.keys())
    if len(written_names) != 244:
        raise AssertionError(f"Output index has {len(written_names)} tensors, expected 244")
    if written_names != set(base_tensors.keys()):
        raise AssertionError("Output index tensor names do not match merged tensor names")

    print(f"Merged {merged_count} LoRA adapter pairs (scale={scale}).")
    print(f"Wrote {len(written_names)} tensors across {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
