#!/usr/bin/env python
"""T5: merge a PEFT-style LoRA adapter into the GPT-2 base checkpoint and
write the result as a sharded safetensors checkpoint.

Approach: plain script directly on the checkpoint files (safetensors + torch),
rather than routing through peft.merge_and_unload(). The base checkpoint here
is a bare state dict (keys like "h.0.attn.c_attn.weight", no "transformer."
prefix), so it does not correspond 1:1 to any single HF model class
(GPT2Model vs GPT2LMHeadModel key prefixes differ); reconstructing a module
just to get PEFT's merge path would add indirection without adding safety,
and would then still need custom code to (a) match the exact 100 MiB
tensor-data shard budget and (b) guarantee no stray adapter/intermediate
tensors survive. Doing the key mapping, scale, transpose and add explicitly
keeps every step visible and checkable.

For each LoRA pair (A, B) with fan_in_fan_out=True (PEFT's signal that the
adapted base weight uses the Conv1D "[in, out]" layout, i.e. transposed
relative to the nn.Linear convention the LoRA factors are computed in):

    base.weight += (lora_alpha / r) * (B @ A).T
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent  # .../<sandbox>/out/T5
REPO_ROOT = HERE.parents[1]  # .../<sandbox>/out/T5 -> out -> sandbox root
BASE_DIR = REPO_ROOT / "inputs" / "base"
LORA_DIR = REPO_ROOT / "inputs" / "lora"
OUT_DIR = REPO_ROOT / "out" / "T5"

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB, tensor data only

LORA_A_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_B\.weight$")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base_path = BASE_DIR / "model.safetensors"
    lora_path = LORA_DIR / "adapter_model.safetensors"
    config_path = LORA_DIR / "adapter_config.json"

    config = json.loads(config_path.read_text())
    r = config["r"]
    lora_alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    target_modules = config["target_modules"]
    if not fan_in_fan_out:
        raise SystemExit(
            "fan_in_fan_out is False in adapter_config.json; this script only "
            "implements the Conv1D (transpose) merge path."
        )
    scale = lora_alpha / r

    base_sd = load_state_dict(base_path)
    lora_sd = load_state_dict(lora_path)

    # --- map adapter pairs to base tensor names -----------------------------
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    unmatched = []
    for key, tensor in lora_sd.items():
        m = LORA_A_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})["A"] = tensor
            continue
        m = LORA_B_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})["B"] = tensor
            continue
        unmatched.append(key)

    if unmatched:
        raise SystemExit(f"adapter tensors that are neither lora_A nor lora_B: {unmatched}")

    for base_name, factors in pairs.items():
        if set(factors) != {"A", "B"}:
            raise SystemExit(f"incomplete LoRA pair for {base_name!r}: found {set(factors)}")
        module_ok = any(base_name.endswith(f".{tm}") or f".{tm}." in base_name for tm in target_modules)
        if not module_ok:
            raise SystemExit(f"LoRA pair {base_name!r} does not match target_modules {target_modules}")

    # --- required check: exactly 12 adapter pairs ---------------------------
    if len(pairs) != 12:
        raise SystemExit(f"expected exactly 12 adapter pairs, found {len(pairs)}: {sorted(pairs)}")

    merged_count = 0
    for base_name, factors in pairs.items():
        weight_name = f"{base_name}.weight"  # adapters target the Conv1D module, not a specific tensor
        if weight_name not in base_sd:
            raise SystemExit(f"adapter targets {weight_name!r}, which is not in the base checkpoint")

        A = factors["A"].to(torch.float32)  # [r, in]
        B = factors["B"].to(torch.float32)  # [out, r]
        if A.shape[0] != r or B.shape[1] != r:
            raise SystemExit(f"{base_name}: expected rank {r}, got A{tuple(A.shape)} B{tuple(B.shape)}")

        base_weight = base_sd[weight_name]
        if base_weight.dtype != torch.float32:
            raise SystemExit(f"{weight_name}: expected float32 base tensor, got {base_weight.dtype}")

        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        delta = delta.T  # -> [in, out], Conv1D convention (fan_in_fan_out=True)

        if delta.shape != base_weight.shape:
            raise SystemExit(
                f"{weight_name}: merged delta shape {tuple(delta.shape)} != "
                f"base shape {tuple(base_weight.shape)}"
            )

        merged = (base_weight + delta).to(torch.float32).contiguous()
        base_sd[weight_name] = merged
        merged_count += 1

    if merged_count != 12:
        raise SystemExit(f"merged {merged_count} tensors, expected 12")

    # --- required checks on the assembled output ----------------------------
    lora_like = [k for k in base_sd if "lora_" in k]
    if lora_like:
        raise SystemExit(f"output still contains adapter-named tensors: {lora_like}")

    probe_name = "h.0.attn.c_attn.weight"
    if tuple(base_sd[probe_name].shape) != (768, 2304):
        raise SystemExit(f"{probe_name}: expected shape (768, 2304), got {tuple(base_sd[probe_name].shape)}")

    if len(base_sd) != 160:
        raise SystemExit(f"expected exactly 160 tensors in the output, got {len(base_sd)}")

    # --- shard: greedy bin-pack in the base file's key order ----------------
    order = list(base_sd.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in order:
        size = base_sd[key].numel() * base_sd[key].element_size()
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
        if current_size > MAX_SHARD_BYTES:
            # a single oversized tensor (e.g. wte.weight) fills its own shard
            shards.append(current)
            current = []
            current_size = 0
    if current:
        shards.append(current)

    n_shards = len(shards)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # clear any stale output from a previous attempt so no old shards linger
    for stale in OUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    index_path = OUT_DIR / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_keys in enumerate(shards, start=1):
        shard_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {k: base_sd[k] for k in shard_keys}
        save_file(shard_tensors, str(OUT_DIR / shard_name), metadata={"format": "pt"})
        for k in shard_keys:
            weight_map[k] = shard_name
            total_size += base_sd[k].numel() * base_sd[k].element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path.write_text(json.dumps(index, indent=2))

    print(f"merged {merged_count} LoRA pairs into {len(base_sd)} tensors, wrote {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
