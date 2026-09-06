"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded
safetensors checkpoint.

Plain torch + safetensors: the merge is a closed-form update on the checkpoint
files, so we never instantiate the model, and we keep exact control over the
shard-packing rule the task specifies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("inputs/base")
LORA = Path("inputs/lora")
OUT = Path("out/T5")

SHARD_LIMIT = 536_870_912  # 512 MiB of tensor data, headers excluded
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114

ADAPTER_KEY = re.compile(r"^base_model\.model\.(model\.layers\.\d+\..*)\.lora_([AB])\.weight$")


def load_base() -> dict[str, torch.Tensor]:
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(BASE / shard, framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    # keep the base index order, which is what the output shards follow
    return {name: tensors[name] for name in weight_map}


def load_adapter() -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    cfg = json.loads((LORA / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("fan_in_fan_out", False):
        raise SystemExit("fan_in_fan_out=True is not handled by this script")
    scale = alpha / r

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(LORA / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            m = ADAPTER_KEY.match(key)
            if m is None:
                raise SystemExit(f"unrecognised adapter key: {key}")
            pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(key)
    return pairs, scale


def merge(base: dict[str, torch.Tensor], pairs, scale: float) -> int:
    merged = 0
    for module, factors in pairs.items():
        if set(factors) != {"A", "B"}:
            raise SystemExit(f"incomplete adapter pair for {module}: {sorted(factors)}")
        target = f"{module}.weight"
        if target not in base:
            raise SystemExit(f"adapter targets a tensor absent from the base: {target}")
        a, b = factors["A"].float(), factors["B"].float()
        w = base[target]
        delta = scale * (b @ a)  # fan_in_fan_out=False: B@A is already [out, in]
        if delta.shape != w.shape:
            raise SystemExit(f"{target}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        base[target] = (w.float() + delta).to(torch.float32).contiguous()
        merged += 1
    return merged


def plan_shards(base: dict[str, torch.Tensor]) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name, t in base.items():
        nbytes = t.numel() * t.element_size()
        if current and used + nbytes > SHARD_LIMIT:
            shards.append(current)
            current, used = [], 0
        current.append(name)
        used += nbytes
    if current:
        shards.append(current)
    return shards


def main() -> None:
    base = load_base()
    pairs, scale = load_adapter()
    merged = merge(base, pairs, scale)

    # Required checks: fail loudly before anything is written.
    if merged != EXPECTED_PAIRS:
        raise SystemExit(f"expected {EXPECTED_PAIRS} merged adapter pairs, found {merged}")
    offenders = [n for n in base if "lora_" in n]
    if offenders:
        raise SystemExit(f"adapter tensors leaked into the output: {offenders[:5]}")
    probe = "model.layers.0.self_attn.q_proj.weight"
    if tuple(base[probe].shape) != (2048, 2048):
        raise SystemExit(f"{probe} has shape {tuple(base[probe].shape)}, expected (2048, 2048)")
    if len(base) != EXPECTED_TENSORS:
        raise SystemExit(f"expected {EXPECTED_TENSORS} tensors, got {len(base)}")

    shards = plan_shards(base)
    total = len(shards)
    OUT.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, names in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{total:05d}.safetensors"
        part = {n: base[n].contiguous() for n in names}
        nbytes = sum(t.numel() * t.element_size() for t in part.values())
        if nbytes > SHARD_LIMIT and len(part) > 1:
            raise SystemExit(f"{filename}: {nbytes} bytes over the shard budget")
        total_size += nbytes
        save_file(part, OUT / filename, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(names, filename))

    (OUT / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
        + "\n"
    )
    print(f"merged {merged} adapter pairs (scale={scale}) into {len(base)} tensors")
    print(f"wrote {total} shards, {total_size} bytes of tensor data, to {OUT}")


if __name__ == "__main__":
    main()
