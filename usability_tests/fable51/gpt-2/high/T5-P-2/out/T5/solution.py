"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
ADAPTER = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
ADAPTER_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data per shard
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    with open(ADAPTER_CFG) as f:
        cfg = json.load(f)
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # Load base in file order (safe_open keys are what the file header lists).
    base: dict[str, torch.Tensor] = {}
    with safe_open(BASE, framework="pt") as f:
        for k in f.keys():
            base[k] = f.get_tensor(k)
    print(f"base tensors: {len(base)}")

    adapter: dict[str, torch.Tensor] = {}
    with safe_open(ADAPTER, framework="pt") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)
    print(f"adapter tensors: {len(adapter)}")

    # Map adapter names to base names: base_model.model.<name>.lora_{A,B}.weight -> <name>.weight
    pat = re.compile(r"^base_model\.model\.(?P<mod>.+)\.lora_(?P<which>[AB])\.weight$")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in adapter.items():
        m = pat.match(k)
        if m is None:
            fail(f"unrecognised adapter tensor name: {k}")
        pairs.setdefault(m.group("mod"), {})[m.group("which")] = v

    merged = 0
    for mod, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            fail(f"incomplete LoRA pair for {mod}: have {sorted(ab)}")
        target = f"{mod}.weight"
        if target not in base:
            fail(f"adapter target {target} not found in base checkpoint")
        A = ab["A"].to(torch.float32)
        B = ab["B"].to(torch.float32)
        W = base[target]
        if W.dtype != torch.float32:
            fail(f"{target} is {W.dtype}, expected float32")
        delta = scale * (B @ A)  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {target}")
        base[target] = (W + delta).to(torch.float32).contiguous()
        merged += 1
        print(f"merged {target}: A{tuple(A.shape)} B{tuple(B.shape)} -> {tuple(base[target].shape)}")

    # Required checks
    if merged != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} adapter pairs merged, got {merged}")
    lora_keys = [k for k in base if "lora_" in k]
    if lora_keys:
        fail(f"adapter tensors present in output: {lora_keys}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(base[probe].shape) != (768, 2304):
        fail(f"{probe} has shape {tuple(base[probe].shape)}, expected (768, 2304)")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
    for k, v in base.items():
        if not v.is_contiguous():
            base[k] = v.contiguous()

    # Shard: greedy in key order, at most MAX_SHARD_BYTES of tensor data per shard;
    # a tensor larger than the budget goes alone in its own shard.
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for k, v in base.items():
        size = v.numel() * v.element_size()
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
            shards.append([k])
            current, current_size = [], 0
            continue
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(k)
        current_size += size
    if current:
        shards.append(current)

    os.makedirs(OUT_DIR, exist_ok=True)
    existing = [
        p for p in os.listdir(OUT_DIR) if p.endswith(".safetensors") or p.endswith(".index.json")
    ]
    if existing:
        fail(f"output directory {OUT_DIR} already contains checkpoint files: {existing}")

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: base[k] for k in names}
        shard_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if len(names) > 1 and shard_bytes > MAX_SHARD_BYTES:
            fail(f"shard {fname} has {shard_bytes} bytes of tensor data > {MAX_SHARD_BYTES}")
        save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
        total_size += shard_bytes
        print(f"wrote {fname}: {len(names)} tensors, {shard_bytes} bytes")

    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote index: {n} shards, {len(weight_map)} tensors, total_size={total_size}")


if __name__ == "__main__":
    main()
