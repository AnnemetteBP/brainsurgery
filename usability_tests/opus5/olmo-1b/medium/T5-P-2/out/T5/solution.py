"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights, write sharded."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
LORA_DIR = os.path.join(ROOT, "inputs", "lora")
OUT_DIR = os.path.join(ROOT, "out", "T5")

SHARD_BUDGET = 536_870_912  # 512 MiB of tensor data
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114


def load_base():
    """Return (ordered names, {name: tensor}) from the sharded base checkpoint."""
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as fh:
        index = json.load(fh)
    weight_map = index["weight_map"]
    order = list(weight_map)  # preserve the base checkpoint's key order
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(BASE_DIR, shard), framework="pt") as fh:
            for name in fh.keys():
                tensors[name] = fh.get_tensor(name)
    missing = [n for n in order if n not in tensors]
    if missing:
        raise SystemExit(f"index lists {len(missing)} tensors absent from the shards: {missing[:5]}")
    extra = [n for n in tensors if n not in weight_map]
    if extra:
        raise SystemExit(f"shards hold {len(extra)} tensors absent from the index: {extra[:5]}")
    return order, tensors


def load_adapter():
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    tensors = {}
    with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as fh:
        for name in fh.keys():
            tensors[name] = fh.get_tensor(name)
    return cfg, tensors


def merge(base, adapter, scale, fan_in_fan_out):
    """Add scale * B @ A into each adapted base weight. Returns the pair count."""
    pat = re.compile(r"^base_model\.model\.(model\.layers\.\d+\..*)\.lora_A\.weight$")
    merged = 0
    for a_name in sorted(adapter):
        m = pat.match(a_name)
        if m is None:
            continue
        stem = m.group(1)
        b_name = f"base_model.model.{stem}.lora_B.weight"
        if b_name not in adapter:
            raise SystemExit(f"lora_A without matching lora_B: {a_name}")
        target = f"{stem}.weight"
        if target not in base:
            raise SystemExit(f"adapter targets a tensor absent from the base: {target}")

        a = adapter[a_name].to(torch.float32)
        b = adapter[b_name].to(torch.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise SystemExit(f"bad factor shapes for {stem}: A={tuple(a.shape)} B={tuple(b.shape)}")

        delta = scale * (b @ a)  # [out, in], the nn.Linear layout
        if fan_in_fan_out:
            delta = delta.T

        w = base[target]
        if w.shape != delta.shape:
            raise SystemExit(
                f"delta shape {tuple(delta.shape)} != base shape {tuple(w.shape)} for {target}"
            )
        if w.dtype != torch.float32:
            raise SystemExit(f"{target} is {w.dtype}, expected float32")
        base[target] = (w.to(torch.float32) + delta).contiguous()
        merged += 1

    stray = [n for n in adapter if pat.match(n) is None and not n.endswith(".lora_B.weight")]
    if stray:
        raise SystemExit(f"unrecognized adapter tensors: {stray[:5]}")
    return merged


def plan_shards(order, tensors):
    """Greedy packing in checkpoint order; an oversized tensor gets its own shard."""
    shards, current, current_bytes = [], [], 0
    for name in order:
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if current and current_bytes + nbytes > SHARD_BUDGET:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


def main():
    order, base = load_base()
    cfg, adapter = load_adapter()

    r = int(cfg["r"])
    scale = float(cfg["lora_alpha"]) / r
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"base: {len(base)} tensors | adapter: {len(adapter)} tensors | scale = {scale}")

    merged = merge(base, adapter, scale, fan_in_fan_out)

    # Required checks, before writing anything.
    if merged != EXPECTED_PAIRS:
        raise SystemExit(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
    leaked = [n for n in base if "lora_" in n]
    if leaked:
        raise SystemExit(f"adapter tensors leaked into the output: {leaked[:5]}")
    probe = "model.layers.0.self_attn.q_proj.weight"
    if tuple(base[probe].shape) != (2048, 2048):
        raise SystemExit(f"{probe} has shape {tuple(base[probe].shape)}, expected (2048, 2048)")
    if len(base) != EXPECTED_TENSORS:
        raise SystemExit(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
    if set(base) != set(order):
        raise SystemExit("output key set differs from the base key set")

    shards = plan_shards(order, base)
    n = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map, total_size = {}, 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        payload = {name: base[name].contiguous() for name in names}
        nbytes = sum(t.numel() * t.element_size() for t in payload.values())
        if nbytes > SHARD_BUDGET and len(payload) > 1:
            raise SystemExit(f"shard {fname} holds {nbytes} bytes over the {SHARD_BUDGET} budget")
        save_file(payload, os.path.join(OUT_DIR, fname))
        for name in names:
            weight_map[name] = fname
        total_size += nbytes
        print(f"{fname}: {len(names)} tensors, {nbytes} bytes")

    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fh, indent=2)
        fh.write("\n")

    print(f"wrote {len(weight_map)} tensors across {n} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
