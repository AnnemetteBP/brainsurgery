"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and export sharded."""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
ADAPTER_CFG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB of tensor data per shard
PREFIX = "base_model.model."
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base = load_file(BASE)
    adapter = load_file(ADAPTER)
    with open(ADAPTER_CFG) as f:
        cfg = json.load(f)
    scale = cfg["lora_alpha"] / cfg["r"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"loaded base ({len(base)} tensors), adapter ({len(adapter)} tensors), "
          f"scale={scale}, fan_in_fan_out={fan_in_fan_out}")

    # Map adapter A/B pairs to base tensor names.
    pairs = {}
    for name, t in adapter.items():
        if not name.startswith(PREFIX):
            fail(f"unexpected adapter name (no PEFT prefix): {name}")
        rest = name[len(PREFIX):]
        if rest.endswith(".lora_A.weight"):
            key, which = rest[: -len(".lora_A.weight")], "A"
        elif rest.endswith(".lora_B.weight"):
            key, which = rest[: -len(".lora_B.weight")], "B"
        else:
            fail(f"unexpected adapter tensor: {name}")
        pairs.setdefault(key, {})[which] = t

    merged = 0
    for key, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            fail(f"incomplete LoRA pair for {key}: {sorted(ab)}")
        base_name = f"{key}.weight"
        if base_name not in base:
            fail(f"adapter target {base_name} not in base")
        A, B = ab["A"].float(), ab["B"].float()
        W = base[base_name]
        if W.dtype != torch.float32:
            fail(f"{base_name} dtype {W.dtype}, expected float32")
        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        if fan_in_fan_out:
            delta = delta.T  # Conv1D layout [in, out]
        if delta.shape != W.shape:
            fail(f"{base_name}: delta shape {tuple(delta.shape)} != weight shape {tuple(W.shape)}")
        base[base_name] = (W + delta).contiguous()
        merged += 1
        print(f"merged {base_name} {tuple(W.shape)} r={A.shape[0]}")

    # Required checks.
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
    lora_names = [n for n in base if "lora_" in n]
    if lora_names:
        fail(f"adapter tensors in output: {lora_names}")
    shp = tuple(base["h.0.attn.c_attn.weight"].shape)
    if shp != (768, 2304):
        fail(f"h.0.attn.c_attn.weight shape {shp}, expected (768, 2304)")
    if base["h.0.attn.c_attn.weight"].dtype != torch.float32:
        fail("h.0.attn.c_attn.weight dtype is not float32")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
    print("all checks passed")

    # Shard: greedy in name order, <= 100 MiB of tensor data per shard;
    # an oversized tensor gets its own shard.
    shards = []  # list of lists of names
    cur, cur_bytes = [], 0
    for name in base:  # base preserves file order
        nbytes = base[name].numel() * base[name].element_size()
        if cur and cur_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map = {}
    total_size = 0
    for i, names in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: base[k].contiguous() for k in names}
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        if size > MAX_SHARD_BYTES and len(tensors) != 1:
            fail(f"shard {fname} oversized with {len(tensors)} tensors")
        save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
        total_size += size
        print(f"wrote {fname}: {len(names)} tensors, {size} bytes")

    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"wrote index: {n} shards, {len(weight_map)} tensors, {total_size} bytes")


if __name__ == "__main__":
    main()
