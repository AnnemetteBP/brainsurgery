"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write a sharded safetensors checkpoint.

Plain script on top of safetensors + torch: reads base and adapter, computes
W += (alpha/r) * B @ A in float32, casts back to float16, and writes shards of
at most 512 MiB of tensor data with an index file.
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
ADAPTER_CFG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
SHARD_LIMIT = 512 * 1024 * 1024  # bytes of tensor data per shard
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
# The task states these are stored alone in their own shard.
ALONE = {"gpt_neox.embed_in.weight", "embed_out.weight"}
PREFIX = "base_model.model."


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(ADAPTER_CFG) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # Load base into memory (tensor names -> tensors), preserve file order.
    base = {}
    with safe_open(BASE, framework="pt") as f:
        for k in f.keys():
            base[k] = f.get_tensor(k)
    if len(base) != EXPECTED_TENSORS:
        fail(f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # Load adapter and pair A/B by module.
    adapter = {}
    with safe_open(ADAPTER, framework="pt") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)
    pairs = {}
    for k in adapter:
        if not k.startswith(PREFIX):
            fail(f"unexpected adapter key without PEFT prefix: {k}")
        stem = k[len(PREFIX):]
        if stem.endswith(".lora_A.weight"):
            pairs.setdefault(stem[: -len(".lora_A.weight")], {})["A"] = adapter[k]
        elif stem.endswith(".lora_B.weight"):
            pairs.setdefault(stem[: -len(".lora_B.weight")], {})["B"] = adapter[k]
        else:
            fail(f"unrecognised adapter key: {k}")
    for mod, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            fail(f"incomplete adapter pair for {mod}: {sorted(ab)}")
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")

    # Merge.
    merged = 0
    for mod, ab in pairs.items():
        name = f"{mod}.weight"
        if name not in base:
            fail(f"adapter module {mod} has no base weight {name}")
        W = base[name]
        delta = scale * (ab["B"].float() @ ab["A"].float())
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {name}")
        base[name] = (W.float() + delta).to(W.dtype).contiguous()
        merged += 1
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")

    # Required checks before writing.
    if any("lora_" in k for k in base):
        fail("output contains a lora_ tensor")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base[probe].shape) != (6144, 2048):
        fail(f"{probe} has shape {tuple(base[probe].shape)}")
    if base[probe].dtype != torch.float16:
        fail(f"{probe} has dtype {base[probe].dtype}")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # Plan shards: greedy fill; an oversized or task-designated tensor goes alone.
    shards, cur, cur_bytes = [], [], 0
    for k, t in base.items():
        nbytes = t.numel() * t.element_size()
        if nbytes > SHARD_LIMIT or k in ALONE:
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([k])
            continue
        if cur_bytes + nbytes > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    # Write.
    for fn in os.listdir(OUT_DIR):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            os.remove(os.path.join(OUT_DIR, fn))
    n = len(shards)
    weight_map, total = {}, 0
    for i, keys in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: base[k] for k in keys}
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        if size > SHARD_LIMIT:
            fail(f"shard {fname} holds {size} bytes > {SHARD_LIMIT}")
        save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname
        total += size
        print(f"{fname}: {len(keys)} tensors, {size} bytes")
    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"index maps {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes")


if __name__ == "__main__":
    main()
