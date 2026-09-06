"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and export sharded."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
CONFIG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
SHARD_LIMIT = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard
PREFIX = "base_model.model."


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(CONFIG) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # Load base state dict (fp16, kept as-is for unchanged tensors).
    state = {}
    with safe_open(BASE, framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    n_base = len(state)
    print(f"base tensors: {n_base}")

    # Load adapter and pair A/B factors.
    adapter = {}
    with safe_open(ADAPTER, framework="pt") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)
    pat = re.compile(r"^(.*)\.lora_A\.weight$")
    pairs = {}
    for k in adapter:
        m = pat.match(k)
        if not m:
            continue
        stem = m.group(1)
        b_key = f"{stem}.lora_B.weight"
        if b_key not in adapter:
            fail(f"adapter has {k} but no matching {b_key}")
        pairs[stem] = (adapter[k], adapter[b_key])
    if 2 * len(pairs) != len(adapter):
        fail(f"adapter has {len(adapter)} tensors but {len(pairs)} A/B pairs")

    merged = 0
    for stem, (A, B) in sorted(pairs.items()):
        if not stem.startswith(PREFIX):
            fail(f"unexpected adapter name prefix: {stem}")
        base_key = stem[len(PREFIX):] + ".weight"
        if base_key not in state:
            fail(f"adapter target {base_key} not found in base")
        W = state[base_key]
        delta = scale * (B.float() @ A.float())
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != W.shape:
            fail(f"{base_key}: delta shape {tuple(delta.shape)} != base {tuple(W.shape)}")
        state[base_key] = (W.float() + delta).to(W.dtype).contiguous()
        merged += 1

    # Required checks.
    if merged != 16:
        fail(f"expected 16 merged adapter pairs, got {merged}")
    if any("lora_" in k for k in state):
        fail("lora_ tensor present in output state")
    qkv = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(state[qkv].shape) != (6144, 2048):
        fail(f"{qkv} has shape {tuple(state[qkv].shape)}")
    if state[qkv].dtype != torch.float16:
        fail(f"{qkv} has dtype {state[qkv].dtype}")
    if len(state) != 244:
        fail(f"expected 244 output tensors, got {len(state)}")
    print(f"merged {merged} pairs; {len(state)} tensors; checks passed")

    # Shard: greedy fill in base key order, at most SHARD_LIMIT bytes per shard.
    shards = []
    cur, cur_bytes = {}, 0
    for k in state:
        t = state[k]
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = t
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if size > SHARD_LIMIT:
            fail(f"shard {fname} holds {size} bytes > limit")
        save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        total_size += size
        print(f"wrote {fname}: {len(shard)} tensors, {size} bytes")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    if len(weight_map) != 244:
        fail(f"index maps {len(weight_map)} tensors, expected 244")
    print(f"done: {n} shards, index written")


if __name__ == "__main__":
    main()
