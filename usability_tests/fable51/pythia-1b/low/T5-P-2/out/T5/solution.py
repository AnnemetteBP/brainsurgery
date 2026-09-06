"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights, write sharded safetensors."""
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
SHARD_BUDGET = 512 * 1024 * 1024  # bytes of tensor data per shard
# TASK.md says the 206 MB embedding matrices get their own shard; isolate tensors above this.
ALONE_THRESHOLD = 128 * 1024 * 1024
PREFIX = "base_model.model."


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


cfg = json.load(open(CONFIG))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)

# Load base into memory (CPU tensors).
state = {}
with safe_open(BASE, framework="pt") as f:
    for k in f.keys():
        state[k] = f.get_tensor(k)
n_base = len(state)

# Pair adapter factors.
pairs = {}
with safe_open(ADAPTER, framework="pt") as f:
    for k in f.keys():
        m = re.fullmatch(r"(.+)\.lora_([AB])\.weight", k)
        if m is None:
            fail(f"unrecognised adapter tensor {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)

merged = 0
for mod, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        fail(f"incomplete pair for {mod}: {sorted(ab)}")
    if not mod.startswith(PREFIX):
        fail(f"adapter module {mod} lacks prefix {PREFIX}")
    target = mod[len(PREFIX):] + ".weight"
    if target not in state:
        fail(f"adapter target {target} not in base")
    A, B = ab["A"].float(), ab["B"].float()
    delta = scale * (B @ A)
    if fan_in_fan_out:
        delta = delta.T
    w = state[target]
    if delta.shape != w.shape:
        fail(f"delta shape {tuple(delta.shape)} != base {tuple(w.shape)} for {target}")
    state[target] = (w.float() + delta).to(w.dtype).contiguous()
    merged += 1

# Required checks.
if merged != 16:
    fail(f"expected 16 merged adapter pairs, got {merged}")
if any("lora_" in k for k in state):
    fail("lora_ tensor present in output")
probe = "gpt_neox.layers.0.attention.query_key_value.weight"
if tuple(state[probe].shape) != (6144, 2048):
    fail(f"{probe} has shape {tuple(state[probe].shape)}")
if len(state) != 244 or len(state) != n_base:
    fail(f"expected 244 output tensors, got {len(state)}")

# Shard: greedy fill in base key order; a tensor larger than budget goes alone.
shards, cur, cur_size = [], {}, 0
for k, t in state.items():
    size = t.numel() * t.element_size()
    if size > ALONE_THRESHOLD:
        if cur:
            shards.append(cur)
            cur, cur_size = {}, 0
        shards.append({k: t})
        continue
    if cur and cur_size + size > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += size
if cur:
    shards.append(cur)

os.makedirs(OUT_DIR, exist_ok=True)
n = len(shards)
weight_map, total = {}, 0
for i, shard in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file(shard, os.path.join(OUT_DIR, name), metadata={"format": "pt"})
    for k, t in shard.items():
        weight_map[k] = name
        total += t.numel() * t.element_size()
index = {"metadata": {"total_size": total}, "weight_map": weight_map}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
    json.dump(index, fh, indent=2, sort_keys=True)
print(f"merged {merged} pairs, wrote {len(weight_map)} tensors in {n} shards to {OUT_DIR}")
