"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights, write sharded safetensors."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
LORA_DIR = os.path.join(SANDBOX, "inputs", "lora")
OUT = ROOT
SHARD_BUDGET = 512 * 1024 * 1024  # bytes of tensor data per shard
ALONE_THRESHOLD = 128 * 1024 * 1024  # tensors at least this large get their own shard (task: embeddings)

def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)

with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
print(f"scale={scale} fan_in_fan_out={fan_in_fan_out}")

# Load base
base = {}
with safe_open(BASE, framework="pt") as f:
    for k in f.keys():
        base[k] = f.get_tensor(k)
print(f"base tensors: {len(base)}")

# Load adapter, pair A/B by module
pairs = {}
pat = re.compile(r"^base_model\.model\.(.*)\.lora_(A|B)\.weight$")
with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        m = pat.match(k)
        if not m:
            fail(f"unexpected adapter key {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)

merged = 0
for module, ab in pairs.items():
    if set(ab) != {"A", "B"}:
        fail(f"incomplete pair for {module}: {set(ab)}")
    name = f"{module}.weight"
    if name not in base:
        fail(f"adapter target {name} not in base")
    w = base[name]
    delta = scale * (ab["B"].float() @ ab["A"].float())
    if fan_in_fan_out:
        delta = delta.T
    if delta.shape != w.shape:
        fail(f"shape mismatch for {name}: {tuple(delta.shape)} vs {tuple(w.shape)}")
    base[name] = (w.float() + delta).to(w.dtype)
    merged += 1

# Required checks
if merged != 16:
    fail(f"expected 16 merged adapter pairs, got {merged}")
if any("lora_" in k for k in base):
    fail("lora_ tensor present in output")
qkv = base["gpt_neox.layers.0.attention.query_key_value.weight"]
if tuple(qkv.shape) != (6144, 2048) or qkv.dtype != torch.float16:
    fail(f"layer0 qkv wrong shape/dtype: {tuple(qkv.shape)} {qkv.dtype}")
if len(base) != 244:
    fail(f"expected 244 output tensors, got {len(base)}")

# Shard: greedy, in base key order; a tensor over budget goes alone.
shards, cur, cur_size = [], {}, 0
for k, t in base.items():
    n = t.numel() * t.element_size()
    if n >= ALONE_THRESHOLD:
        if cur:
            shards.append(cur)
        shards.append({k: t})
        cur, cur_size = {}, 0
        continue
    if cur and cur_size + n > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += n
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
for old in os.listdir(OUT):
    if old.endswith(".safetensors") or old == "model.safetensors.index.json":
        os.remove(os.path.join(OUT, old))
weight_map, total = {}, 0
n = len(shards)
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in shard.values())
    if size > SHARD_BUDGET:
        fail(f"shard {fname} over budget")
    save_file({k: t.contiguous() for k, t in shard.items()}, os.path.join(OUT, fname),
              metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = fname
    total += size
    print(f"{fname}: {len(shard)} tensors, {size} bytes")
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
if len(weight_map) != 244:
    fail(f"index has {len(weight_map)} entries")
print(f"OK: {n} shards, {len(weight_map)} tensors merged={merged}")
