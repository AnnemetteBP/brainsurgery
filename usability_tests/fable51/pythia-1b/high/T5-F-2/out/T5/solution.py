"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write a sharded
safetensors checkpoint. Plain safetensors + torch, no model instantiation.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
ADAPTER_CFG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
SHARD_BUDGET = 536_870_912  # 512 MiB of tensor data per shard
# Tensors that the task requires to sit alone in their own shard (in addition
# to any tensor that exceeds SHARD_BUDGET on its own).
STANDALONE = {"gpt_neox.embed_in.weight", "embed_out.weight"}
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
PROBE = "gpt_neox.layers.0.attention.query_key_value.weight"


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


cfg = json.load(open(ADAPTER_CFG))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
print(f"scale = {scale}, fan_in_fan_out = {fan_in_fan_out}")

# ---- load base (dict, preserves file key order) -------------------------------
tensors = {}
with safe_open(BASE, "pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
if len(tensors) != EXPECTED_TENSORS:
    fail(f"base has {len(tensors)} tensors, expected {EXPECTED_TENSORS}")

# ---- collect adapter pairs -----------------------------------------------------
pat = re.compile(r"^base_model\.model\.(.+)\.lora_(A|B)\.weight$")
pairs = {}
with safe_open(ADAPTER, "pt") as f:
    for k in f.keys():
        m = pat.match(k)
        if not m:
            fail(f"unrecognised adapter tensor name {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)

# ---- merge ---------------------------------------------------------------------
merged = 0
for module, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        fail(f"{module}: incomplete adapter pair {sorted(ab)}")
    name = f"{module}.weight"
    if name not in tensors:
        fail(f"{module}: base tensor {name} not found")
    w = tensors[name]
    delta = scale * (ab["B"].float() @ ab["A"].float())  # [out, in]
    if fan_in_fan_out:
        delta = delta.T
    if delta.shape != w.shape:
        fail(f"{name}: delta shape {tuple(delta.shape)} != weight shape {tuple(w.shape)}")
    tensors[name] = (w.float() + delta).to(w.dtype).contiguous()
    merged += 1
print(f"merged {merged} adapter pairs")

# ---- required checks -----------------------------------------------------------
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
lora_keys = [k for k in tensors if "lora_" in k]
if lora_keys:
    fail(f"adapter tensors in output: {lora_keys}")
if tuple(tensors[PROBE].shape) != (6144, 2048):
    fail(f"{PROBE} has shape {tuple(tensors[PROBE].shape)}")
if tensors[PROBE].dtype != torch.float16:
    fail(f"{PROBE} has dtype {tensors[PROBE].dtype}")
if len(tensors) != EXPECTED_TENSORS:
    fail(f"output has {len(tensors)} tensors, expected {EXPECTED_TENSORS}")

# ---- shard (greedy, base key order; oversized tensors alone) -------------------
shards, cur, cur_size = [], {}, 0
for k, t in tensors.items():
    n = t.numel() * t.element_size()
    if k in STANDALONE or n > SHARD_BUDGET:
        if cur:
            shards.append(cur)
            cur, cur_size = {}, 0
        shards.append({k: t})
        continue
    if cur and cur_size + n > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += n
if cur:
    shards.append(cur)

for s in shards:
    size = sum(t.numel() * t.element_size() for t in s.values())
    if size > SHARD_BUDGET and len(s) != 1:
        fail(f"shard of {size} bytes exceeds budget with {len(s)} tensors")
for k in STANDALONE:
    if not any(list(s) == [k] for s in shards):
        fail(f"{k} is not alone in its own shard")

os.makedirs(OUT_DIR, exist_ok=True)
n_shards = len(shards)
weight_map, total_size = {}, 0
for i, s in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
    save_file(s, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for k, t in s.items():
        weight_map[k] = fname
        total_size += t.numel() * t.element_size()
    print(f"wrote {fname}: {len(s)} tensors, {sum(t.numel()*t.element_size() for t in s.values())} bytes")

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
if len(weight_map) != EXPECTED_TENSORS:
    fail(f"index maps {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")
print(f"done: {n_shards} shards, {len(weight_map)} tensors, {total_size} bytes")
