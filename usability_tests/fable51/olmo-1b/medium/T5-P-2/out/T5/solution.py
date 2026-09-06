"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = "inputs/base"
LORA_DIR = "inputs/lora"
OUT_DIR = "out/T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024
# Tensors at least this large get a shard of their own (task: embed_tokens and lm_head, 412 MB each).
ALONE_BYTES = MAX_SHARD_BYTES // 2
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# --- load base ---
with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    base_index = json.load(f)
base_order = list(base_index["weight_map"].keys())
state = {}
for shard in sorted(set(base_index["weight_map"].values())):
    with safe_open(os.path.join(BASE_DIR, shard), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
if len(state) != EXPECTED_TENSORS:
    fail(f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# --- load adapter ---
with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
adapter = {}
with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)

# --- pair A/B and merge ---
pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for a_name, A in adapter.items():
    m = pat.match(a_name)
    if not m:
        continue
    b_name = f"base_model.model.{m.group(1)}.lora_B.weight"
    base_name = f"{m.group(1)}.weight"
    if b_name not in adapter:
        fail(f"missing lora_B for {a_name}")
    if base_name not in state:
        fail(f"base tensor {base_name} not found for adapter {a_name}")
    B = adapter[b_name]
    W = state[base_name]
    if W.dtype != torch.float32:
        fail(f"{base_name} dtype {W.dtype}, expected float32")
    delta = (B.float() @ A.float()) * scale
    if fan_in_fan_out:
        delta = delta.T
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {base_name}")
    state[base_name] = (W + delta).contiguous()
    merged += 1
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")

# --- required checks ---
if any("lora_" in k for k in state):
    fail("lora_ tensor present in output")
q0 = state["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048):
    fail(f"layer 0 q_proj shape {tuple(q0.shape)}")
if len(state) != EXPECTED_TENSORS:
    fail(f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")
if set(state) != set(base_order):
    fail("output key set differs from base")

# --- shard (greedy, base index order) ---
shards = []  # list of lists of names
cur, cur_bytes = [], 0
for name in base_order:
    nbytes = state[name].numel() * state[name].element_size()
    if nbytes >= ALONE_BYTES:
        if cur:
            shards.append(cur)
            cur, cur_bytes = [], 0
        shards.append([name])
        continue
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
    tensors = {k: state[k].contiguous() for k in names}
    shard_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    if shard_bytes > MAX_SHARD_BYTES and len(tensors) > 1:
        fail(f"shard {fname} exceeds budget with {len(tensors)} tensors")
    total_size += shard_bytes
    save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    print(f"wrote {fname}: {len(names)} tensors, {shard_bytes} bytes")

with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
print(f"OK: merged {merged} pairs, {len(weight_map)} tensors in {n} shards")
