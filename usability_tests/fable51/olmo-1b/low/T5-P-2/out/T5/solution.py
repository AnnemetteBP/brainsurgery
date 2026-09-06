"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(os.path.dirname(ROOT))
BASE_DIR = os.path.join(SANDBOX, "inputs", "base")
LORA_DIR = os.path.join(SANDBOX, "inputs", "lora")
OUT_DIR = ROOT
MAX_SHARD = 512 * 1024 * 1024
PREFIX = "base_model.model."


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


# ---- load base ----
with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)
state = {}
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(os.path.join(BASE_DIR, shard), "pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
base_names = set(state)
print(f"loaded {len(state)} base tensors")

# ---- load adapter ----
with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
scale = cfg["lora_alpha"] / cfg["r"]
if cfg.get("fan_in_fan_out", False):
    fail("fan_in_fan_out=true not handled")
adapter = {}
with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)

# ---- merge ----
merged = 0
for name, A in adapter.items():
    if not name.endswith(".lora_A.weight"):
        continue
    b_name = name.replace(".lora_A.weight", ".lora_B.weight")
    if b_name not in adapter:
        fail(f"missing lora_B for {name}")
    B = adapter[b_name]
    if not name.startswith(PREFIX):
        fail(f"unexpected adapter name {name}")
    base_name = name[len(PREFIX):].replace(".lora_A.weight", ".weight")
    if base_name not in state:
        fail(f"base tensor {base_name} not found")
    W = state[base_name]
    if W.dtype != torch.float32 or A.dtype != torch.float32 or B.dtype != torch.float32:
        fail(f"non-float32 tensors for {base_name}")
    delta = scale * (B.float() @ A.float())
    if delta.shape != W.shape:
        fail(f"shape mismatch {base_name}: {tuple(delta.shape)} vs {tuple(W.shape)}")
    state[base_name] = (W + delta).contiguous()
    merged += 1

# ---- checks ----
if merged != 32:
    fail(f"expected 32 merged pairs, got {merged}")
if any("lora_" in k for k in state):
    fail("lora_ tensor in output")
if tuple(state["model.layers.0.self_attn.q_proj.weight"].shape) != (2048, 2048):
    fail("q_proj shape changed")
if len(state) != 114 or set(state) != base_names:
    fail(f"expected 114 tensors with base names, got {len(state)}")

# ---- shard ----
shards = []
cur, cur_size = {}, 0
for k in state:  # base shard order preserved via index iteration
    t = state[k]
    n = t.numel() * t.element_size()
    if cur and cur_size + n > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += n
if cur:
    shards.append(cur)
for s in shards:
    if sum(t.numel() * t.element_size() for t in s.values()) > MAX_SHARD and len(s) > 1:
        fail("shard over budget")

# ---- write ----
os.makedirs(OUT_DIR, exist_ok=True)
weight_map = {}
total = len(shards)
for i, s in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    save_file(s, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for k in s:
        weight_map[k] = fname
total_size = sum(t.numel() * t.element_size() for t in state.values())
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
print(f"wrote {total} shards, {len(weight_map)} tensors, merged {merged}")
