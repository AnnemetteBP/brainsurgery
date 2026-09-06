"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded safetensors checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_DIR = os.path.join(ROOT, "inputs", "lora")
OUT = os.path.join(ROOT, "out", "T5")
SHARD_BUDGET = 100 * 1024 * 1024


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


base = load_file(BASE)
lora = load_file(os.path.join(LORA_DIR, "adapter_model.safetensors"))
cfg = json.load(open(os.path.join(LORA_DIR, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)

n_base = len(base)
pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for a_name in list(lora):
    m = pat.match(a_name)
    if not m:
        continue
    b_name = f"base_model.model.{m.group(1)}.lora_B.weight"
    w_name = f"{m.group(1)}.weight"
    if b_name not in lora:
        fail(f"missing {b_name}")
    if w_name not in base:
        fail(f"missing base tensor {w_name}")
    A, B = lora[a_name].float(), lora[b_name].float()
    delta = scale * (B @ A)
    if fan_in_fan_out:
        delta = delta.T
    W = base[w_name]
    if W.shape != delta.shape:
        fail(f"shape mismatch for {w_name}: {tuple(W.shape)} vs {tuple(delta.shape)}")
    base[w_name] = (W.float() + delta).to(W.dtype).contiguous()
    merged += 1

if merged != 12:
    fail(f"expected 12 merged adapter pairs, got {merged}")
if any("lora_" in k for k in base):
    fail("lora_ tensor in output")
if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
    fail(f"bad c_attn shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
if len(base) != 160 or len(base) != n_base:
    fail(f"expected 160 tensors, got {len(base)}")

# Greedy sharding: keep base order; a tensor exceeding the budget gets its own shard.
shards, cur, cur_size = [], {}, 0
for name, t in base.items():
    nbytes = t.numel() * t.element_size()
    if cur and cur_size + nbytes > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t
    cur_size += nbytes
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map = {}
total = 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k, t in shard.items():
        weight_map[k] = fname
        total += t.numel() * t.element_size()
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
print(f"merged {merged} pairs, wrote {len(weight_map)} tensors in {n} shards to {OUT}")
