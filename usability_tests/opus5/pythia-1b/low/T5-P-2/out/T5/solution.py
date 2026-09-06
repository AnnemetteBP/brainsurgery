"""T5: merge a PEFT LoRA adapter into Pythia-1B and write a sharded checkpoint."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_DIR = os.path.join(ROOT, "inputs", "lora")
OUT_DIR = os.path.join(ROOT, "out", "T5")
MAX_SHARD = 512 * 1024 * 1024

with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
scale = cfg["lora_alpha"] / cfg["r"]
assert scale == 2, f"unexpected scale {scale}"
assert cfg["fan_in_fan_out"] is False, "fan_in_fan_out=True not handled"

state = {}
with safe_open(BASE, framework="pt") as f:
    for k in f.keys():
        state[k] = f.get_tensor(k)
n_base = len(state)

adapters = {}
with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        adapters[k] = f.get_tensor(k)

# group lora_A/lora_B by the base tensor they adapt
pairs = {}
for name in adapters:
    if ".lora_A.weight" in name:
        side, stem = "A", name.split(".lora_A.weight")[0]
    elif ".lora_B.weight" in name:
        side, stem = "B", name.split(".lora_B.weight")[0]
    else:
        raise SystemExit(f"unexpected adapter tensor: {name}")
    base_name = stem
    for prefix in ("base_model.model.",):
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix) :]
    pairs.setdefault(base_name + ".weight", {})[side] = adapters[name]

merged = 0
for base_name, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        raise SystemExit(f"incomplete adapter pair for {base_name}: {sorted(ab)}")
    if base_name not in state:
        raise SystemExit(f"adapter targets missing base tensor {base_name}")
    A, B = ab["A"].float(), ab["B"].float()
    W = state[base_name]
    delta = scale * (B @ A)
    if delta.shape != W.shape:
        raise SystemExit(f"delta shape {tuple(delta.shape)} != base {tuple(W.shape)}")
    state[base_name] = (W.float() + delta).to(W.dtype)
    merged += 1

# --- required checks -------------------------------------------------------
if merged != 16:
    raise SystemExit(f"expected 16 merged adapter pairs, got {merged}")
leftover = [k for k in state if "lora_" in k]
if leftover:
    raise SystemExit(f"adapter tensors leaked into output: {leftover[:5]}")
probe = "gpt_neox.layers.0.attention.query_key_value.weight"
if tuple(state[probe].shape) != (6144, 2048):
    raise SystemExit(f"{probe} has shape {tuple(state[probe].shape)}")
if len(state) != 244:
    raise SystemExit(f"expected 244 output tensors, got {len(state)} (base had {n_base})")

# --- shard -----------------------------------------------------------------
def nbytes(t):
    return t.numel() * t.element_size()

shards, cur, cur_size = [], {}, 0
for name in state:  # preserve base key order
    size = nbytes(state[name])
    if cur and cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = state[name]
    cur_size += size
if cur:
    shards.append(cur)

os.makedirs(OUT_DIR, exist_ok=True)
for stale in os.listdir(OUT_DIR):
    if stale.endswith(".safetensors") or stale == "model.safetensors.index.json":
        os.remove(os.path.join(OUT_DIR, stale))

n = len(shards)
weight_map, total = {}, 0
for i, shard in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    shard_bytes = sum(nbytes(t) for t in shard.values())
    if shard_bytes > MAX_SHARD and len(shard) > 1:
        raise SystemExit(f"{fname} holds {shard_bytes} bytes over the limit")
    total += shard_bytes
    save_file({k: v.contiguous() for k, v in shard.items()}, os.path.join(OUT_DIR, fname))
    for k in shard:
        weight_map[k] = fname

with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

print(f"merged {merged} adapter pairs; wrote {len(weight_map)} tensors in {n} shards ({total} bytes)")
