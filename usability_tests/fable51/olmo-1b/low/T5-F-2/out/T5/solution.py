"""T5: merge a PEFT LoRA adapter into OLMo-1B-0724-hf base weights and write a sharded checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "inputs", "base")
LORA = os.path.join(ROOT, "inputs", "lora")
OUT = HERE
MAX_SHARD = 512 * 1024 * 1024
EXPECTED_PAIRS, EXPECTED_TENSORS = 32, 114


def fail(msg):
    print("CHECK FAILED:", msg, file=sys.stderr)
    sys.exit(1)


# ---- load base ----
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
state = {}
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(os.path.join(BASE, shard), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
base_names = set(state)

# ---- load adapter ----
with open(os.path.join(LORA, "adapter_config.json")) as f:
    cfg = json.load(f)
if cfg.get("fan_in_fan_out", False):
    fail("fan_in_fan_out=true not expected for nn.Linear targets")
scale = cfg["lora_alpha"] / cfg["r"]
with safe_open(os.path.join(LORA, "adapter_model.safetensors"), framework="pt") as f:
    adapter = {k: f.get_tensor(k) for k in f.keys()}

pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for a_name, A in adapter.items():
    m = pat.match(a_name)
    if not m:
        continue
    b_name = a_name.replace(".lora_A.", ".lora_B.")
    if b_name not in adapter:
        fail(f"missing lora_B for {a_name}")
    target = m.group(1) + ".weight"
    if target not in state:
        fail(f"adapter target {target} not in base")
    B = adapter[b_name]
    W = state[target]
    if A.dtype != torch.float32 or B.dtype != torch.float32 or W.dtype != torch.float32:
        fail(f"non-float32 tensor for {target}")
    delta = scale * (B @ A)
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != weight shape {tuple(W.shape)} for {target}")
    state[target] = (W + delta).contiguous()
    merged += 1

# ---- required checks ----
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
if 2 * merged != len(adapter):
    fail(f"adapter has {len(adapter)} tensors, expected {2 * merged}")
if any("lora_" in k for k in state):
    fail("lora_ tensor in output")
q0 = state["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048) or q0.dtype != torch.float32:
    fail(f"layer0 q_proj shape/dtype wrong: {tuple(q0.shape)} {q0.dtype}")
if len(state) != EXPECTED_TENSORS or set(state) != base_names:
    fail(f"output has {len(state)} tensors / key set differs from base")

# ---- shard (greedy, order as in base index) ----
order = list(index["weight_map"].keys())
shards, cur, cur_size = [], [], 0
for k in order:
    n = state[k].numel() * state[k].element_size()
    if n > MAX_SHARD // 2:  # large tensors (embed_tokens, lm_head) get their own shard
        if cur:
            shards.append(cur)
        shards.append([k])
        cur, cur_size = [], 0
        continue
    if cur and cur_size + n > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(k)
    cur_size += n
if cur:
    shards.append(cur)
for names in shards:
    if sum(state[k].numel() * state[k].element_size() for k in names) > MAX_SHARD and len(names) > 1:
        fail("shard exceeds budget")

weight_map, total = {}, 0
N = len(shards)
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{N:05d}.safetensors"
    save_file({k: state[k] for k in names}, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
        total += state[k].numel() * state[k].element_size()
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
print(f"merged {merged} pairs, wrote {len(weight_map)} tensors in {N} shards")
