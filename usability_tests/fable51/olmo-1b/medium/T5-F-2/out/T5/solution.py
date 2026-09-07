"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded
safetensors checkpoint under out/T5/.

Plain torch + safetensors: no model instantiation, full control over dtype,
bit-exactness of untouched tensors, and the 512 MiB shard budget.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE = os.path.join(SANDBOX, "inputs", "base")
LORA = os.path.join(SANDBOX, "inputs", "lora")
OUT = ROOT  # out/T5
SHARD_BUDGET = 512 * 1024 * 1024  # bytes of tensor data per shard
# Tensors over half the budget (embed_tokens, lm_head at 412 MB) go alone in a shard,
# as the task describes; anything else is packed greedily in base index order.
ALONE_THRESHOLD = SHARD_BUDGET // 2
ADAPTER_PREFIX = "base_model.model."


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


# ---- adapter config ---------------------------------------------------------
cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
r, alpha = cfg["r"], cfg["lora_alpha"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
scale = alpha / r
if r != 16 or alpha != 32 or scale != 2.0:
    fail(f"unexpected adapter config r={r} alpha={alpha}")
if fan_in_fan_out:
    fail("fan_in_fan_out=true not expected for nn.Linear base weights")

# ---- adapter pairs ----------------------------------------------------------
pairs = {}  # base tensor name -> (A, B)
with safe_open(os.path.join(LORA, "adapter_model.safetensors"), "pt") as sf:
    keys = list(sf.keys())
    for k in keys:
        if not k.startswith(ADAPTER_PREFIX):
            fail(f"adapter key without expected prefix: {k}")
        if k.endswith(".lora_A.weight"):
            base_name = k[len(ADAPTER_PREFIX):-len(".lora_A.weight")] + ".weight"
            b_key = k[:-len(".lora_A.weight")] + ".lora_B.weight"
            if b_key not in keys:
                fail(f"missing lora_B for {k}")
            A = sf.get_tensor(k)
            B = sf.get_tensor(b_key)
            pairs[base_name] = (A, B)
        elif k.endswith(".lora_B.weight"):
            continue
        else:
            fail(f"unrecognized adapter key: {k}")
if len(pairs) != 32:
    fail(f"expected 32 adapter pairs, found {len(pairs)}")
for name in pairs:
    if not any(name.endswith(f".{m}.weight") for m in cfg["target_modules"]):
        fail(f"adapter targets a non-target module: {name}")

# ---- load base in index order, merge ---------------------------------------
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
weight_map = index["weight_map"]
if len(weight_map) != 114:
    fail(f"base index has {len(weight_map)} tensors, expected 114")

tensors = {}  # name -> tensor, in base index order
merged = 0
for shard in sorted(set(weight_map.values())):
    with safe_open(os.path.join(BASE, shard), "pt") as sf:
        for name in sf.keys():
            if weight_map[name] != shard:
                fail(f"index/shard mismatch for {name}")
            t = sf.get_tensor(name)
            if name in pairs:
                A, B = pairs[name]
                if t.dtype != torch.float32 or A.dtype != torch.float32 or B.dtype != torch.float32:
                    fail(f"non-float32 tensor in merge for {name}")
                delta = scale * (B @ A)  # [out, r] @ [r, in] -> [out, in]
                if delta.shape != t.shape:
                    fail(f"delta shape {tuple(delta.shape)} != base {tuple(t.shape)} for {name}")
                t = (t + delta).contiguous()
                merged += 1
            tensors[name] = t

if merged != 32:
    fail(f"merged {merged} weights, expected 32")
missing = set(pairs) - set(tensors)
if missing:
    fail(f"adapter targets not in base: {sorted(missing)[:3]}")

# ---- required checks --------------------------------------------------------
if any("lora_" in n for n in tensors):
    fail("lora_ tensor present in output")
q0 = tensors["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048):
    fail(f"q_proj layer 0 shape {tuple(q0.shape)}")
if q0.dtype != torch.float32:
    fail(f"q_proj layer 0 dtype {q0.dtype}")
if len(tensors) != 114:
    fail(f"output has {len(tensors)} tensors, expected 114")
if set(tensors) != set(weight_map):
    fail("output key set differs from base")

# ---- shard greedily (base index order), one oversized tensor per shard ----
shards = []  # list of (names, size)
cur, cur_size = [], 0
for name, t in tensors.items():
    nbytes = t.numel() * t.element_size()
    if nbytes > ALONE_THRESHOLD:
        if cur:
            shards.append((cur, cur_size))
            cur, cur_size = [], 0
        shards.append(([name], nbytes))
        continue
    if cur_size + nbytes > SHARD_BUDGET:
        shards.append((cur, cur_size))
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += nbytes
if cur:
    shards.append((cur, cur_size))

n = len(shards)
out_map = {}
total = 0
for i, (names, size) in enumerate(shards, 1):
    if size > SHARD_BUDGET and len(names) != 1:
        fail(f"shard {i} exceeds budget with multiple tensors")
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: tensors[k] for k in names}, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in names:
        out_map[k] = fname
    total += size
    print(f"wrote {fname}: {len(names)} tensors, {size} bytes")

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": out_map}, f, indent=2)

# ---- post-write verification -----------------------------------------------
seen = 0
for fname in sorted(set(out_map.values())):
    with safe_open(os.path.join(OUT, fname), "pt") as sf:
        for k in sf.keys():
            if out_map[k] != fname:
                fail(f"written index mismatch for {k}")
            seen += 1
if seen != 114 or len(out_map) != 114:
    fail(f"written checkpoint has {seen} tensors")
print(f"OK: merged {merged} weights, {seen} tensors in {n} shards, total {total} bytes")
