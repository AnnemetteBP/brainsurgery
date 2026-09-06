"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded
safetensors checkpoint. Works directly on the checkpoint files (no model
instantiation). Uses only torch + safetensors + json."""
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
MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114
PREFIX = "base_model.model."
# TASK.md states these are stored alone in their own shard even though each
# (412 MB) fits under the 512 MiB budget; a tensor larger than the budget is
# stored alone regardless.
STANDALONE = {"model.embed_tokens.weight", "lm_head.weight"}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- adapter cfg
cfg = json.load(open(os.path.join(LORA_DIR, "adapter_config.json")))
r, alpha = cfg["r"], cfg["lora_alpha"]
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
scale = alpha / r
print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

# ------------------------------------------------------------- adapter pairs
lora = {}
with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)
pat = re.compile(r"^(.*)\.lora_(A|B)\.weight$")
pairs = {}  # base tensor name -> {"A": tensor, "B": tensor}
for k in lora:
    m = pat.match(k)
    if not m or not k.startswith(PREFIX):
        fail(f"unexpected adapter tensor name: {k}")
    base_name = m.group(1)[len(PREFIX):] + ".weight"
    pairs.setdefault(base_name, {})[m.group(2)] = lora[k]
for name, ab in pairs.items():
    if set(ab) != {"A", "B"}:
        fail(f"incomplete LoRA pair for {name}: {sorted(ab)}")
if len(pairs) != EXPECTED_PAIRS:
    fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")
print(f"found {len(pairs)} adapter pairs")

# ------------------------------------------------------------- base listing
index = json.load(open(os.path.join(BASE_DIR, "model.safetensors.index.json")))
base_map = index["weight_map"]
shard_files = sorted(set(base_map.values()))
# Keep the base's own order: shard order, then key order inside each shard.
order = []  # (name, shard_file, nbytes)
handles = {}
for sf in shard_files:
    h = safe_open(os.path.join(BASE_DIR, sf), "pt")
    handles[sf] = h
    for k in h.keys():
        if base_map[k] != sf:
            fail(f"index/shard disagree for {k}")
        sl = h.get_slice(k)
        if sl.get_dtype() != "F32":
            fail(f"{k}: expected F32, got {sl.get_dtype()}")
        n = 4
        for d in sl.get_shape():
            n *= d
        order.append((k, sf, n))
names = [o[0] for o in order]
if len(names) != len(set(names)):
    fail("duplicate tensor names across base shards")
if len(names) != EXPECTED_TENSORS:
    fail(f"base has {len(names)} tensors, expected {EXPECTED_TENSORS}")
missing = [n for n in pairs if n not in base_map]
if missing:
    fail(f"adapter targets not in base: {missing}")

# ------------------------------------------------------------- shard packing
groups = [[]]
sizes = [0]
for name, sf, n in order:
    if n > MAX_SHARD_BYTES or name in STANDALONE:
        if groups[-1]:
            groups.append([])
            sizes.append(0)
        groups[-1].append(name)
        sizes[-1] = n
        groups.append([])
        sizes.append(0)
        continue
    if sizes[-1] + n > MAX_SHARD_BYTES:
        groups.append([])
        sizes.append(0)
    groups[-1].append(name)
    sizes[-1] += n
if not groups[-1]:
    groups.pop()
    sizes.pop()
n_shards = len(groups)
shard_name = lambda i: f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
for i, s in enumerate(sizes):
    if s > MAX_SHARD_BYTES:
        fail(f"shard {i} would hold {s} bytes > {MAX_SHARD_BYTES}")
print(f"packing into {n_shards} shards: {sizes}")

# ------------------------------------------------------------- pre-write checks
os.makedirs(OUT_DIR, exist_ok=True)
for i in range(n_shards):
    p = os.path.join(OUT_DIR, shard_name(i))
    if os.path.exists(p):
        fail(f"destination exists: {p}")
if any("lora_" in n for n in names):
    fail("output would contain a lora_ tensor")

# ------------------------------------------------------------- merge + write
merged = 0
weight_map = {}
total_size = 0
for i, group in enumerate(groups):
    tensors = {}
    for name in group:
        t = handles[base_map[name]].get_tensor(name)
        if name in pairs:
            A, B = pairs[name]["A"], pairs[name]["B"]
            if A.dtype != torch.float32 or B.dtype != torch.float32:
                fail(f"{name}: adapter factors are not float32")
            delta = (B @ A).to(torch.float32)
            if fan_in_fan_out:
                delta = delta.T
            if delta.shape != t.shape:
                fail(f"{name}: delta shape {tuple(delta.shape)} != {tuple(t.shape)}")
            t = t + scale * delta
            merged += 1
        if t.dtype != torch.float32:
            fail(f"{name}: dtype {t.dtype}")
        tensors[name] = t.contiguous()
        weight_map[name] = shard_name(i)
        total_size += t.numel() * t.element_size()
    if "model.layers.0.self_attn.q_proj.weight" in tensors:
        shp = tuple(tensors["model.layers.0.self_attn.q_proj.weight"].shape)
        if shp != (2048, 2048):
            fail(f"layer0 q_proj shape {shp}")
    if any("lora_" in n for n in tensors):
        fail("lora_ tensor about to be written")
    save_file(tensors, os.path.join(OUT_DIR, shard_name(i)), metadata={"format": "pt"})
    print(f"wrote {shard_name(i)}: {len(tensors)} tensors, {sizes[i]} bytes")
    del tensors

if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
if len(weight_map) != EXPECTED_TENSORS:
    fail(f"output has {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

# ------------------------------------------------------------- post-write verify
seen = {}
for i in range(n_shards):
    with safe_open(os.path.join(OUT_DIR, shard_name(i)), "pt") as h:
        nbytes = 0
        for k in h.keys():
            sl = h.get_slice(k)
            n = 4
            for d in sl.get_shape():
                n *= d
            nbytes += n
            seen[k] = shard_name(i)
        if nbytes > MAX_SHARD_BYTES:
            fail(f"written shard {shard_name(i)} holds {nbytes} bytes")
if seen != weight_map:
    fail("written shards disagree with index weight_map")
if len(seen) != EXPECTED_TENSORS or set(seen) != set(names):
    fail("output key set differs from base")
if any("lora_" in k for k in seen):
    fail("lora_ tensor found in output")
with safe_open(os.path.join(OUT_DIR, seen["model.layers.0.self_attn.q_proj.weight"]), "pt") as h:
    q0 = h.get_tensor("model.layers.0.self_attn.q_proj.weight")
if tuple(q0.shape) != (2048, 2048) or q0.dtype != torch.float32:
    fail(f"layer0 q_proj {tuple(q0.shape)} {q0.dtype}")
b0 = handles[base_map["model.layers.0.self_attn.q_proj.weight"]].get_tensor(
    "model.layers.0.self_attn.q_proj.weight")
p0 = pairs["model.layers.0.self_attn.q_proj.weight"]
ref = b0.double() + scale * (p0["B"].double() @ p0["A"].double())
rel = (q0.double() - ref).norm() / ref.norm()
if rel > 1e-6:
    fail(f"layer0 q_proj relative error {rel}")
print(f"OK: {merged} pairs merged, {len(seen)} tensors in {n_shards} shards, "
      f"layer0 q_proj rel err {rel:.2e}")
