"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and export sharded safetensors.

Plain torch + safetensors script (see REPORT.md for why).
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base/model.safetensors"
LORA_DIR = "inputs/lora"
OUT = "out/T5"
SHARD_LIMIT = 100 * 1024 * 1024  # 100 MiB of tensor data

def fail(msg):
    raise SystemExit("CHECK FAILED: " + msg)

cfg = json.load(open(os.path.join(LORA_DIR, "adapter_config.json")))
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / r
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

base = load_file(BASE)
lora = load_file(os.path.join(LORA_DIR, "adapter_model.safetensors"))

# pair up lora_A / lora_B by their base target name
pairs = {}
for name in lora:
    if ".lora_A.weight" in name:
        target, side = name.split(".lora_A.weight")[0], "A"
    elif ".lora_B.weight" in name:
        target, side = name.split(".lora_B.weight")[0], "B"
    else:
        fail(f"unexpected adapter tensor {name}")
    target = target.removeprefix("base_model.model.")
    pairs.setdefault(target, {})[side] = lora[name]

merged = 0
for target, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        fail(f"incomplete adapter pair for {target}")
    key = target + ".weight"
    if key not in base:
        fail(f"adapter target {key} not present in base checkpoint")
    A, B = ab["A"].float(), ab["B"].float()
    if A.shape[0] != r or B.shape[1] != r:
        fail(f"{target}: factor rank does not match r={r} (A={tuple(A.shape)}, B={tuple(B.shape)})")
    delta = scale * (B @ A)              # [out, in], nn.Linear convention
    if fan_in_fan_out:
        delta = delta.T                  # base uses Conv1D [in, out]
    w = base[key]
    if w.shape != delta.shape:
        fail(f"{target}: delta {tuple(delta.shape)} does not match base {tuple(w.shape)}")
    base[key] = (w.float() + delta).to(torch.float32)
    merged += 1

# --- required checks -------------------------------------------------------
if merged != 12:
    fail(f"expected 12 adapter pairs merged, got {merged}")
if any("lora_" in k for k in base):
    fail("adapter tensor leaked into the output")
if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
    fail(f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
if base["h.0.attn.c_attn.weight"].dtype != torch.float32:
    fail("h.0.attn.c_attn.weight is not float32")
if len(base) != 160:
    fail(f"expected 160 tensors, got {len(base)}")

# --- shard -----------------------------------------------------------------
def nbytes(t):
    return t.numel() * t.element_size()

shards, cur, cur_size = [], {}, 0
for k, t in base.items():
    s = nbytes(t)
    if cur and cur_size + s > SHARD_LIMIT:
        shards.append(cur); cur, cur_size = {}, 0
    cur[k] = t; cur_size += s
    if cur_size >= SHARD_LIMIT:          # includes the single-oversize-tensor case
        shards.append(cur); cur, cur_size = {}, 0
if cur:
    shards.append(cur)

for sh in shards:
    tot = sum(nbytes(t) for t in sh.values())
    if tot > SHARD_LIMIT and len(sh) > 1:
        fail("a multi-tensor shard exceeds 100 MiB")

if os.path.isdir(OUT):
    for f in os.listdir(OUT):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(OUT, f))
os.makedirs(OUT, exist_ok=True)

n = len(shards)
weight_map, total = {}, 0
for i, sh in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: v.contiguous() for k, v in sh.items()}, os.path.join(OUT, fname),
              metadata={"format": "pt"})
    for k, t in sh.items():
        weight_map[k] = fname
        total += nbytes(t)

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)

# --- post-write verification ----------------------------------------------
seen = {}
for fname in sorted(set(weight_map.values())):
    d = load_file(os.path.join(OUT, fname))
    seen.update(d)
if set(seen) != set(load_file(BASE)):
    fail("output key set differs from the base key set")
if len(seen) != 160:
    fail(f"output has {len(seen)} tensors, expected 160")
if any("lora_" in k for k in seen):
    fail("adapter tensor in written output")

print(f"merged {merged} pairs (scale={scale}, fan_in_fan_out={fan_in_fan_out})")
print(f"wrote {n} shards, {len(seen)} tensors, {total} bytes -> {OUT}")
