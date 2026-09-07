"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, renumbering contiguously."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT_DIR = "out/T1"
OUT = os.path.join(OUT_DIR, "model.safetensors")
REMOVE = {2, 6, 10, 14}
OLD_LAYERS = 16
KEEP = [i for i in range(OLD_LAYERS) if i not in REMOVE]
NEW_INDEX = {old: new for new, old in enumerate(KEEP)}  # old -> new
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
EXPECTED_TENSORS = 86
EXPECTED_LAYERS = 12


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all shards into one state dict.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
sd = {}
for shard in shards:
    part = load_file(os.path.join(BASE, shard))
    if set(part) & set(sd):
        fail(f"duplicate keys across shards: {shard}")
    sd.update(part)
if len(sd) != 114:
    fail(f"expected 114 input tensors, got {len(sd)}")

# Build the new state dict; renumber into a fresh dict so no collision is possible.
new_sd = {}
for name, tensor in sd.items():
    m = LAYER_RE.match(name)
    if m is None:
        new_sd[name] = tensor
        continue
    old = int(m.group(1))
    if old in REMOVE:
        continue
    if old not in NEW_INDEX:
        fail(f"unexpected layer index {old} in {name}")
    new_name = f"model.layers.{NEW_INDEX[old]}.{m.group(2)}"
    if new_name in new_sd:
        fail(f"collision on {new_name}")
    new_sd[new_name] = tensor

# Required checks.
layer_ids = set()
for name in new_sd:
    m = LAYER_RE.match(name)
    if m:
        layer_ids.add(int(m.group(1)))
bad = layer_ids & set(range(EXPECTED_LAYERS, OLD_LAYERS))
if bad:
    fail(f"tensors of removed-range blocks remain: {sorted(bad)}")
q_count = sum(1 for n in new_sd if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", n))
if q_count != EXPECTED_LAYERS or layer_ids != set(range(EXPECTED_LAYERS)):
    fail(f"expected exactly {EXPECTED_LAYERS} blocks, got q_proj count {q_count}, ids {sorted(layer_ids)}")
if len(new_sd) != EXPECTED_TENSORS:
    fail(f"expected {EXPECTED_TENSORS} tensors, got {len(new_sd)}")
# Values/shapes/dtypes preserved for each surviving block.
for old, new in NEW_INDEX.items():
    for rest in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
        a = sd[f"model.layers.{old}.{rest}"]
        b = new_sd[f"model.layers.{new}.{rest}"]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            fail(f"mismatch old {old} -> new {new} for {rest}")

os.makedirs(OUT_DIR, exist_ok=True)
save_file({k: v.contiguous() for k, v in new_sd.items()}, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(new_sd)} tensors, {len(layer_ids)} blocks")
