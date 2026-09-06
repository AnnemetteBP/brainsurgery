#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks with renumbering.

Plain script on torch + safetensors: load the sharded input, drop blocks
2/6/10/14, renumber the survivors in order, verify, then write out/T1/model.safetensors.
"""
import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base")
DST = os.path.join(HERE, "model.safetensors")

DROP = {2, 6, 10, 14}
N_OLD = 16
PER_BLOCK = 7
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


survivors = [i for i in range(N_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(survivors)}

index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
weight_map = index["weight_map"]

out = {}
for key, shard in sorted(weight_map.items()):
    m = LAYER_RE.match(key)
    if m is not None:
        old = int(m.group(1))
        if old in DROP:
            continue
        new_key = LAYER_RE.sub(f"model.layers.{remap[old]}.", key, count=1)
    else:
        new_key = key
    if new_key in out:
        fail(f"collision: {new_key} produced twice")
    with safe_open(os.path.join(SRC, shard), framework="pt") as f:
        out[new_key] = f.get_tensor(key).contiguous()

# --- required checks (before anything is written) ---
for i in (12, 13, 14, 15):
    stale = [k for k in out if k.startswith(f"model.layers.{i}.")]
    if stale:
        fail(f"tensors of block {i} remain: {stale[:3]}")

q = [k for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
if len(q) != 12:
    fail(f"expected 12 blocks, found {len(q)}")

idxs = sorted(int(LAYER_RE.match(k).group(1)) for k in q)
if idxs != list(range(12)):
    fail(f"block indices not contiguous 0..11: {idxs}")

for i in range(12):
    n = len([k for k in out if k.startswith(f"model.layers.{i}.")])
    if n != PER_BLOCK:
        fail(f"block {i} has {n} tensors, expected {PER_BLOCK}")

if len(out) != 86:
    fail(f"expected 86 tensors, found {len(out)}")

for k in ("model.embed_tokens.weight", "lm_head.weight"):
    if k not in out:
        fail(f"missing non-block tensor {k}")

# value/shape/dtype identity against the source, under the old names
for old, new in remap.items():
    for key, shard in weight_map.items():
        if not key.startswith(f"model.layers.{old}."):
            continue
        new_key = LAYER_RE.sub(f"model.layers.{new}.", key, count=1)
        with safe_open(os.path.join(SRC, shard), framework="pt") as f:
            ref = f.get_tensor(key)
        got = out[new_key]
        if got.shape != ref.shape or got.dtype != ref.dtype or not got.equal(ref):
            fail(f"{new_key} does not match source {key}")

save_file(out, DST, metadata={"format": "pt"})
print(f"wrote {DST} with {len(out)} tensors, {len(q)} blocks")
