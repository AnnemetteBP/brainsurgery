#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks with renumbering."""

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

BASE = Path("inputs/base")
OUT_DIR = Path("out/T1")
OUT_FILE = OUT_DIR / "model.safetensors"

DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = 12
EXPECTED_TENSORS = 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# --- load all shards ---------------------------------------------------
index = json.loads((BASE / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))

state = {}
for shard in shards:
    for k, v in load_file(BASE / shard).items():
        if k in state:
            fail(f"duplicate key across shards: {k}")
        state[k] = v

if len(state) != 114:
    fail(f"expected 114 input tensors, got {len(state)}")

# --- build the old -> new block index map ------------------------------
survivors = [i for i in range(N_OLD) if i not in DROP]
if len(survivors) != N_NEW:
    fail(f"expected {N_NEW} surviving blocks, got {len(survivors)}")
remap = {old: new for new, old in enumerate(survivors)}

# --- rename ------------------------------------------------------------
new_state = {}
for key, tensor in state.items():
    m = LAYER_RE.match(key)
    if m is None:
        new_key = key  # embed_tokens / lm_head
    else:
        old_idx = int(m.group(1))
        if old_idx in DROP:
            continue
        if old_idx not in remap:
            fail(f"block index {old_idx} out of range in key {key}")
        new_key = f"model.layers.{remap[old_idx]}.{m.group(2)}"
    if new_key in new_state:
        fail(f"destination key collision: {new_key}")
    new_state[new_key] = tensor

# --- required checks ---------------------------------------------------
present = set()
for key in new_state:
    m = LAYER_RE.match(key)
    if m:
        present.add(int(m.group(1)))

stale = sorted(i for i in present if i >= N_NEW)
if stale:
    fail(f"tensors remain for out-of-range blocks {stale}")

if present != set(range(N_NEW)):
    fail(f"block indices are not contiguous 0..{N_NEW - 1}: {sorted(present)}")

n_q = sum(
    1 for k in new_state if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)
)
if n_q != N_NEW:
    fail(f"expected {N_NEW} q_proj tensors, got {n_q}")

if len(new_state) != EXPECTED_TENSORS:
    fail(f"expected {EXPECTED_TENSORS} output tensors, got {len(new_state)}")

for name in ("model.embed_tokens.weight", "lm_head.weight"):
    if name not in new_state:
        fail(f"missing non-block tensor {name}")
    if new_state[name] is not state[name]:
        fail(f"non-block tensor {name} was modified")

# --- write -------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in new_state.items()}, str(OUT_FILE))
print(f"wrote {OUT_FILE} with {len(new_state)} tensors ({N_NEW} blocks)")
