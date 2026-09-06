"""T1: depth-prune Pythia-1B 16 -> 12 layers, renumber blocks contiguously."""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD, N_NEW = 16, 12
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")

sd = load_file(SRC)
assert len(sd) == 244, f"expected 244 input tensors, got {len(sd)}"

keep = [i for i in range(N_OLD) if i not in DROP]
assert len(keep) == N_NEW
remap = {old: new for new, old in enumerate(keep)}

out = {}
for name, t in sd.items():
    m = LAYER_RE.match(name)
    if m is None:
        out[name] = t  # non-block tensor, unchanged
        continue
    old = int(m.group(1))
    assert 0 <= old < N_OLD, name
    if old in DROP:
        continue
    new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
    assert new_name not in out, f"collision on {new_name}"
    out[new_name] = t

# Required checks (fail loudly before anything is written).
for name in out:
    m = LAYER_RE.match(name)
    if m is not None and int(m.group(1)) >= N_NEW:
        sys.exit(f"CHECK FAILED: block {m.group(1)} remains: {name}")
qkv = [n for n in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", n)]
if len(qkv) != N_NEW:
    sys.exit(f"CHECK FAILED: expected {N_NEW} blocks, found {len(qkv)}")
for i in range(N_NEW):
    if sum(1 for n in out if n.startswith(f"gpt_neox.layers.{i}.")) != 15:
        sys.exit(f"CHECK FAILED: block {i} does not have 15 tensors")
if len(out) != 184:
    sys.exit(f"CHECK FAILED: expected 184 tensors, got {len(out)}")
non_block = [n for n in out if LAYER_RE.match(n) is None]
if len(non_block) != 4:
    sys.exit(f"CHECK FAILED: expected 4 non-block tensors, got {non_block}")

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(out, DST, metadata={"format": "pt"})
print(f"wrote {DST} with {len(out)} tensors; remap {remap}")
