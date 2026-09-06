"""T1: depth-prune Pythia-1B from 16 to 12 layers, renumbering survivors."""
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base" / "model.safetensors"
DST = ROOT / "out" / "T1" / "model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD, N_NEW = 16, 12

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


sd = load_file(str(SRC))
if len(sd) != 244:
    fail(f"expected 244 input tensors, got {len(sd)}")

survivors = [i for i in range(N_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(survivors)}  # old -> new index

out = {}
for name, t in sd.items():
    m = LAYER_RE.match(name)
    if m is None:
        out[name] = t  # non-block tensor, unchanged
        continue
    old = int(m.group(1))
    if old >= N_OLD:
        fail(f"unexpected layer index in {name}")
    if old in DROP:
        continue
    new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
    if new_name in out:
        fail(f"collision on {new_name}")
    out[new_name] = t

# Required checks
for name in out:
    m = LAYER_RE.match(name)
    if m and int(m.group(1)) >= N_NEW:
        fail(f"block index >= {N_NEW} remains: {name}")
qkv = [n for n in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", n)]
if len(qkv) != N_NEW:
    fail(f"expected {N_NEW} qkv weights, got {len(qkv)}")
blocks = {int(LAYER_RE.match(n).group(1)) for n in out if LAYER_RE.match(n)}
if blocks != set(range(N_NEW)):
    fail(f"block indices not contiguous 0..{N_NEW-1}: {sorted(blocks)}")
per_block = {b: sum(1 for n in out if LAYER_RE.match(n) and int(LAYER_RE.match(n).group(1)) == b) for b in blocks}
if any(c != 15 for c in per_block.values()):
    fail(f"blocks with wrong tensor count: {per_block}")
if len(out) != 184:
    fail(f"expected 184 output tensors, got {len(out)}")
# Sanity: values preserved for each surviving block
for old, new in remap.items():
    a = sd[f"gpt_neox.layers.{old}.attention.dense.weight"]
    b = out[f"gpt_neox.layers.{new}.attention.dense.weight"]
    if a.data_ptr() != b.data_ptr() and not a.equal(b):
        fail(f"value mismatch old {old} -> new {new}")

DST.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
print(f"wrote {DST} with {len(out)} tensors")
