"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks with contiguous renumbering."""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 5, 8}
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def die(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


src = load_file(str(SRC))

kept = [i for i in range(12) if i not in DROP]
remap = {old: new for new, old in enumerate(kept)}

out: dict = {}
for name, tensor in src.items():
    m = BLOCK_RE.match(name)
    if m is None:
        out[name] = tensor
        continue
    old = int(m.group(1))
    if old in DROP:
        continue
    new_name = f"h.{remap[old]}.{m.group(2)}"
    if new_name in out:
        die(f"renumbering collision on {new_name}")
    out[new_name] = tensor

# Required checks.
indices = {int(m.group(1)) for m in (BLOCK_RE.match(k) for k in out) if m}
stale = sorted(i for i in indices if i >= 9)
if stale:
    die(f"tensors of blocks {stale} remain")

n_blocks = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
if n_blocks != 9:
    die(f"expected 9 blocks, found {n_blocks}")

if len(out) != 121:
    die(f"expected 121 tensors, found {len(out)}")

save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
print(f"wrote {DST} with {len(out)} tensors, {n_blocks} blocks")
