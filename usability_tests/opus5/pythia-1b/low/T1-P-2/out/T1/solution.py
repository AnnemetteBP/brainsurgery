import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "inputs" / "base" / "model.safetensors"
DST = BASE / "out" / "T1" / "model.safetensors"

DROP = {2, 6, 10, 14}
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")

src = load_file(str(SRC))

old_layers = sorted({int(m.group(1)) for k in src if (m := LAYER_RE.match(k))})
keep = [i for i in old_layers if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}

out = {}
for k, v in src.items():
    m = LAYER_RE.match(k)
    if m is None:
        out[k] = v
        continue
    idx = int(m.group(1))
    if idx in DROP:
        continue
    new_k = f"gpt_neox.layers.{remap[idx]}.{m.group(2)}"
    assert new_k not in out, f"collision on {new_k}"
    out[new_k] = v

new_layers = sorted({int(m.group(1)) for k in out if (m := LAYER_RE.match(k))})

errs = []
if any(i >= 12 for i in new_layers):
    errs.append(f"layer indices >= 12 remain: {[i for i in new_layers if i >= 12]}")
if new_layers != list(range(12)):
    errs.append(f"layer indices are not 0..11: {new_layers}")
n_qkv = sum(1 for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k))
if n_qkv != 12:
    errs.append(f"expected 12 query_key_value.weight tensors, found {n_qkv}")
if len(out) != 184:
    errs.append(f"expected 184 tensors, found {len(out)}")
if errs:
    for e in errs:
        print("CHECK FAILED:", e, file=sys.stderr)
    sys.exit(1)

save_file({k: v.contiguous().clone() for k, v in out.items()}, str(DST))
print(f"wrote {DST} with {len(out)} tensors, layers {new_layers[0]}..{new_layers[-1]}")
