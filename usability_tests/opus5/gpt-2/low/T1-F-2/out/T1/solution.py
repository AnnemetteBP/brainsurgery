"""T1: depth-prune GPT-2 blocks 2, 5, 8 and renumber survivors contiguously."""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

SANDBOX = Path(__file__).resolve().parents[2]
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = SANDBOX / "out" / "T1" / "model.safetensors"

DROP = {2, 5, 8}
BLOCK = re.compile(r"^h\.(\d+)\.(.+)$")


def main() -> None:
    src = load_file(str(SRC))

    survivors = sorted({int(m.group(1)) for k in src if (m := BLOCK.match(k))} - DROP)
    remap = {old: new for new, old in enumerate(survivors)}

    out: dict = {}
    for key, tensor in src.items():
        m = BLOCK.match(key)
        if m is None:
            out[key] = tensor
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        new_key = f"h.{remap[old]}.{m.group(2)}"
        if new_key in out:
            raise SystemExit(f"collision: {key} -> {new_key} already present")
        out[new_key] = tensor

    # Required checks: fail loudly before anything is written.
    indices = {int(m.group(1)) for k in out if (m := BLOCK.match(k))}
    stale = indices & {9, 10, 11}
    if stale:
        raise SystemExit(f"blocks {sorted(stale)} must not remain")
    if indices != set(range(9)):
        raise SystemExit(f"block indices are not 0..8 contiguous: {sorted(indices)}")
    n_blocks = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if n_blocks != 9:
        raise SystemExit(f"expected 9 blocks, found {n_blocks}")
    if len(out) != 121:
        raise SystemExit(f"expected 121 tensors, found {len(out)}")
    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in out or out[name] is not src[name]:
            raise SystemExit(f"non-block tensor {name} was altered")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST))
    print(f"wrote {DST} with {len(out)} tensors", file=sys.stderr)


main()
