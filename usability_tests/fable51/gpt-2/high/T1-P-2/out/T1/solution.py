"""T1: depth pruning of GPT-2 (124M) from 12 to 9 layers with layer renumbering.

Removes blocks 2, 5, 8 and renumbers the survivors contiguously. Builds a fresh
state dict from an explicit old->new index map (no in-place renames, so no
collision hazard), verifies the result, and only then writes the output file.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base" / "model.safetensors"
DST = ROOT / "out" / "T1" / "model.safetensors"

NUM_LAYERS_IN = 12
REMOVE = {2, 5, 8}
TENSORS_PER_BLOCK = 13
NON_BLOCK = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if DST.exists():
        fail(f"destination already exists: {DST}")
    src = load_file(str(SRC))
    if len(src) != NUM_LAYERS_IN * TENSORS_PER_BLOCK + len(NON_BLOCK):
        fail(f"unexpected input tensor count {len(src)}")

    survivors = [i for i in range(NUM_LAYERS_IN) if i not in REMOVE]
    old_to_new = {old: new for new, old in enumerate(survivors)}
    num_layers_out = len(survivors)

    out: dict = {}
    for name, t in src.items():
        m = BLOCK_RE.match(name)
        if m is None:
            if name not in NON_BLOCK:
                fail(f"unexpected non-block tensor: {name}")
            new_name = name
        else:
            old = int(m.group(1))
            if old >= NUM_LAYERS_IN:
                fail(f"block index out of range in {name}")
            if old in REMOVE:
                continue
            new_name = f"h.{old_to_new[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision on destination name {new_name}")
        out[new_name] = t.contiguous()

    # Required checks.
    stale = [n for n in out if (m := BLOCK_RE.match(n)) and int(m.group(1)) >= num_layers_out]
    if stale:
        fail(f"tensors of removed block indices remain: {stale[:5]}")
    c_attn = sorted(int(BLOCK_RE.match(n).group(1)) for n in out if n.endswith(".attn.c_attn.weight"))
    if c_attn != list(range(num_layers_out)):
        fail(f"expected exactly {num_layers_out} contiguous blocks, got c_attn indices {c_attn}")
    expected_total = num_layers_out * TENSORS_PER_BLOCK + len(NON_BLOCK)
    if len(out) != expected_total or expected_total != 121:
        fail(f"expected 121 output tensors, got {len(out)}")

    # Extra checks: every surviving block is complete and values are the originals.
    for old, new in old_to_new.items():
        old_names = {m.group(2) for n in src if (m := BLOCK_RE.match(n)) and int(m.group(1)) == old}
        new_names = {m.group(2) for n in out if (m := BLOCK_RE.match(n)) and int(m.group(1)) == new}
        if old_names != new_names or len(new_names) != TENSORS_PER_BLOCK:
            fail(f"block {old}->{new} is incomplete or mismatched")
        for rest in old_names:
            a, b = src[f"h.{old}.{rest}"], out[f"h.{new}.{rest}"]
            if a.shape != b.shape or a.dtype != b.dtype or not a.equal(b):
                fail(f"value mismatch for h.{old}.{rest} -> h.{new}.{rest}")
    for name in NON_BLOCK:
        if not src[name].equal(out[name]):
            fail(f"non-block tensor changed: {name}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata={"format": "pt"})
    print(f"wrote {DST} with {len(out)} tensors ({num_layers_out} blocks)")


if __name__ == "__main__":
    main()
