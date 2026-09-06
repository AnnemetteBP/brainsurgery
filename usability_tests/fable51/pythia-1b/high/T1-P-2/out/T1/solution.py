"""T1: depth-prune Pythia-1B from 16 to 12 blocks and renumber contiguously.

Removes blocks 2, 6, 10, 14, renumbers the survivors 0..11 in original order,
keeps the four non-block tensors, and writes out/T1/model.safetensors.
All checks run on the in-memory result before anything is written.
"""

import os
import re
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

REMOVE = {2, 6, 10, 14}
N_OLD = 16
N_NEW = N_OLD - len(REMOVE)  # 12
TENSORS_PER_BLOCK = 15
N_NONBLOCK = 4
EXPECTED_TOTAL = N_NEW * TENSORS_PER_BLOCK + N_NONBLOCK  # 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # Old index -> new index for survivors, in original order.
    survivors = [i for i in range(N_OLD) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(survivors)}
    if sorted(remap.values()) != list(range(N_NEW)):
        fail(f"renumbering is not contiguous: {remap}")

    src: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for k in f.keys():
            src[k] = f.get_tensor(k)
    if len(src) != N_OLD * TENSORS_PER_BLOCK + N_NONBLOCK:
        fail(f"unexpected input tensor count {len(src)}")

    out: dict[str, torch.Tensor] = {}
    removed = 0
    per_old_block: dict[int, int] = {}
    for name, t in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor: copy unchanged.
            if name in out:
                fail(f"duplicate destination {name}")
            out[name] = t
            continue
        old = int(m.group(1))
        rest = m.group(2)
        per_old_block[old] = per_old_block.get(old, 0) + 1
        if old in REMOVE:
            removed += 1
            continue
        if old not in remap:
            fail(f"block index {old} outside 0..{N_OLD - 1}: {name}")
        new_name = f"gpt_neox.layers.{remap[old]}.{rest}"
        if new_name in out:
            fail(f"collision: {name} -> {new_name} already assigned")
        out[new_name] = t

    # --- Required checks (on the in-memory result, before writing) ---
    for old, n in per_old_block.items():
        if n != TENSORS_PER_BLOCK:
            fail(f"input block {old} has {n} tensors, expected {TENSORS_PER_BLOCK}")
    if removed != len(REMOVE) * TENSORS_PER_BLOCK:
        fail(f"removed {removed} tensors, expected {len(REMOVE) * TENSORS_PER_BLOCK}")

    out_block_ids = set()
    for name in out:
        m = LAYER_RE.match(name)
        if m:
            out_block_ids.add(int(m.group(1)))
    stale = sorted(i for i in out_block_ids if i >= N_NEW)
    if stale:
        fail(f"tensors of blocks >= {N_NEW} remain: {stale}")
    if out_block_ids != set(range(N_NEW)):
        fail(f"block indices are not exactly 0..{N_NEW - 1}: {sorted(out_block_ids)}")

    qkv = [n for n in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", n)]
    if len(qkv) != N_NEW:
        fail(f"{len(qkv)} query_key_value.weight tensors, expected {N_NEW}")
    for i in range(N_NEW):
        n_i = sum(1 for n in out if n.startswith(f"gpt_neox.layers.{i}."))
        if n_i != TENSORS_PER_BLOCK:
            fail(f"output block {i} has {n_i} tensors, expected {TENSORS_PER_BLOCK}")

    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    # Value-level verification against the source mapping.
    for old, new in remap.items():
        for name, t in src.items():
            m = LAYER_RE.match(name)
            if m and int(m.group(1)) == old:
                o = out[f"gpt_neox.layers.{new}.{m.group(2)}"]
                if o.shape != t.shape or o.dtype != t.dtype or not torch.equal(o, t):
                    fail(f"block {old}->{new} tensor {m.group(2)} differs")
    for name, t in src.items():
        if LAYER_RE.match(name) is None and not torch.equal(out[name], t):
            fail(f"non-block tensor {name} differs")

    # Write atomically: temp file in the output dir, then rename.
    out = {k: v.contiguous() for k, v in out.items()}
    fd, tmp = tempfile.mkstemp(prefix=".model-", suffix=".safetensors", dir=HERE)
    os.close(fd)
    try:
        save_file(out, tmp, metadata=metadata)
        os.replace(tmp, DST)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    # Post-write verification of the file on disk.
    with safe_open(DST, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if len(keys) != EXPECTED_TOTAL:
            fail(f"written file has {len(keys)} tensors, expected {EXPECTED_TOTAL}")
        if set(keys) != set(out):
            fail("written key set differs from expected")
        for k in keys:
            if not torch.equal(f.get_tensor(k), out[k]):
                fail(f"written tensor {k} differs")

    print(f"OK: wrote {DST} with {len(keys)} tensors ({N_NEW} blocks)")


if __name__ == "__main__":
    main()
