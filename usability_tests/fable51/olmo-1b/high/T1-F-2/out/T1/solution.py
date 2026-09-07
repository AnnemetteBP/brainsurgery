"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers with contiguous renumbering.

Removes blocks 2, 6, 10, 14, renumbers the survivors in original order, and
writes a single out/T1/model.safetensors. Every check runs on the in-memory
result before anything is written; the file is staged to a temp path and
renamed into place only after the saved file has been re-read and verified.
"""

import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "inputs", "base")
OUT = os.path.join(HERE, "model.safetensors")
TMP = OUT + ".tmp"

N_LAYERS_IN = 16
DROP = {2, 6, 10, 14}
TENSORS_PER_BLOCK = 7
EXPECTED_TOTAL = 86
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    for p in (TMP, OUT):
        if os.path.exists(p):
            os.remove(p)
    sys.exit(1)


def main() -> None:
    with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))

    src = {}
    for shard in shards:
        with safe_open(os.path.join(BASE, shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in src:
                    fail(f"duplicate key across shards: {k}")
                src[k] = f.get_tensor(k)
    if len(src) != N_LAYERS_IN * TENSORS_PER_BLOCK + 2:
        fail(f"unexpected input tensor count {len(src)}")

    # Old index -> new index for survivors, in original order.
    keep = [i for i in range(N_LAYERS_IN) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}
    if len(remap) != 12:
        fail(f"expected 12 surviving blocks, computed {len(remap)}")

    dst = {}
    seen_blocks = set()
    for k, t in src.items():
        m = LAYER_RE.match(k)
        if m is None:
            dst[k] = t  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old not in remap:
            if old not in DROP:
                fail(f"block {old} is neither kept nor dropped: {k}")
            continue
        seen_blocks.add(old)
        nk = f"model.layers.{remap[old]}.{m.group(2)}"
        if nk in dst:
            fail(f"renumbering collision on {nk} (from {k})")
        dst[nk] = t
    if seen_blocks != set(keep):
        fail(f"surviving blocks seen {sorted(seen_blocks)} != expected {keep}")

    check(dst, src, remap)

    # Save to temp, re-read, verify the file, then move into place.
    save_file(dst, TMP, metadata={"format": "pt"})
    reread = {}
    with safe_open(TMP, framework="pt", device="cpu") as f:
        for k in f.keys():
            reread[k] = f.get_tensor(k)
    check(reread, src, remap)
    os.replace(TMP, OUT)
    print(f"OK: wrote {OUT} with {len(reread)} tensors, blocks 0..11")


def check(sd: dict, src: dict, remap: dict) -> None:
    """Required checks plus value/shape/dtype integrity against the source."""
    if len(sd) != EXPECTED_TOTAL:
        fail(f"output has {len(sd)} tensors, expected {EXPECTED_TOTAL}")
    blocks = {}
    for k in sd:
        m = LAYER_RE.match(k)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(m.group(2))
    stale = {i for i in blocks if i >= 12}
    if stale:
        fail(f"tensors of blocks {sorted(stale)} remain")
    if sorted(blocks) != list(range(12)):
        fail(f"block indices {sorted(blocks)} != 0..11")
    q = [k for k in sd if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
    if len(q) != 12:
        fail(f"{len(q)} q_proj tensors, expected 12")
    for i, rest in blocks.items():
        if len(rest) != TENSORS_PER_BLOCK:
            fail(f"block {i} has {len(rest)} tensors, expected {TENSORS_PER_BLOCK}")
    # Integrity: every output tensor is bit-identical to its source tensor.
    inv = {new: old for old, new in remap.items()}
    for k, t in sd.items():
        m = LAYER_RE.match(k)
        sk = f"model.layers.{inv[int(m.group(1))]}.{m.group(2)}" if m else k
        s = src.get(sk)
        if s is None:
            fail(f"{k} has no source tensor {sk}")
        if s.shape != t.shape or s.dtype != t.dtype:
            fail(f"{k}: shape/dtype {tuple(t.shape)}/{t.dtype} != source {tuple(s.shape)}/{s.dtype}")
        if not s.equal(t):
            fail(f"{k}: values differ from source {sk}")
    non_block = [k for k in sd if not LAYER_RE.match(k)]
    if sorted(non_block) != ["lm_head.weight", "model.embed_tokens.weight"]:
        fail(f"unexpected non-block tensors {non_block}")


if __name__ == "__main__":
    main()
