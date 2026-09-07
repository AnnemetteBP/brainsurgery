"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, renumbering survivors.

Reads the sharded safetensors checkpoint under inputs/base/, drops blocks
2, 6, 10, 14, renumbers the remaining blocks 0..11 in original order, and
writes a single out/T1/model.safetensors. Fails loudly (non-zero exit, no
output written) if any required check does not hold.
"""

import json
import os
import re
import sys

import torch  # noqa: F401  (ensures torch tensors are the backing type)
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base")
OUT = os.path.join(HERE, "model.safetensors")

N_LAYERS_IN = 16
DROP = {2, 6, 10, 14}
TENSORS_PER_BLOCK = 7
N_NON_BLOCK = 2
EXPECTED_IN = N_LAYERS_IN * TENSORS_PER_BLOCK + N_NON_BLOCK  # 114
N_LAYERS_OUT = N_LAYERS_IN - len(DROP)  # 12
EXPECTED_OUT = N_LAYERS_OUT * TENSORS_PER_BLOCK + N_NON_BLOCK  # 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # --- load all shards into one dict ------------------------------------
    with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state = {}
    for shard in shard_files:
        part = load_file(os.path.join(BASE, shard))
        dup = set(part) & set(state)
        if dup:
            fail(f"duplicate keys across shards: {sorted(dup)[:5]}")
        state.update(part)
    if set(state) != set(index["weight_map"]):
        fail("loaded keys do not match the index weight_map")
    if len(state) != EXPECTED_IN:
        fail(f"expected {EXPECTED_IN} input tensors, got {len(state)}")

    # --- build old -> new block index map (order-preserving) --------------
    survivors = [i for i in range(N_LAYERS_IN) if i not in DROP]
    remap = {old: new for new, old in enumerate(survivors)}
    if len(remap) != N_LAYERS_OUT:
        fail(f"remap has {len(remap)} entries, expected {N_LAYERS_OUT}")

    # --- construct new state dict into a fresh dict (no in-place collisions)
    new_state = {}
    seen_blocks_in = set()
    for name, tensor in state.items():
        m = LAYER_RE.match(name)
        if m is None:
            # non-block tensor: pass through unchanged
            new_state[name] = tensor
            continue
        old = int(m.group(1))
        rest = m.group(2)
        if old >= N_LAYERS_IN:
            fail(f"unexpected block index {old} in {name}")
        seen_blocks_in.add(old)
        if old in DROP:
            continue
        new_name = f"model.layers.{remap[old]}.{rest}"
        if new_name in new_state:
            fail(f"collision: {new_name} already written (from {name})")
        new_state[new_name] = tensor
    if seen_blocks_in != set(range(N_LAYERS_IN)):
        fail(f"input blocks seen {sorted(seen_blocks_in)} != 0..{N_LAYERS_IN - 1}")

    # --- required checks --------------------------------------------------
    # (a) no tensor of blocks 12..15 remains
    out_blocks = {}
    for name in new_state:
        m = LAYER_RE.match(name)
        if m:
            out_blocks.setdefault(int(m.group(1)), []).append(m.group(2))
    stale = sorted(b for b in out_blocks if b >= N_LAYERS_OUT)
    if stale:
        fail(f"tensors of blocks {stale} remain in the output")

    # (b) exactly 12 blocks remain, contiguous 0..11, each with 7 tensors
    if sorted(out_blocks) != list(range(N_LAYERS_OUT)):
        fail(f"output block indices {sorted(out_blocks)} != 0..{N_LAYERS_OUT - 1}")
    q_count = sum(
        1 for n in new_state if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", n)
    )
    if q_count != N_LAYERS_OUT:
        fail(f"{q_count} q_proj tensors, expected {N_LAYERS_OUT}")
    for b, rests in out_blocks.items():
        if len(rests) != TENSORS_PER_BLOCK:
            fail(f"block {b} has {len(rests)} tensors, expected {TENSORS_PER_BLOCK}")

    # (c) exactly 86 tensors
    if len(new_state) != EXPECTED_OUT:
        fail(f"output has {len(new_state)} tensors, expected {EXPECTED_OUT}")

    # (d) every survivor is the identical tensor object from its source block
    for old, new in remap.items():
        for rest in out_blocks[new]:
            src = state[f"model.layers.{old}.{rest}"]
            dst = new_state[f"model.layers.{new}.{rest}"]
            if dst is not src:
                fail(f"model.layers.{new}.{rest} is not sourced from block {old}")
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if name not in new_state or new_state[name] is not state[name]:
            fail(f"non-block tensor {name} missing or altered")

    # --- write (only reached if all checks passed) ------------------------
    if os.path.exists(OUT):
        fail(f"output already exists: {OUT}")
    new_state = {k: v.contiguous() for k, v in new_state.items()}
    save_file(new_state, OUT, metadata={"format": "pt"})

    # --- post-write verification ------------------------------------------
    written = load_file(OUT)
    if set(written) != set(new_state):
        os.remove(OUT)
        fail("written key set differs from intended key set")
    for k, v in written.items():
        if v.shape != new_state[k].shape or v.dtype != new_state[k].dtype:
            os.remove(OUT)
            fail(f"shape/dtype mismatch after write for {k}")
    print(f"OK: wrote {len(written)} tensors to {OUT}")
    print(f"block map (old -> new): {remap}")


if __name__ == "__main__":
    main()
