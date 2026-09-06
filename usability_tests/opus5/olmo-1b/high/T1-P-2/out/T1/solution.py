"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks, renumbering survivors.

Drops blocks 2, 6, 10, 14 and renumbers the remaining blocks in their original
order so indices run 0..11 contiguously. Everything is built in memory and all
required checks run before a single byte is written, so a failed check leaves
no output behind.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "inputs" / "base"
OUT = HERE / "model.safetensors"

DROP = {2, 6, 10, 14}
N_LAYERS_IN = 16
N_LAYERS_OUT = 12
EXPECTED_IN_TENSORS = 114
EXPECTED_OUT_TENSORS = 86
PER_BLOCK = 7
NON_BLOCK = {"model.embed_tokens.weight", "lm_head.weight"}

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded(src: Path) -> dict:
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        fail(f"missing index {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state: dict = {}
    for shard in sorted(set(weight_map.values())):
        shard_tensors = load_file(str(src / shard))
        for name, tensor in shard_tensors.items():
            if name in state:
                fail(f"tensor {name!r} appears in more than one shard")
            state[name] = tensor
    missing = set(weight_map) - set(state)
    if missing:
        fail(f"index lists {len(missing)} tensor(s) absent from the shards: {sorted(missing)[:5]}")
    extra = set(state) - set(weight_map)
    if extra:
        fail(f"shards hold {len(extra)} tensor(s) absent from the index: {sorted(extra)[:5]}")
    return state


def main() -> None:
    state = load_sharded(SRC)
    if len(state) != EXPECTED_IN_TENSORS:
        fail(f"input has {len(state)} tensors, expected {EXPECTED_IN_TENSORS}")

    # Group input keys by block index and sanity-check the block layout.
    blocks: dict[int, list[str]] = {}
    others: list[str] = []
    for name in state:
        m = LAYER_RE.match(name)
        if m is None:
            others.append(name)
        else:
            blocks.setdefault(int(m.group(1)), []).append(name)

    if sorted(blocks) != list(range(N_LAYERS_IN)):
        fail(f"expected blocks 0..{N_LAYERS_IN - 1}, found {sorted(blocks)}")
    for idx, names in blocks.items():
        if len(names) != PER_BLOCK:
            fail(f"block {idx} owns {len(names)} tensors, expected {PER_BLOCK}")
    if set(others) != NON_BLOCK:
        fail(f"unexpected non-block tensors: {sorted(set(others) ^ NON_BLOCK)}")

    # Renumber survivors in original order: old index -> new contiguous index.
    survivors = [i for i in range(N_LAYERS_IN) if i not in DROP]
    if len(survivors) != N_LAYERS_OUT:
        fail(f"{len(survivors)} blocks survive, expected {N_LAYERS_OUT}")
    remap = {old: new for new, old in enumerate(survivors)}

    # Build a fresh dict, so a renumbering collision cannot silently overwrite
    # a surviving block the way an in-place rename would.
    out: dict = {}
    for name in others:
        out[name] = state[name]
    for old, new in remap.items():
        for name in blocks[old]:
            rest = LAYER_RE.match(name).group(2)
            new_name = f"model.layers.{new}.{rest}"
            if new_name in out:
                fail(f"renumbering collision: {name!r} -> {new_name!r} already present")
            tensor = state[name]
            out[new_name] = tensor.contiguous() if not tensor.is_contiguous() else tensor

    # --- Required checks (all before writing anything) ---

    # 1. No tensor of blocks 12, 13, 14, 15 remains.
    stale = sorted(n for n in out if (m := LAYER_RE.match(n)) and int(m.group(1)) >= N_LAYERS_OUT)
    if stale:
        fail(f"{len(stale)} tensor(s) of blocks >= {N_LAYERS_OUT} remain: {stale[:5]}")

    # 2. Exactly 12 blocks remain, witnessed per tensor role.
    out_blocks: dict[int, list[str]] = {}
    for name in out:
        m = LAYER_RE.match(name)
        if m is not None:
            out_blocks.setdefault(int(m.group(1)), []).append(name)
    if sorted(out_blocks) != list(range(N_LAYERS_OUT)):
        fail(f"output blocks are {sorted(out_blocks)}, expected 0..{N_LAYERS_OUT - 1}")
    for role in (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ):
        n = sum(1 for name in out if LAYER_RE.match(name) and name.endswith("." + role))
        if n != N_LAYERS_OUT:
            fail(f"{n} tensors match model.layers.<i>.{role}, expected {N_LAYERS_OUT}")

    # 3. Exactly 86 tensors in the output.
    if len(out) != EXPECTED_OUT_TENSORS:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_OUT_TENSORS}")

    # 4. Values, shapes and dtypes are carried over untouched.
    for old, new in remap.items():
        for name in blocks[old]:
            rest = LAYER_RE.match(name).group(2)
            src_t, dst_t = state[name], out[f"model.layers.{new}.{rest}"]
            if src_t.shape != dst_t.shape or src_t.dtype != dst_t.dtype:
                fail(f"{name!r} changed shape/dtype during renumbering")
            if dst_t.data_ptr() != src_t.data_ptr() and not dst_t.equal(src_t):
                fail(f"{name!r} changed values during renumbering")
    for name in NON_BLOCK:
        if out[name] is not state[name]:
            fail(f"non-block tensor {name!r} was replaced")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    written = load_file(str(OUT))
    if set(written) != set(out) or len(written) != EXPECTED_OUT_TENSORS:
        fail("readback of the written file does not match the intended key set")
    for name, tensor in written.items():
        if tensor.shape != out[name].shape or tensor.dtype != out[name].dtype:
            fail(f"readback mismatch for {name!r}")
        if not tensor.equal(out[name]):
            fail(f"readback values differ for {name!r}")

    dropped = ", ".join(str(i) for i in sorted(DROP))
    print(f"dropped blocks {dropped}; remap " + ", ".join(f"{o}->{n}" for o, n in remap.items()))
    print(f"wrote {OUT} with {len(written)} tensors ({N_LAYERS_OUT} blocks + {len(NON_BLOCK)})")


if __name__ == "__main__":
    main()
