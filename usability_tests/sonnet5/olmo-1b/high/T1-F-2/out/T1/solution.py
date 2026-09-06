"""
T1: depth pruning with layer renumbering (OLMo-1B-0724-hf).

Plain script on top of `safetensors` + `torch` (both allowed under
condition F). Loads the sharded input checkpoint, drops every tensor of
blocks 2, 6, 10, 14, renumbers the surviving blocks to 0..11 in original
order, leaves the 2 non-block tensors untouched, and writes a single
`out/T1/model.safetensors`.

Fails loudly (non-zero exit, no output file written) if any required
check does not hold.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent.parent / "inputs" / "base"
OUT_PATH = HERE / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
# Explicit old -> new block index map, in original order, skipping DROP_BLOCKS.
OLD_TO_NEW = {}
_new = 0
for _old in range(16):
    if _old in DROP_BLOCKS:
        continue
    OLD_TO_NEW[_old] = _new
    _new += 1
assert _new == 12, f"expected 12 surviving blocks, got {_new}"

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    index_path = IN_DIR / "model.safetensors.index.json"
    if not index_path.exists():
        fail(f"missing index file: {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Load every shard referenced by the index exactly once.
    shard_names = sorted(set(weight_map.values()))
    shards = {name: load_file(IN_DIR / name) for name in shard_names}

    state = {}
    for key, shard_name in weight_map.items():
        tensor = shards[shard_name][key]
        state[key] = tensor

    if len(state) != 114:
        fail(f"expected 114 input tensors, found {len(state)}")

    new_state = {}
    seen_new_blocks = set()
    for key, tensor in state.items():
        m = LAYER_RE.match(key)
        if m is None:
            # Non-block tensor: copy through unchanged.
            new_state[key] = tensor
            continue
        old_idx, rest = int(m.group(1)), m.group(2)
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = OLD_TO_NEW[old_idx]
        new_key = f"model.layers.{new_idx}.{rest}"
        if new_key in new_state:
            fail(f"collision writing {new_key} (from old block {old_idx})")
        new_state[new_key] = tensor
        seen_new_blocks.add(new_idx)

    # --- Required checks: fail loudly, write nothing, if any is violated ---

    # No tensor of blocks 12, 13, 14, 15 remains in the output naming.
    forbidden = re.compile(r"^model\.layers\.(12|13|14|15)\.")
    leaked = [k for k in new_state if forbidden.match(k)]
    if leaked:
        fail(f"forbidden block indices present in output: {leaked}")

    # Exactly 12 blocks remain.
    q_proj_blocks = {
        int(re.match(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", k).group(1))
        for k in new_state
        if re.match(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", k)
    }
    if q_proj_blocks != set(range(12)):
        fail(f"expected block indices 0..11, got {sorted(q_proj_blocks)}")

    # Each surviving block has exactly its 7 tensors.
    expected_suffixes = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    for idx in range(12):
        prefix = f"model.layers.{idx}."
        suffixes = {k[len(prefix):] for k in new_state if k.startswith(prefix)}
        if suffixes != expected_suffixes:
            fail(f"block {idx} has wrong tensor set: {sorted(suffixes)}")

    # The 2 non-block tensors are present and unchanged (bit-identical, same shape/dtype).
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if key not in new_state:
            fail(f"missing non-block tensor {key}")
        if not torch.equal(new_state[key], state[key]):
            fail(f"non-block tensor {key} was modified")

    # Exactly 86 tensors total.
    if len(new_state) != 86:
        fail(f"expected 86 output tensors, got {len(new_state)}")

    # Shapes/dtypes of surviving block tensors must match their old counterparts exactly.
    for old_idx, new_idx in OLD_TO_NEW.items():
        for suffix in expected_suffixes:
            old_key = f"model.layers.{old_idx}.{suffix}"
            new_key = f"model.layers.{new_idx}.{suffix}"
            old_t, new_t = state[old_key], new_state[new_key]
            if old_t.shape != new_t.shape or old_t.dtype != new_t.dtype:
                fail(f"{old_key} -> {new_key}: shape/dtype mismatch")
            if not torch.equal(old_t, new_t):
                fail(f"{old_key} -> {new_key}: values changed")

    # --- All checks passed: write output ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in new_state.items()}, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(new_state)} tensors")


if __name__ == "__main__":
    main()
