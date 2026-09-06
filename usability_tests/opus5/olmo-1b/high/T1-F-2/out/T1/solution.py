#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 transformer blocks.

Drop blocks 2, 6, 10, 14; renumber the survivors to a contiguous 0..11 range;
leave `model.embed_tokens.weight` and `lm_head.weight` untouched.

Tools: safetensors for I/O, torch-state-bridge for the key rewriting.

The renumbering is done as two rule passes through a scratch namespace
(`model.__pruned__.<i>.`) rather than one in-place pass. torch-state-bridge
applies rules sequentially and feeds each rule's output into the next, so a
single-pass rule set can cascade (a key renamed 4->3 being picked up again by a
later 3->2 rule) and silently overwrite a surviving block. Routing through a
namespace that no source key and no rule source can match makes the rewrite
order-independent; `detect_collision=True` is the backstop on both passes.

Every check runs before anything is written, and the payload is written to a
temporary file that is only renamed into place once the write succeeds, so a
failing run leaves no output behind.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from torch_state_bridge import state_bridge

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
IN_DIR = SANDBOX / "inputs" / "base"
OUT_FILE = HERE / "model.safetensors"

# --- the specification, spelled out so the script fails if reality disagrees ---
N_BLOCKS_IN = 16
DROP_BLOCKS = [2, 6, 10, 14]
BLOCK_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]
NON_BLOCK_KEYS = ["model.embed_tokens.weight", "lm_head.weight"]

N_BLOCKS_OUT = 12
N_TENSORS_IN = 114
N_TENSORS_OUT = 86

# The renumbering exactly as TASK.md states it, kept literal so the derived
# mapping below is checked against the spec instead of just against itself.
EXPECTED_MAPPING = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7, 11: 8, 12: 9, 13: 10, 15: 11}

SCRATCH = "model.__pruned__."


class CheckFailed(RuntimeError):
    """A required check did not hold; nothing is written."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def build_mapping() -> dict[int, int]:
    """old block index -> new block index, derived and then checked against the spec."""
    survivors = [i for i in range(N_BLOCKS_IN) if i not in DROP_BLOCKS]
    mapping = {old: new for new, old in enumerate(survivors)}
    require(
        mapping == EXPECTED_MAPPING,
        f"derived renumbering {mapping} does not match the specified {EXPECTED_MAPPING}",
    )
    require(
        sorted(mapping.values()) == list(range(N_BLOCKS_OUT)),
        f"renumbered indices are not contiguous 0..{N_BLOCKS_OUT - 1}: {sorted(mapping.values())}",
    )
    return mapping


def load_surviving(mapping: dict[int, int]) -> dict[str, object]:
    """Read the sharded input, keeping only the tensors of surviving blocks."""
    index_path = IN_DIR / "model.safetensors.index.json"
    require(index_path.is_file(), f"missing shard index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    require(
        len(weight_map) == N_TENSORS_IN,
        f"input has {len(weight_map)} tensors, expected {N_TENSORS_IN}",
    )

    # Confirm the input really is the 16-block model with the documented names,
    # so a changed input surfaces here rather than as a wrong-sized output.
    expected_in = {
        f"model.layers.{i}.{suffix}" for i in range(N_BLOCKS_IN) for suffix in BLOCK_SUFFIXES
    } | set(NON_BLOCK_KEYS)
    require(
        set(weight_map) == expected_in,
        "input key set is not the expected 16-block OLMo-1B layout; "
        f"unexpected={sorted(set(weight_map) - expected_in)[:5]} "
        f"missing={sorted(expected_in - set(weight_map))[:5]}",
    )

    keep = {
        f"model.layers.{old}.{suffix}" for old in mapping for suffix in BLOCK_SUFFIXES
    } | set(NON_BLOCK_KEYS)
    require(
        len(keep) == N_TENSORS_OUT,
        f"selected {len(keep)} tensors to keep, expected {N_TENSORS_OUT}",
    )
    # Every tensor of a dropped block is excluded. This is checked here, on the
    # original numbering; it cannot be checked on the output, where indices
    # 2, 6 and 10 are valid names again for the renumbered survivors.
    still_dropped = sorted(
        k for i in DROP_BLOCKS for k in keep if k.startswith(f"model.layers.{i}.")
    )
    require(not still_dropped, f"tensors of dropped blocks {DROP_BLOCKS} selected: {still_dropped}")

    by_shard: dict[str, list[str]] = {}
    for key in sorted(keep):
        by_shard.setdefault(weight_map[key], []).append(key)

    tensors: dict[str, object] = {}
    for shard, keys in by_shard.items():
        with safe_open(IN_DIR / shard, framework="pt") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)

    require(
        len(tensors) == N_TENSORS_OUT,
        f"loaded {len(tensors)} tensors, expected {N_TENSORS_OUT}",
    )
    return tensors


def renumber(tensors: dict[str, object], mapping: dict[int, int]) -> dict[str, object]:
    """Rewrite block indices via a scratch namespace so no rule can cascade."""
    # Pass 1: model.layers.<old>. -> model.__pruned__.<new>.
    # No rule source can match a destination of this pass, so rule order is
    # irrelevant and no surviving block can be renamed twice.
    to_scratch = "\n".join(
        f"model.layers.{old}., {SCRATCH}{new}." for old, new in sorted(mapping.items())
    )
    # Pass 2: model.__pruned__.<n>. -> model.layers.<n>., a pure prefix swap.
    from_scratch = f"{SCRATCH}{{n}}., model.layers.{{n}}."

    staged = state_bridge(tensors, to_scratch, detect_collision=True)
    return state_bridge(staged, from_scratch, detect_collision=True)


def check_output(out: dict[str, object], src: dict[str, object], mapping: dict[int, int]) -> None:
    """Required checks from TASK.md, plus a full key/identity audit. Runs before any write."""
    names = set(out)

    # Required check 1: no tensor of a block index at or beyond the new depth.
    stale = sorted(
        name
        for name in names
        for i in range(N_BLOCKS_OUT, N_BLOCKS_IN)
        if name.startswith(f"model.layers.{i}.")
    )
    require(not stale, f"tensors of removed block indices 12..15 remain: {stale}")

    # Required check 2: exactly 12 blocks remain.
    q_proj = sorted(n for n in names if n.endswith(".self_attn.q_proj.weight"))
    require(
        len(q_proj) == N_BLOCKS_OUT,
        f"{len(q_proj)} blocks remain, expected {N_BLOCKS_OUT}: {q_proj}",
    )

    # Required check 3: exactly 86 tensors.
    require(len(out) == N_TENSORS_OUT, f"output has {len(out)} tensors, expected {N_TENSORS_OUT}")

    # Exact key set: 12 contiguous blocks x 7 suffixes, plus the 2 non-block tensors.
    expected = {
        f"model.layers.{new}.{suffix}"
        for new in range(N_BLOCKS_OUT)
        for suffix in BLOCK_SUFFIXES
    } | set(NON_BLOCK_KEYS)
    require(
        names == expected,
        f"output key set is wrong; unexpected={sorted(names - expected)} "
        f"missing={sorted(expected - names)}",
    )

    # Every output tensor is the very object loaded from its source key, which
    # proves values, shape and dtype are untouched and that nothing was
    # overwritten by a collision during renumbering.
    for old, new in mapping.items():
        for suffix in BLOCK_SUFFIXES:
            old_key = f"model.layers.{old}.{suffix}"
            new_key = f"model.layers.{new}.{suffix}"
            require(
                out[new_key] is src[old_key],
                f"{new_key} does not carry the tensor of {old_key}",
            )
    for key in NON_BLOCK_KEYS:
        require(out[key] is src[key], f"{key} was not carried through unchanged")

    # No two output names share a tensor object: a collision would have dropped
    # a block and duplicated another.
    ids = {id(t) for t in out.values()}
    require(len(ids) == N_TENSORS_OUT, f"output holds {len(ids)} distinct tensors, expected {N_TENSORS_OUT}")


def verify_written(path: Path, expected: dict[str, object]) -> None:
    """Re-open the written file and confirm key set, shapes, dtypes and values."""
    with safe_open(path, framework="pt") as handle:
        names = set(handle.keys())
        require(
            names == set(expected),
            f"written file key set differs; unexpected={sorted(names - set(expected))} "
            f"missing={sorted(set(expected) - names)}",
        )
        require(
            len(names) == N_TENSORS_OUT,
            f"written file has {len(names)} tensors, expected {N_TENSORS_OUT}",
        )
        for name in sorted(names):
            want = expected[name]
            got = handle.get_tensor(name)
            require(
                got.shape == want.shape and got.dtype == want.dtype,
                f"{name}: wrote {tuple(got.shape)}/{got.dtype}, "
                f"expected {tuple(want.shape)}/{want.dtype}",
            )
            require(got.equal(want), f"{name}: written values are not bit-identical to the source")


def main() -> int:
    try:
        mapping = build_mapping()
        src = load_surviving(mapping)
        # state_bridge returns a new dict of the same tensor objects; keep the
        # originals so check_output can verify old key -> new key routing.
        out = renumber(dict(src), mapping)
        check_output(out, src, mapping)
    except (CheckFailed, KeyError, ValueError, OSError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FILE.with_suffix(".safetensors.tmp")
    try:
        save_file(out, str(tmp))
        # Re-read the file just written and re-run the checks against what is
        # actually on disk, not against the in-memory dict.
        verify_written(tmp, out)
    except Exception as exc:  # noqa: BLE001 - report and leave no partial output
        tmp.unlink(missing_ok=True)
        print(f"FAILED while writing output: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    os.replace(tmp, OUT_FILE)
    print(
        f"wrote {OUT_FILE}: {len(out)} tensors, {N_BLOCKS_OUT} blocks, renumbering "
        + ", ".join(f"{old}->{new}" for old, new in sorted(mapping.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
