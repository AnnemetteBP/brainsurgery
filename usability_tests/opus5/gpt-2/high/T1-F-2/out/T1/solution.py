#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks with contiguous renumbering.

Route: torch-state-bridge rule chains for the key rewrite (it is the
condition-F tool made for rule-based key renaming), safetensors for I/O.

The rename is done in two phases through a marker namespace
(``h.<old>.`` -> ``h.#<new>#.`` -> ``h.<new>.``) because RuleEngine applies
its rules *sequentially* to every key, so a single-phase old->new rule set
could cascade one rename into a later rule and silently collide.  The marker
namespace makes the second phase disjoint from the first, so order does not
matter.

Nothing is written until every required check has passed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch_state_bridge import parse_rules

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")

DROP = (2, 5, 8)
N_OLD = 12
TENSORS_PER_BLOCK = 13
N_NON_BLOCK = 4

BLOCK_RE = re.compile(r"^h\.(\d+)\.")


class CheckFailed(Exception):
    """A required check did not hold."""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def build_engines() -> tuple[object, object, dict[int, int]]:
    """Return (phase1, phase2, old->new block map)."""
    keep = [i for i in range(N_OLD) if i not in DROP]
    mapping = {old: new for new, old in enumerate(keep)}
    phase1 = "\n".join(f"h.{old}., h.#{new}#." for old, new in mapping.items())
    phase2 = "\n".join(f"h.#{new}#., h.{new}." for new in mapping.values())
    return parse_rules(phase1), parse_rules(phase2), mapping


def main() -> int:
    require(SRC.is_file(), f"input not found: {SRC}")
    src = load_file(str(SRC))

    # --- sanity on the input ---------------------------------------------
    require(
        len(src) == N_OLD * TENSORS_PER_BLOCK + N_NON_BLOCK,
        f"input has {len(src)} tensors, expected {N_OLD * TENSORS_PER_BLOCK + N_NON_BLOCK}",
    )

    p1, p2, mapping = build_engines()

    # --- rewrite ----------------------------------------------------------
    out: dict = {}
    provenance: dict[str, str] = {}  # new key -> old key
    for key, tensor in src.items():
        m = BLOCK_RE.match(key)
        if m is None:
            new_key = key  # non-block tensor, untouched
        else:
            idx = int(m.group(1))
            require(idx < N_OLD, f"unexpected block index in {key!r}")
            if idx in DROP:
                continue  # requirement 1: drop blocks 2, 5, 8 entirely
            new_key = p2.apply(p1.apply(key))
            require(
                new_key != key or mapping[idx] == idx,
                f"rule chain did not rewrite {key!r}",
            )
            require(
                "#" not in new_key,
                f"marker namespace leaked into output key {new_key!r}",
            )
            require(
                new_key == f"h.{mapping[idx]}." + key.split(".", 2)[2],
                f"rewrite of {key!r} produced unexpected {new_key!r}",
            )
        # requirement: no collisions -- a surviving block must never be
        # overwritten by another block.
        require(
            new_key not in out,
            f"collision: {new_key!r} written by both {provenance.get(new_key)!r} and {key!r}",
        )
        out[new_key] = tensor
        provenance[new_key] = key

    # --- required checks --------------------------------------------------
    blocks = {int(m.group(1)) for k in out if (m := BLOCK_RE.match(k))}

    # (a) no tensor of blocks 9, 10, 11 remains
    stale = sorted(k for k in out if BLOCK_RE.match(k) and int(BLOCK_RE.match(k).group(1)) >= 9)
    require(not stale, f"tensors of blocks >= 9 remain: {stale[:5]}")

    # (b) exactly 9 blocks remain
    n_cattn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    require(n_cattn == 9, f"{n_cattn} tensors match h.<i>.attn.c_attn.weight, expected 9")
    require(blocks == set(range(9)), f"block indices are {sorted(blocks)}, expected 0..8")

    # (c) exactly 121 tensors
    require(len(out) == 121, f"output has {len(out)} tensors, expected 121")

    # --- structural checks beyond the required minimum --------------------
    for i in range(9):
        n = sum(1 for k in out if BLOCK_RE.match(k) and int(BLOCK_RE.match(k).group(1)) == i)
        require(n == TENSORS_PER_BLOCK, f"block {i} has {n} tensors, expected {TENSORS_PER_BLOCK}")

    # the three required checks are structural: they would also accept dropping
    # some *other* three blocks.  Pin the mapping to the one the task specifies.
    require(
        mapping == {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8},
        f"block mapping {mapping} is not the one the task specifies",
    )

    expected = set()
    for old, new in mapping.items():
        for k in src:
            if k.startswith(f"h.{old}."):
                expected.add(f"h.{new}." + k.split(".", 2)[2])
    expected |= {k for k in src if not BLOCK_RE.match(k)}
    require(set(out) == expected, "output key set differs from the expected key set")

    for new_key, old_key in provenance.items():
        a, b = out[new_key], src[old_key]
        require(a.shape == b.shape, f"shape changed for {new_key}")
        require(a.dtype == b.dtype, f"dtype changed for {new_key}")
        require(a.data_ptr() == b.data_ptr(), f"{new_key} is not the original tensor")

    for k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        require(k in out and out[k].data_ptr() == src[k].data_ptr(), f"non-block tensor {k} changed")

    # --- write ------------------------------------------------------------
    DST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DST.with_suffix(".safetensors.tmp")
    save_file({k: v.contiguous() for k, v in out.items()}, str(tmp), metadata={"format": "pt"})

    # verify what landed on disk before publishing it
    try:
        back = load_file(str(tmp))
        require(len(back) == 121, f"written file has {len(back)} tensors")
        require(set(back) == set(out), "written key set differs")
        for k in back:
            require(back[k].dtype == out[k].dtype, f"dtype changed on disk for {k}")
            require(back[k].shape == out[k].shape, f"shape changed on disk for {k}")
            require(bool((back[k] == out[k]).all()), f"values changed on disk for {k}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(DST)

    print(f"wrote {DST} : {len(out)} tensors, blocks 0..8")
    print("kept old blocks -> new: " + ", ".join(f"{o}->{n}" for o, n in sorted(mapping.items())))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        DST.unlink(missing_ok=True)
        sys.exit(1)
