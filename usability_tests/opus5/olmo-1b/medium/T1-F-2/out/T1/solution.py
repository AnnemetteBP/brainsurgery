#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks, renumbering survivors.

Tools: safetensors (load/save), torch-state-bridge (rule-based key rewriting
with collision detection). See REPORT.md.

Renumbering is done in two rule passes through a temporary namespace, because
torch_state_bridge.RuleEngine applies *every* rule in sequence to each key: a
direct rule set (3->2, 4->3, ...) would cascade old 4 into 2 and silently
collide. Rewriting into `model.__pruned__.<new>.` first makes each key match at
most one rule of pass 1; pass 2 renames the namespace back.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from torch_state_bridge import state_bridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "inputs" / "base"
DST_DIR = ROOT / "out" / "T1"
DST = DST_DIR / "model.safetensors"

N_OLD = 16
DROP = (2, 6, 10, 14)
KEEP = [i for i in range(N_OLD) if i not in DROP]          # 12 survivors
N_NEW = len(KEEP)
TENSORS_PER_BLOCK = 7
EXPECTED_TOTAL = N_NEW * TENSORS_PER_BLOCK + 2             # 86
NON_BLOCK = {"model.embed_tokens.weight", "lm_head.weight"}
TMP = "model.__pruned__."
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_state_dict() -> dict:
    index = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    sd = {}
    for shard in sorted(set(index.values())):
        with safe_open(SRC / shard, framework="pt") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    if len(sd) != len(index):
        die(f"loaded {len(sd)} tensors, index lists {len(index)}")
    return sd


def main() -> None:
    sd = load_state_dict()

    # --- prune whole blocks by pattern -------------------------------------
    dropped = {k for k in sd if (m := LAYER_RE.match(k)) and int(m.group(1)) in DROP}
    if len(dropped) != len(DROP) * TENSORS_PER_BLOCK:
        die(f"dropped {len(dropped)} tensors, expected {len(DROP) * TENSORS_PER_BLOCK}")
    pruned = {k: v for k, v in sd.items() if k not in dropped}

    # --- renumber survivors, two passes through a temporary namespace ------
    rules_1 = "\n".join(f"model.layers.{old}., {TMP}{new}." for new, old in enumerate(KEEP))
    rules_2 = f"{TMP}, model.layers."
    staged = state_bridge(pruned, rules_1, detect_collision=True)
    out = state_bridge(staged, rules_2, detect_collision=True)

    # --- required checks ---------------------------------------------------
    if len(out) != EXPECTED_TOTAL:
        die(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    indices = sorted({int(m.group(1)) for k in out if (m := LAYER_RE.match(k))})
    if indices != list(range(N_NEW)):
        die(f"block indices are {indices}, expected 0..{N_NEW - 1}")

    stale = [k for k in out if (m := LAYER_RE.match(k)) and int(m.group(1)) >= N_NEW]
    if stale:
        die(f"tensors of removed blocks 12..15 remain: {sorted(stale)[:5]}")

    n_q = sum(1 for k in out if LAYER_RE.match(k) and k.endswith(".self_attn.q_proj.weight"))
    if n_q != N_NEW:
        die(f"{n_q} blocks match self_attn.q_proj.weight, expected {N_NEW}")

    if not NON_BLOCK <= set(out):
        die(f"missing non-block tensors: {sorted(NON_BLOCK - set(out))}")
    for k in NON_BLOCK:
        if out[k] is not sd[k]:
            die(f"non-block tensor {k} was not passed through unchanged")

    # every survivor kept its suffix, values, shape and dtype
    for new, old in enumerate(KEEP):
        for k in (k for k in sd if k.startswith(f"model.layers.{old}.")):
            nk = f"model.layers.{new}." + k[len(f"model.layers.{old}."):]
            if nk not in out:
                die(f"expected renamed tensor {nk} (from {k}) missing")
            if out[nk] is not sd[k]:
                die(f"{nk} is not the original tensor of {k}")

    # --- write -------------------------------------------------------------
    DST_DIR.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, DST)

    with safe_open(DST, framework="pt") as f:
        written = list(f.keys())
    if len(written) != EXPECTED_TOTAL or set(written) != set(out):
        DST.unlink(missing_ok=True)
        die(f"readback mismatch: {len(written)} tensors on disk")

    print(f"OK: wrote {DST} with {len(written)} tensors, {N_NEW} blocks 0..{N_NEW - 1}")


if __name__ == "__main__":
    main()
