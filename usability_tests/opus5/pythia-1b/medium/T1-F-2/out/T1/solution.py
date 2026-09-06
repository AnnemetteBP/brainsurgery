#!/usr/bin/env python3
"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering survivors.

Route: torch-state-bridge rule-based key rewriting (simultaneous rule set with
built-in collision detection) + safetensors for I/O.  All required checks are
enforced before anything is written; the output file is moved into place only
after every check passes.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T1"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

DROP = [2, 6, 10, 14]
N_OLD = 16
N_NEW = 12
EXPECTED_TENSORS = 184
TENSORS_PER_BLOCK = 15

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def die(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    sd = load_file(IN_PATH)
    if len(sd) != 244:
        die(f"input has {len(sd)} tensors, expected 244")

    keep = [i for i in range(N_OLD) if i not in DROP]
    if len(keep) != N_NEW:
        die(f"keep list has {len(keep)} blocks, expected {N_NEW}")

    # 1. Remove every tensor of the dropped blocks.
    pruned = {}
    for k, v in sd.items():
        m = LAYER_RE.match(k)
        if m is not None and int(m.group(1)) in DROP:
            continue
        pruned[k] = v
    if len(pruned) != EXPECTED_TENSORS:
        die(f"after pruning: {len(pruned)} tensors, expected {EXPECTED_TENSORS}")

    # 2. Renumber survivors.  One rule per surviving block, old -> new; the
    #    rules are written in ascending source order and every destination is
    #    strictly below its source, so no key can be rewritten twice, and
    #    state_bridge's collision detection catches it if that reasoning is wrong.
    rules = "\n".join(
        f"gpt_neox.layers.{old}., gpt_neox.layers.{new}."
        for new, old in enumerate(keep)
    )
    out = state_bridge(pruned, rules, detect_collision=True)

    # ---- Required checks -------------------------------------------------
    if len(out) != EXPECTED_TENSORS:
        die(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    idxs = sorted({int(m.group(1)) for k in out if (m := LAYER_RE.match(k))})
    if idxs != list(range(N_NEW)):
        die(f"block indices are {idxs}, expected 0..{N_NEW - 1}")

    for stale in (12, 13, 14, 15):
        leftover = [k for k in out if LAYER_RE.match(k) and int(LAYER_RE.match(k).group(1)) == stale]
        if leftover:
            die(f"block {stale} still present: {leftover[:3]}")

    qkv = [k for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)]
    if len(qkv) != N_NEW:
        die(f"{len(qkv)} query_key_value.weight tensors, expected {N_NEW}")

    block_keys = [k for k in out if LAYER_RE.match(k)]
    if len(block_keys) != N_NEW * TENSORS_PER_BLOCK:
        die(f"{len(block_keys)} block tensors, expected {N_NEW * TENSORS_PER_BLOCK}")

    # Non-block tensors untouched.
    nonblock = {k: v for k, v in out.items() if not LAYER_RE.match(k)}
    expected_nonblock = {
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    }
    if set(nonblock) != expected_nonblock:
        die(f"non-block tensors are {sorted(nonblock)}, expected {sorted(expected_nonblock)}")
    for k in expected_nonblock:
        if not torch.equal(out[k], sd[k]):
            die(f"non-block tensor {k} changed")

    # Anti-collision proof: every output block tensor is bit-identical to the
    # source tensor of the block it claims to come from, with an unchanged suffix.
    for new, old in enumerate(keep):
        for k, v in sd.items():
            if not k.startswith(f"gpt_neox.layers.{old}."):
                continue
            rest = k[len(f"gpt_neox.layers.{old}."):]
            nk = f"gpt_neox.layers.{new}.{rest}"
            if nk not in out:
                die(f"missing {nk} (from {k})")
            if out[nk].dtype != v.dtype or out[nk].shape != v.shape:
                die(f"{nk}: shape/dtype {tuple(out[nk].shape)}/{out[nk].dtype} != {tuple(v.shape)}/{v.dtype}")
            if not torch.equal(out[nk], v):
                die(f"{nk} does not match source {k} bit-exactly")

    # ---- Write only after all checks pass --------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=OUT_DIR, suffix=".tmp")
    os.close(fd)
    try:
        save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata={"format": "pt"})
        os.replace(tmp, OUT_PATH)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    written = load_file(OUT_PATH)
    if len(written) != EXPECTED_TENSORS or set(written) != set(out):
        os.remove(OUT_PATH)
        die("re-read of the written file does not match the checked state dict")
    print(f"OK: wrote {OUT_PATH} with {len(written)} tensors, blocks 0..{N_NEW - 1}")


if __name__ == "__main__":
    main()
