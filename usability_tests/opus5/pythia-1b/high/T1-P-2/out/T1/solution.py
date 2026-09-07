"""T1: depth-prune Pythia-1B from 16 to 12 transformer blocks.

Drops blocks 2, 6, 10, 14 and renumbers the survivors to 0..11, keeping the
rest of every name, and all values, shapes and dtypes, unchanged.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SANDBOX = Path(__file__).resolve().parents[2]
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST_DIR = SANDBOX / "out" / "T1"
DST = DST_DIR / "model.safetensors"

DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = 12
N_EXPECTED_TENSORS = 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def same_bits(a, b):
    """Bit-exact comparison; goes through raw bytes so NaNs compare equal."""
    return a.numpy().tobytes() == b.numpy().tobytes()


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # Old block index -> new block index, survivors in original order.
    survivors = [i for i in range(N_OLD) if i not in DROP]
    if len(survivors) != N_NEW:
        fail(f"expected {N_NEW} surviving blocks, got {len(survivors)}")
    remap = {old: new for new, old in enumerate(survivors)}

    with safe_open(SRC, framework="pt") as f:
        metadata = f.metadata()
        src_keys = list(f.keys())
        new_state = {}
        for key in src_keys:
            m = LAYER_RE.match(key)
            if m is None:
                # Non-block tensor: embeddings, final layer norm, lm head.
                new_key = key
            else:
                old_idx = int(m.group(1))
                if old_idx not in remap:
                    continue  # dropped block
                new_key = f"gpt_neox.layers.{remap[old_idx]}.{m.group(2)}"
            if new_key in new_state:
                fail(f"name collision: {key} -> {new_key} already present")
            new_state[new_key] = f.get_tensor(key)

    # --- Required checks (before anything is written) ---

    if len(src_keys) != 244:
        fail(f"input has {len(src_keys)} tensors, expected 244")

    # No tensor of blocks 12..15 remains.
    stale = sorted(k for k in new_state if LAYER_RE.match(k)
                   and int(LAYER_RE.match(k).group(1)) >= N_NEW)
    if stale:
        fail(f"{len(stale)} tensor(s) of blocks >= {N_NEW} remain, e.g. {stale[0]}")

    # Exactly 12 blocks remain, indices contiguous 0..11.
    qkv = sorted(int(LAYER_RE.match(k).group(1)) for k in new_state
                 if LAYER_RE.match(k)
                 and LAYER_RE.match(k).group(2) == "attention.query_key_value.weight")
    if qkv != list(range(N_NEW)):
        fail(f"block indices are {qkv}, expected {list(range(N_NEW))}")

    idx_seen = sorted({int(LAYER_RE.match(k).group(1)) for k in new_state if LAYER_RE.match(k)})
    if idx_seen != list(range(N_NEW)):
        fail(f"block index set is {idx_seen}, expected {list(range(N_NEW))}")

    # Exactly 184 tensors out.
    if len(new_state) != N_EXPECTED_TENSORS:
        fail(f"output would have {len(new_state)} tensors, expected {N_EXPECTED_TENSORS}")

    # Every surviving block carries its full 15 tensors, byte-identical to the
    # old block it came from.
    with safe_open(SRC, framework="pt") as f:
        for old, new in remap.items():
            old_suffixes = {LAYER_RE.match(k).group(2) for k in src_keys
                            if LAYER_RE.match(k) and int(LAYER_RE.match(k).group(1)) == old}
            new_suffixes = {LAYER_RE.match(k).group(2) for k in new_state
                            if LAYER_RE.match(k) and int(LAYER_RE.match(k).group(1)) == new}
            if len(old_suffixes) != 15 or old_suffixes != new_suffixes:
                fail(f"block {old} -> {new}: tensor set differs "
                     f"({len(old_suffixes)} vs {len(new_suffixes)})")
            for suf in old_suffixes:
                src_t = f.get_tensor(f"gpt_neox.layers.{old}.{suf}")
                dst_t = new_state[f"gpt_neox.layers.{new}.{suf}"]
                if src_t.shape != dst_t.shape or src_t.dtype != dst_t.dtype:
                    fail(f"block {old} -> {new} tensor {suf}: shape/dtype changed")
                if not same_bits(src_t, dst_t):
                    fail(f"block {old} -> {new} tensor {suf}: values changed")

        # Non-block tensors untouched.
        for key in ("gpt_neox.embed_in.weight", "embed_out.weight",
                    "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias"):
            if key not in new_state:
                fail(f"non-block tensor {key} missing from output")
            src_t = f.get_tensor(key)
            dst_t = new_state[key]
            if src_t.shape != dst_t.shape or src_t.dtype != dst_t.dtype:
                fail(f"non-block tensor {key}: shape/dtype changed")
            if not same_bits(src_t, dst_t):
                fail(f"non-block tensor {key}: values changed")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in new_state.items()}, DST,
              metadata=metadata or {"format": "pt"})

    print(f"wrote {DST} with {len(new_state)} tensors, {N_NEW} blocks "
          f"(dropped {sorted(DROP)}, remap {remap})")


if __name__ == "__main__":
    main()
