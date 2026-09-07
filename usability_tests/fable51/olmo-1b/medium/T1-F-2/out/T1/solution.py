"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers and renumber blocks.

Plain safetensors script. Surviving blocks are written into a fresh dict via a
single old->new index map, so no renumbering order can collide. All required
checks run before anything is written; the output file is written last, via a
temporary path that is renamed only after the checks pass.
"""
import json
import os
import re
import sys

from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = 12
EXPECT_TENSORS = 86
KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    src = {}
    for shard in shards:
        src.update(load_file(os.path.join(BASE, shard)))
    if len(src) != 114:
        fail(f"expected 114 input tensors, got {len(src)}")

    # Old index -> new index for survivors, in original order.
    keep = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}
    if len(remap) != N_NEW:
        fail(f"remap has {len(remap)} blocks, expected {N_NEW}")

    out = {}
    for key, tensor in src.items():
        m = KEY_RE.match(key)
        if m is None:
            out[key] = tensor  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old >= N_OLD:
            fail(f"unexpected block index {old} in {key}")
        if old in DROP:
            continue
        new_key = f"model.layers.{remap[old]}.{m.group(2)}"
        if new_key in out:
            fail(f"collision: {new_key} already written (from {key})")
        out[new_key] = tensor

    # Required checks.
    block_ids = sorted({int(m.group(1)) for k in out if (m := KEY_RE.match(k))})
    stale = [b for b in block_ids if b >= N_NEW]
    if stale:
        fail(f"blocks >= {N_NEW} remain: {stale}")
    if block_ids != list(range(N_NEW)):
        fail(f"block indices are not 0..{N_NEW - 1}: {block_ids}")
    q = [k for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
    if len(q) != N_NEW:
        fail(f"expected {N_NEW} q_proj tensors, got {len(q)}")
    for b in range(N_NEW):
        n = sum(1 for k in out if k.startswith(f"model.layers.{b}."))
        if n != 7:
            fail(f"block {b} has {n} tensors, expected 7")
    if len(out) != EXPECT_TENSORS:
        fail(f"expected {EXPECT_TENSORS} output tensors, got {len(out)}")

    # Value/shape/dtype identity against the source under the mapping.
    for old, new in remap.items():
        for rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            a = src[f"model.layers.{old}.{rest}.weight"]
            b = out[f"model.layers.{new}.{rest}.weight"]
            if a.shape != b.shape or a.dtype != b.dtype or a.data_ptr() != b.data_ptr():
                fail(f"mismatch for old block {old} -> new {new}: {rest}")
    for k in ("model.embed_tokens.weight", "lm_head.weight"):
        if out[k].data_ptr() != src[k].data_ptr():
            fail(f"non-block tensor {k} changed")

    tmp = OUT + ".tmp"
    save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata={"format": "pt"})
    os.replace(tmp, OUT)

    # Post-write verification of the file on disk.
    written = load_file(OUT)
    if len(written) != EXPECT_TENSORS or set(written) != set(out):
        os.remove(OUT)
        fail("written file does not match expected key set")
    print(f"OK: wrote {OUT} with {len(written)} tensors, blocks {block_ids[0]}..{block_ids[-1]}")


if __name__ == "__main__":
    main()
