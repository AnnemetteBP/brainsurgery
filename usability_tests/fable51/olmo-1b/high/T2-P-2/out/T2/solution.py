"""T2: prune attention head 5 from every layer of OLMo-1B-0724-hf.

Reads the sharded safetensors checkpoint under inputs/base/, slices the head-
bearing projections, and writes a single out/T2/model.safetensors.
"""

import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE = os.path.join(SANDBOX, "inputs", "base")
OUT = os.path.join(ROOT, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = 2048
PRUNE_HEAD = 5
EXPECTED_TENSORS = 114


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded(base: str) -> dict[str, torch.Tensor]:
    with open(os.path.join(base, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        part = load_file(os.path.join(base, shard))
        dup = set(part) & set(state)
        if dup:
            fail(f"duplicate keys across shards: {sorted(dup)[:5]}")
        state.update(part)
    missing = set(index["weight_map"]) - set(state)
    if missing:
        fail(f"index lists tensors not found in shards: {sorted(missing)[:5]}")
    return state


def keep_index() -> torch.Tensor:
    lo = PRUNE_HEAD * HEAD_DIM
    hi = (PRUNE_HEAD + 1) * HEAD_DIM
    return torch.cat([torch.arange(0, lo), torch.arange(hi, NUM_HEADS * HEAD_DIM)])


def main() -> None:
    state = load_sharded(BASE)
    if len(state) != EXPECTED_TENSORS:
        fail(f"input has {len(state)} tensors, expected {EXPECTED_TENSORS}")

    keep = keep_index()
    new_dim = (NUM_HEADS - 1) * HEAD_DIM  # 1920
    out: dict[str, torch.Tensor] = {}
    touched = 0
    for name, t in state.items():
        prefix, _, leaf = name.rpartition(".self_attn.")
        if prefix.startswith("model.layers.") and leaf in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
        ):
            if tuple(t.shape) != (HIDDEN, HIDDEN):
                fail(f"{name}: unexpected input shape {tuple(t.shape)}")
            if leaf == "o_proj.weight":
                t = t.index_select(1, keep)  # heads are column blocks
            else:
                t = t.index_select(0, keep)  # heads are row blocks
            touched += 1
        out[name] = t.contiguous()

    if touched != 4 * NUM_LAYERS:
        fail(f"sliced {touched} projections, expected {4 * NUM_LAYERS}")

    # Required checks.
    expect = {
        "model.layers.0.self_attn.q_proj.weight": (new_dim, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": (new_dim, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": (new_dim, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": (HIDDEN, new_dim),
    }
    for name, shape in expect.items():
        if name not in out:
            fail(f"missing {name}")
        if tuple(out[name].shape) != shape:
            fail(f"{name}: shape {tuple(out[name].shape)}, expected {shape}")
    if len(out) != EXPECTED_TENSORS:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")
    if set(out) != set(state):
        fail("output key set differs from input key set")

    # Extra sanity: every layer sliced, dtypes preserved, block order preserved.
    for i in range(NUM_LAYERS):
        for leaf, shape in (
            ("q_proj.weight", (new_dim, HIDDEN)),
            ("k_proj.weight", (new_dim, HIDDEN)),
            ("v_proj.weight", (new_dim, HIDDEN)),
            ("o_proj.weight", (HIDDEN, new_dim)),
        ):
            name = f"model.layers.{i}.self_attn.{leaf}"
            if tuple(out[name].shape) != shape:
                fail(f"{name}: shape {tuple(out[name].shape)}, expected {shape}")
    for name in out:
        if out[name].dtype != state[name].dtype:
            fail(f"{name}: dtype changed")
    q0, q0_in = out["model.layers.0.self_attn.q_proj.weight"], state[
        "model.layers.0.self_attn.q_proj.weight"
    ]
    if not torch.equal(q0[:640], q0_in[:640]) or not torch.equal(q0[640:], q0_in[768:]):
        fail("q_proj row blocks not preserved in order")
    o0, o0_in = out["model.layers.0.self_attn.o_proj.weight"], state[
        "model.layers.0.self_attn.o_proj.weight"
    ]
    if not torch.equal(o0[:, :640], o0_in[:, :640]) or not torch.equal(o0[:, 640:], o0_in[:, 768:]):
        fail("o_proj column blocks not preserved in order")

    if os.path.exists(OUT):
        fail(f"output already exists: {OUT}")
    save_file(out, OUT, metadata={"format": "pt"})
    print(f"wrote {OUT}: {len(out)} tensors, {touched} projections pruned")


if __name__ == "__main__":
    main()
