"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer: rows 640..767 of q/k/v_proj and
columns 640..767 of o_proj. Everything else is copied through unchanged.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T2"
OUT_FILE = os.path.join(OUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
PRUNE_HEAD = 5
LO = PRUNE_HEAD * HEAD_DIM          # 640
HI = LO + HEAD_DIM                  # 768
EXPECTED_TENSORS = 114


def load_sharded(path):
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        state.update(load_file(os.path.join(path, shard)))
    missing = set(weight_map) - set(state)
    extra = set(state) - set(weight_map)
    if missing or extra:
        raise RuntimeError(f"shard/index mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return state


def drop_rows(t, name):
    if tuple(t.shape) != (HIDDEN, HIDDEN):
        raise RuntimeError(f"{name}: expected {(HIDDEN, HIDDEN)}, got {tuple(t.shape)}")
    return torch.cat([t[:LO, :], t[HI:, :]], dim=0).contiguous()


def drop_cols(t, name):
    if tuple(t.shape) != (HIDDEN, HIDDEN):
        raise RuntimeError(f"{name}: expected {(HIDDEN, HIDDEN)}, got {tuple(t.shape)}")
    return torch.cat([t[:, :LO], t[:, HI:]], dim=1).contiguous()


def main():
    state = load_sharded(IN_DIR)
    if len(state) != EXPECTED_TENSORS:
        raise RuntimeError(f"input has {len(state)} tensors, expected {EXPECTED_TENSORS}")

    out = {}
    touched = 0
    for name, tensor in state.items():
        parts = name.split(".")
        is_attn = (
            len(parts) >= 5
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[3] == "self_attn"
        )
        if is_attn and parts[4] in ("q_proj", "k_proj", "v_proj"):
            out[name] = drop_rows(tensor, name)
            touched += 1
        elif is_attn and parts[4] == "o_proj":
            out[name] = drop_cols(tensor, name)
            touched += 1
        else:
            out[name] = tensor.clone().contiguous()

    if touched != 4 * NUM_LAYERS:
        raise RuntimeError(f"pruned {touched} tensors, expected {4 * NUM_LAYERS}")

    # Required checks: fail loudly before writing anything.
    for proj in ("q_proj", "k_proj", "v_proj"):
        key = f"model.layers.0.self_attn.{proj}.weight"
        got = tuple(out[key].shape)
        if got != (1920, 2048):
            raise AssertionError(f"{key}: shape {got}, expected (1920, 2048)")
    key = "model.layers.0.self_attn.o_proj.weight"
    got = tuple(out[key].shape)
    if got != (2048, 1920):
        raise AssertionError(f"{key}: shape {got}, expected (2048, 1920)")
    if len(out) != EXPECTED_TENSORS:
        raise AssertionError(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # Same checks across every layer, plus value-level spot check on the seam.
    for i in range(NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"model.layers.{i}.self_attn.{proj}.weight"
            if tuple(out[key].shape) != (1920, 2048):
                raise AssertionError(f"{key}: shape {tuple(out[key].shape)}")
            if not torch.equal(out[key][:LO], state[key][:LO]):
                raise AssertionError(f"{key}: rows before the pruned head changed")
            if not torch.equal(out[key][LO:], state[key][HI:]):
                raise AssertionError(f"{key}: rows after the pruned head misaligned")
        key = f"model.layers.{i}.self_attn.o_proj.weight"
        if tuple(out[key].shape) != (2048, 1920):
            raise AssertionError(f"{key}: shape {tuple(out[key].shape)}")
        if not torch.equal(out[key][:, :LO], state[key][:, :LO]):
            raise AssertionError(f"{key}: columns before the pruned head changed")
        if not torch.equal(out[key][:, LO:], state[key][:, HI:]):
            raise AssertionError(f"{key}: columns after the pruned head misaligned")

    for name, tensor in out.items():
        if tensor.dtype != state[name].dtype:
            raise AssertionError(f"{name}: dtype changed {state[name].dtype} -> {tensor.dtype}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out, OUT_FILE)

    written = load_file(OUT_FILE)
    if len(written) != EXPECTED_TENSORS:
        raise AssertionError(f"wrote {len(written)} tensors, expected {EXPECTED_TENSORS}")
    if set(written) != set(state):
        raise AssertionError("tensor names changed on write")
    print(f"wrote {OUT_FILE}: {len(written)} tensors, pruned head {PRUNE_HEAD} in {NUM_LAYERS} layers")


if __name__ == "__main__":
    main()
