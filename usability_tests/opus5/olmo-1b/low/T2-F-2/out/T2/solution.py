"""T2: prune attention head 5 from every layer of OLMo-1B-0724-hf.

Plain script on safetensors + torch: load every shard, slice the four
head-bearing projections per layer, assert the required shapes, then write a
single out/T2/model.safetensors.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base")
DST = Path("out/T2/model.safetensors")

NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
PRUNE = 5

KEEP = torch.cat([
    torch.arange(0, PRUNE * HEAD_DIM),
    torch.arange((PRUNE + 1) * HEAD_DIM, HIDDEN),
])

ROW_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.[qkv]_proj\.weight$")
COL_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")


def main() -> None:
    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        for name, tensor in load_file(SRC / shard).items():
            if name in state:
                raise SystemExit(f"duplicate tensor across shards: {name}")
            state[name] = tensor

    if len(state) != len(index["weight_map"]):
        raise SystemExit(f"loaded {len(state)} tensors, index lists {len(index['weight_map'])}")
    n_in = len(state)

    touched = 0
    for name, tensor in list(state.items()):
        if ROW_RE.match(name):
            if tuple(tensor.shape) != (HIDDEN, HIDDEN):
                raise SystemExit(f"{name}: unexpected shape {tuple(tensor.shape)}")
            state[name] = tensor.index_select(0, KEEP).contiguous()
            touched += 1
        elif COL_RE.match(name):
            if tuple(tensor.shape) != (HIDDEN, HIDDEN):
                raise SystemExit(f"{name}: unexpected shape {tuple(tensor.shape)}")
            state[name] = tensor.index_select(1, KEEP).contiguous()
            touched += 1

    n_layers = json.loads((SRC / "config.json").read_text())["num_hidden_layers"]
    if touched != 4 * n_layers:
        raise SystemExit(f"expected {4 * n_layers} head-bearing tensors, sliced {touched}")

    # Required checks: fail loudly before writing.
    expected = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, shape in expected.items():
        if name not in state:
            raise SystemExit(f"missing required tensor {name}")
        if tuple(state[name].shape) != shape:
            raise SystemExit(f"{name}: got {tuple(state[name].shape)}, expected {shape}")
    if len(state) != 114:
        raise SystemExit(f"output has {len(state)} tensors, expected 114")
    if len(state) != n_in:
        raise SystemExit(f"tensor count changed: {n_in} -> {len(state)}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(DST))
    print(f"wrote {DST} with {len(state)} tensors ({touched} sliced)")


if __name__ == "__main__":
    main()
