"""Remove attention head 5 from every OLMo layer checkpoint tensor."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16
HIDDEN_SIZE = 2048
HEAD_SIZE = 128
HEAD_TO_REMOVE = 5
EXPECTED_TENSORS = 114


def main() -> None:
    with (BASE / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]

    shard_names = sorted(set(weight_map.values()))
    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = load_file(BASE / shard_name, device="cpu")
        overlap = state.keys() & shard.keys()
        assert not overlap, f"duplicate tensor keys across shards: {sorted(overlap)}"
        state.update(shard)

    assert set(state) == set(weight_map), "loaded key set differs from shard index"
    assert len(state) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} input tensors, got {len(state)}"
    )

    cut_start = HEAD_TO_REMOVE * HEAD_SIZE
    cut_end = cut_start + HEAD_SIZE
    assert (cut_start, cut_end) == (640, 768)

    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            key = f"{prefix}.{projection}.weight"
            tensor = state[key]
            assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"unexpected source shape for {key}: {tuple(tensor.shape)}"
            )
            state[key] = torch.cat(
                (tensor[:cut_start, :], tensor[cut_end:, :]), dim=0
            )

        key = f"{prefix}.o_proj.weight"
        tensor = state[key]
        assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
            f"unexpected source shape for {key}: {tuple(tensor.shape)}"
        )
        state[key] = torch.cat(
            (tensor[:, :cut_start], tensor[:, cut_end:]), dim=1
        )

    # Required pre-write checks.
    layer0 = "model.layers.0.self_attn"
    assert tuple(state[f"{layer0}.q_proj.weight"].shape) == (1920, 2048)
    assert tuple(state[f"{layer0}.k_proj.weight"].shape) == (1920, 2048)
    assert tuple(state[f"{layer0}.v_proj.weight"].shape) == (1920, 2048)
    assert tuple(state[f"{layer0}.o_proj.weight"].shape) == (2048, 1920)
    assert len(state) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} output tensors, got {len(state)}"
    )

    # Also protect every layer against an incomplete or wrong-axis edit.
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            assert tuple(state[f"{prefix}.{projection}.weight"].shape) == (1920, 2048)
        assert tuple(state[f"{prefix}.o_proj.weight"].shape) == (2048, 1920)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, OUTPUT)


if __name__ == "__main__":
    main()
