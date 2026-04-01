from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_reference_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "flexmore_examples"
        / "olmo_1b_0724_hf_moe_to_low_rank_expert_reference.py"
    )
    spec = importlib.util.spec_from_file_location("olmo_1b_0724_low_rank_reference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_moe_state_dict() -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for layer in range(16):
        offset = float(layer * 100)
        values[f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"] = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32) + offset
        )
        values[f"model.layers.{layer}.mlp.experts.1.gate_proj.weight"] = (
            torch.tensor([[3.0, 2.0], [0.0, 1.0]], dtype=torch.float32) + offset
        )
        values[f"model.layers.{layer}.mlp.experts.0.up_proj.weight"] = (
            torch.tensor([[4.0, 1.0], [0.0, 2.0]], dtype=torch.float32) + offset
        )
        values[f"model.layers.{layer}.mlp.experts.1.up_proj.weight"] = (
            torch.tensor([[4.0, 1.0], [3.0, 5.0]], dtype=torch.float32) + offset
        )
        values[f"model.layers.{layer}.mlp.experts.0.down_proj.weight"] = (
            torch.tensor([[2.0, 0.0], [0.0, 5.0]], dtype=torch.float32) + offset
        )
        values[f"model.layers.{layer}.mlp.experts.1.down_proj.weight"] = (
            torch.tensor([[3.0, 0.0], [1.0, 5.0]], dtype=torch.float32) + offset
        )
    return values


def test_reference_low_rank_converter_matches_expected_ranked_rewrite() -> None:
    module = _load_reference_module()
    dense = _synthetic_moe_state_dict()

    out = module.upcycle_moe_state_dict_to_low_rank_expert(dense, rank=1)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        anchor = dense[f"model.layers.0.mlp.experts.0.{proj}.weight"]
        original = dense[f"model.layers.0.mlp.experts.1.{proj}.weight"]
        delta = original - anchor
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        expected = anchor + (u[:, :1] * s[:1]) @ vh[:1, :]
        assert torch.allclose(out[f"model.layers.0.mlp.experts.1.{proj}.weight"], expected)

