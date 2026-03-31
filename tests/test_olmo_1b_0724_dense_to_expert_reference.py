from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_reference_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "flexmore_examples"
        / "olmo_1b_0724_hf_dense_to_expert_moe_reference.py"
    )
    spec = importlib.util.spec_from_file_location("olmo_1b_0724_reference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_dense_state_dict() -> dict[str, torch.Tensor]:
    values = {
        "model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "lm_head.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) + 100,
    }
    for layer in range(16):
        offset = float(layer * 1000)
        values[f"model.layers.{layer}.self_attn.q_proj.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + 10
        )
        values[f"model.layers.{layer}.self_attn.k_proj.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + 20
        )
        values[f"model.layers.{layer}.self_attn.v_proj.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + 30
        )
        values[f"model.layers.{layer}.self_attn.o_proj.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + 40
        )
        values[f"model.layers.{layer}.mlp.gate_proj.weight"] = (
            torch.arange(32, dtype=torch.float32).reshape(8, 4) + offset + 50
        )
        values[f"model.layers.{layer}.mlp.up_proj.weight"] = (
            torch.arange(32, dtype=torch.float32).reshape(8, 4) + offset + 100
        )
        values[f"model.layers.{layer}.mlp.down_proj.weight"] = (
            torch.arange(32, dtype=torch.float32).reshape(4, 8) + offset + 150
        )
    return values


def test_reference_upcycling_matches_expected_structure() -> None:
    module = _load_reference_module()
    dense = _synthetic_dense_state_dict()

    out = module.upcycle_hf_dense_state_dict_to_expert_moe(dense, dense)

    assert torch.equal(out["model.embed_tokens.weight"], dense["model.embed_tokens.weight"])
    assert torch.equal(out["lm_head.weight"], dense["lm_head.weight"])

    for layer in range(16):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expert0 = out[f"model.layers.{layer}.mlp.experts.0.{proj}.weight"]
            expert1 = out[f"model.layers.{layer}.mlp.experts.1.{proj}.weight"]
            assert torch.equal(expert0, dense[f"model.layers.{layer}.mlp.{proj}.weight"])
            assert torch.equal(expert1, dense[f"model.layers.{layer}.mlp.{proj}.weight"])

        gate = out[f"model.layers.{layer}.mlp.gate.weight"]
        assert gate.shape == (2, 4)
        assert torch.equal(gate, torch.zeros((2, 4), dtype=torch.float32))

        assert f"model.layers.{layer}.mlp.gate_proj.weight" not in out
        assert f"model.layers.{layer}.mlp.up_proj.weight" not in out
        assert f"model.layers.{layer}.mlp.down_proj.weight" not in out
