from __future__ import annotations

import torch

from brainsurgery.engine.execution import _execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
from brainsurgery.engine.state_dicts import _InMemoryStateDict


class _Provider:
    def __init__(self, state_dicts: dict[str, _InMemoryStateDict]) -> None:
        self._state_dicts = state_dicts

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        return self._state_dicts[model]


def _make_state_dict(values: dict[str, torch.Tensor]) -> _InMemoryStateDict:
    sd = _InMemoryStateDict()
    for key, tensor in values.items():
        sd[key] = tensor
    return sd


def _small_moe_state_dict() -> dict[str, torch.Tensor]:
    gate_base = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    gate_delta = torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    up_base = torch.tensor([[4.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
    up_delta = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32)
    down_base = torch.tensor([[2.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    down_delta = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    return {
        "model.layers.0.mlp.experts.0.gate_proj.weight": gate_base,
        "model.layers.0.mlp.experts.1.gate_proj.weight": gate_base + gate_delta,
        "model.layers.0.mlp.experts.0.up_proj.weight": up_base,
        "model.layers.0.mlp.experts.1.up_proj.weight": up_base + up_delta,
        "model.layers.0.mlp.experts.0.down_proj.weight": down_base,
        "model.layers.0.mlp.experts.1.down_proj.weight": down_base + down_delta,
    }


def test_moe_to_low_rank_expert_plan_rewrites_expert_in_place() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {
                "subtract_": {
                    "from": r"model.layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight",
                    "to": r"model.layers.\1.mlp.experts.1.gate_proj.weight",
                }
            },
            {
                "subtract_": {
                    "from": r"model.layers\.(\d+)\.mlp\.experts\.0\.up_proj\.weight",
                    "to": r"model.layers.\1.mlp.experts.1.up_proj.weight",
                }
            },
            {
                "subtract_": {
                    "from": r"model.layers\.(\d+)\.mlp\.experts\.0\.down_proj\.weight",
                    "to": r"model.layers.\1.mlp.experts.1.down_proj.weight",
                }
            },
            {
                "phlora_": {
                    "target": r"model.layers\.(\d+)\.mlp\.experts\.1\.gate_proj\.weight",
                    "rank": 1,
                }
            },
            {
                "phlora_": {
                    "target": r"model.layers\.(\d+)\.mlp\.experts\.1\.up_proj\.weight",
                    "rank": 1,
                }
            },
            {
                "phlora_": {
                    "target": r"model.layers\.(\d+)\.mlp\.experts\.1\.down_proj\.weight",
                    "rank": 1,
                }
            },
            {
                "add_": {
                    "from": r"model.layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight",
                    "to": r"model.layers.\1.mlp.experts.1.gate_proj.weight",
                }
            },
            {
                "add_": {
                    "from": r"model.layers\.(\d+)\.mlp\.experts\.0\.up_proj\.weight",
                    "to": r"model.layers.\1.mlp.experts.1.up_proj.weight",
                }
            },
            {
                "add_": {
                    "from": r"model.layers\.(\d+)\.mlp\.experts\.0\.down_proj\.weight",
                    "to": r"model.layers.\1.mlp.experts.1.down_proj.weight",
                }
            },
        ],
    }

    dense = _small_moe_state_dict()
    original = {key: value.clone() for key, value in dense.items()}
    plan = compile_plan(raw)
    provider = _Provider({"model": _make_state_dict(dense)})

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])

    out = provider.get_state_dict("model")
    assert torch.equal(
        out["model.layers.0.mlp.experts.0.gate_proj.weight"],
        original["model.layers.0.mlp.experts.0.gate_proj.weight"],
    )

    for proj in ("gate_proj", "up_proj", "down_proj"):
        rewritten = out[f"model.layers.0.mlp.experts.1.{proj}.weight"]
        anchor = original[f"model.layers.0.mlp.experts.0.{proj}.weight"]
        original_expert = original[f"model.layers.0.mlp.experts.1.{proj}.weight"]
        ranked_delta = original_expert - anchor
        u, s, vh = torch.linalg.svd(ranked_delta, full_matrices=False)
        expected = anchor + (u[:, :1] * s[:1]) @ vh[:1, :]
        assert torch.allclose(rewritten, expected)
        assert rewritten.shape == original_expert.shape
