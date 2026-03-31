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


def _small_dense_state_dict() -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "lm_head.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) + 100,
    }
    for layer in range(2):
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


def test_olmo_dense_to_expert_plan_rewrites_all_layers() -> None:
    raw = {
        "inputs": ["m0::/tmp/m0.safetensors", "m1::/tmp/m1.safetensors"],
        "transforms": [
            {"assert": {"equal": {"left": "m0::model.embed_tokens.weight", "right": "m1::model.embed_tokens.weight"}}},
            {"assert": {"equal": {"left": "m0::lm_head.weight", "right": "m1::lm_head.weight"}}},
            {
                "assert": {
                    "equal": {
                        "left": r"m0::model.layers\.(\d+)\.self_attn\.q_proj\.weight",
                        "right": r"m1::model.layers.\1.self_attn.q_proj.weight",
                    }
                }
            },
            {
                "assert": {
                    "equal": {
                        "left": r"m0::model.layers\.(\d+)\.mlp\.gate_proj\.weight",
                        "right": r"m1::model.layers.\1.mlp.gate_proj.weight",
                    }
                }
            },
            {
                "copy": {
                    "from": r"m0::model.layers\.(\d+)\.mlp\.gate_proj\.weight",
                    "to": r"m0::model.layers.\1.mlp.experts.0.gate_proj.weight",
                }
            },
            {
                "copy": {
                    "from": r"m0::model.layers\.(\d+)\.mlp\.up_proj\.weight",
                    "to": r"m0::model.layers.\1.mlp.experts.0.up_proj.weight",
                }
            },
            {
                "copy": {
                    "from": r"m0::model.layers\.(\d+)\.mlp\.down_proj\.weight",
                    "to": r"m0::model.layers.\1.mlp.experts.0.down_proj.weight",
                }
            },
            {
                "copy": {
                    "from": r"m1::model.layers\.(\d+)\.mlp\.gate_proj\.weight",
                    "to": r"m0::model.layers.\1.mlp.experts.1.gate_proj.weight",
                }
            },
            {
                "copy": {
                    "from": r"m1::model.layers\.(\d+)\.mlp\.up_proj\.weight",
                    "to": r"m0::model.layers.\1.mlp.experts.1.up_proj.weight",
                }
            },
            {
                "copy": {
                    "from": r"m1::model.layers\.(\d+)\.mlp\.down_proj\.weight",
                    "to": r"m0::model.layers.\1.mlp.experts.1.down_proj.weight",
                }
            },
            {
                "fill": {
                    "from": r"m0::model.layers\.(\d+)\.self_attn\.q_proj\.weight::[:2, :]",
                    "to": r"m0::model.layers.\1.mlp.gate.weight",
                    "mode": "constant",
                    "value": 0,
                }
            },
            {"delete": {"target": r"m0::model.layers\.(\d+)\.mlp\.gate_proj\.weight"}},
            {"delete": {"target": r"m0::model.layers\.(\d+)\.mlp\.up_proj\.weight"}},
            {"delete": {"target": r"m0::model.layers\.(\d+)\.mlp\.down_proj\.weight"}},
            {
                "assert": {
                    "shape": {
                        "of": r"m0::model.layers\.(\d+)\.mlp\.gate\.weight",
                        "is": [2, 4],
                    }
                }
            },
            {"assert": {"not": {"exists": r"m0::model.layers\.0\.mlp\.gate_proj\.weight"}}},
            {"assert": {"not": {"exists": r"m0::model.layers\.1\.mlp\.down_proj\.weight"}}},
        ],
    }

    plan = compile_plan(raw)
    dense = _small_dense_state_dict()
    provider = _Provider(
        {
            "m0": _make_state_dict(dense),
            "m1": _make_state_dict(dense),
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])

    out = provider.get_state_dict("m0")
    assert torch.equal(out["model.embed_tokens.weight"], dense["model.embed_tokens.weight"])
    assert torch.equal(out["lm_head.weight"], dense["lm_head.weight"])

    for layer in range(2):
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
