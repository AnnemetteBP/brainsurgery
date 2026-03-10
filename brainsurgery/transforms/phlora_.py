from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from ..refs import TensorRef, must_model
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
    require_numeric,
)


class PhloraInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class PhloraInPlaceSpec(UnarySpec):
    rank: int


class PhloraInPlaceTransform(UnaryTransform[PhloraInPlaceSpec]):
    name = "phlora_"
    error_type = PhloraInPlaceTransformError
    spec_type = PhloraInPlaceSpec
    allowed_keys = {"target", "rank"}
    required_keys = {"target", "rank"}
    progress_desc = "Applying phlora_ transforms"
    help_text = (
        "Applies in-place PHLoRA low-rank reconstruction.\n"
        "\n"
        "For each matched 2D tensor W:\n"
        "  u, s, vh = svd(W)\n"
        "  W <- (u[:, :r] * s[:r]) @ vh[:r, :]\n"
        "\n"
        "Examples:\n"
        "  phlora_: { target: '.*weight', rank: 64 }\n"
        "  phlora_: { target: h.0.attn.c_proj.weight, rank: 16 }"
    )

    def __init__(self) -> None:
        super().__init__()
        self._svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def build_spec(self, target_ref: TensorRef, payload: dict) -> PhloraInPlaceSpec:
        rank = _require_positive_int(payload, op_name=self.name, key="rank")
        return PhloraInPlaceSpec(target_ref=target_ref, rank=rank)

    def apply_to_target(
        self,
        spec: PhloraInPlaceSpec,
        name: str,
        provider: StateDictProvider,
    ) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        source = sd[name]
        if source.ndim != 2:
            raise PhloraInPlaceTransformError(
                f"phlora_ target must be 2D (got shape {tuple(source.shape)}): {model}::{name}"
            )

        rank = min(spec.rank, min(source.shape))
        if rank <= 0:
            raise PhloraInPlaceTransformError(
                f"phlora_ rank became zero for {model}::{name} with shape {tuple(source.shape)}"
            )

        u, s, vh = self._get_svd(source, cache_key=f"{model}::{name}")
        new_tensor = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        sd[name] = new_tensor.to(dtype=source.dtype, device=source.device)

    def _get_svd(
        self,
        source: torch.Tensor,
        *,
        cache_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_key = f"{cache_key}|{tuple(source.shape)}|{source.dtype}|{source.device}"
        if full_key not in self._svd_cache:
            self._svd_cache[full_key] = torch.linalg.svd(source, full_matrices=False)
        return self._svd_cache[full_key]


def _require_positive_int(payload: dict, *, op_name: str, key: str) -> int:
    numeric = require_numeric(payload, op_name=op_name, key=key)
    integer = int(numeric)
    if float(integer) != float(numeric) or integer <= 0:
        raise PhloraInPlaceTransformError(f"{op_name}.{key} must be a positive integer")
    return integer


def _unit_test_phlora_in_place_compile_rejects_non_integral_rank() -> None:
    try:
        PhloraInPlaceTransform().compile({"target": "x", "rank": 3.5}, default_model="model")
    except PhloraInPlaceTransformError as exc:
        assert "positive integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected rank validation error")


def _unit_test_phlora_in_place_rewrites_target_with_ranked_matrix() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "w": torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraInPlaceTransform()
    spec = transform.compile({"target": "w", "rank": 1}, default_model="model")
    transform.apply(spec, provider)

    assert "w" in provider._state_dict
    assert provider._state_dict["w"].shape == (2, 2)
    assert torch.allclose(provider._state_dict["w"], torch.tensor([[3.0, 0.0], [0.0, 0.0]]))


def _unit_test_phlora_in_place_rejects_non_2d_target() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraInPlaceTransform()
    spec = transform.compile({"target": "x", "rank": 1}, default_model="model")
    try:
        transform.apply(spec, provider)
    except PhloraInPlaceTransformError as exc:
        assert "must be 2D" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected 2D validation error")


__unit_tests__ = [
    _unit_test_phlora_in_place_compile_rejects_non_integral_rank,
    _unit_test_phlora_in_place_rewrites_target_with_ranked_matrix,
    _unit_test_phlora_in_place_rejects_non_2d_target,
]


register_transform(PhloraInPlaceTransform())
