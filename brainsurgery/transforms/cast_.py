from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dtypes import parse_torch_dtype
from .unary import UnarySpec, UnaryTransform
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    register_transform,
    require_nonempty_string,
)


class CastInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class CastInPlaceSpec(UnarySpec):
    dtype: torch.dtype


class CastInPlaceTransform(UnaryTransform[CastInPlaceSpec]):
    name = "cast_"
    error_type = CastInPlaceTransformError
    spec_type = CastInPlaceSpec
    allowed_keys = {"target", "to"}
    required_keys = {"target", "to"}
    progress_desc = "Applying cast_ transforms"
    help_text = (
        "Casts one or more tensors to a different dtype in-place.\n"
        "\n"
        "The 'target' selects tensors by name or pattern. The entire tensor is cast; "
        "slicing is not supported.\n"
        "\n"
        "Examples:\n"
        "  cast_: { target: ln_f.weight, to: float16 }\n"
        "  cast_: { target: '.*weight', to: bfloat16 }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> CastInPlaceSpec:
        raw_dtype = require_nonempty_string(payload, op_name=self.name, key="to")
        dtype = parse_torch_dtype(
            raw_dtype,
            error_type=CastInPlaceTransformError,
            op_name=self.name,
            field_name="to",
        )
        return CastInPlaceSpec(target_ref=target_ref, dtype=dtype)

    def apply_to_target(self, spec: CastInPlaceSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        sd[name] = sd[name].to(dtype=spec.dtype)


def _unit_test_cast_in_place_compile_rejects_unknown_dtype() -> None:
    try:
        CastInPlaceTransform().compile({"target": "x", "to": "not_a_dtype"}, default_model="model")
    except CastInPlaceTransformError as exc:
        assert "cast_.to" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected cast_ dtype parse error")


def _unit_test_cast_in_place_compile_requires_to_key() -> None:
    try:
        CastInPlaceTransform().compile({"target": "x"}, default_model="model")
    except TransformError as exc:
        assert "cast_.to is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing to key error")


def _unit_test_cast_in_place_apply_changes_dtype() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.ones((2,), dtype=torch.float32)}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = CastInPlaceSpec(target_ref=TensorRef(model="model", expr="x"), dtype=torch.float16)
    CastInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].dtype == torch.float16


__unit_tests__ = [
    _unit_test_cast_in_place_compile_rejects_unknown_dtype,
    _unit_test_cast_in_place_compile_requires_to_key,
    _unit_test_cast_in_place_apply_changes_dtype,
]


register_transform(CastInPlaceTransform())
