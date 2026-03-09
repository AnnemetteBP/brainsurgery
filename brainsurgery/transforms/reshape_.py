from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from .reshape import _parse_shape
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    register_transform,
)


class ReshapeInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ReshapeInPlaceSpec(UnarySpec):
    shape: tuple[int, ...]


class ReshapeInPlaceTransform(UnaryTransform[ReshapeInPlaceSpec]):
    name = "reshape_"
    error_type = ReshapeInPlaceTransformError
    spec_type = ReshapeInPlaceSpec
    allowed_keys = {"target", "shape"}
    required_keys = {"target", "shape"}
    progress_desc = "Applying reshape_ transforms"
    help_text = (
        "Reshapes target tensors in-place (rebinds the tensor at the same name).\n"
        "\n"
        "Slicing is not supported for reshape_. Shape may include one '-1'.\n"
        "\n"
        "Example:\n"
        "  reshape_: { target: x, shape: [1024, -1] }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ReshapeInPlaceSpec:
        shape = _parse_shape(payload.get("shape"), op_name=self.name, error_type=ReshapeInPlaceTransformError)
        return ReshapeInPlaceSpec(target_ref=target_ref, shape=shape)

    def apply_to_target(self, spec: ReshapeInPlaceSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        try:
            sd[name] = sd[name].reshape(spec.shape).clone()
        except RuntimeError as exc:
            raise ReshapeInPlaceTransformError(
                f"reshape_ failed for {model}::{name}: {exc}"
            ) from exc


def _unit_test_reshape_in_place_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.arange(6)}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = ReshapeInPlaceSpec(target_ref=TensorRef(model="m", expr="x"), shape=(2, 3))
    ReshapeInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].shape == (2, 3)


__unit_tests__ = [
    _unit_test_reshape_in_place_apply_success,
]


register_transform(ReshapeInPlaceTransform())
