from __future__ import annotations

from dataclasses import dataclass

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    parse_slice,
    register_transform,
    require_numeric,
    select_tensor,
    validate_payload_keys,
)


class ScaleTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ScaleSpec(BinaryMappingSpec):
    factor: float


class ScaleTransform(BinaryMappingTransform[ScaleSpec]):
    name = "scale"
    error_type = ScaleTransformError
    spec_type = ScaleSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "by"}
    required_keys = {"from", "to", "by"}
    progress_desc = "Applying scale transforms"
    help_text = (
        "Scales source tensors by a numeric factor into new destination tensors.\n"
        "\n"
        "Source references may be specified by name or pattern and may include slicing "
        "(written after '::'). Destination tensors must not already exist and must not "
        "be sliced.\n"
        "\n"
        "Examples:\n"
        "  scale: { from: ln_f.weight, to: ln_f_half.weight, by: 0.5 }\n"
        "  scale: { from: '.*bias', to: 'scaled.\\\\g<0>', by: -1 }\n"
        "  scale: { from: 'h.0.attn.c_attn.weight::[:, :10]', to: h.0.attn.c_attn.partial, by: 2.0 }"
    )

    def compile(self, payload: dict, default_model: str | None) -> ScaleSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)
        factor = require_numeric(payload, op_name=self.name, key="by")
        return ScaleSpec(from_ref=from_ref, to_ref=to_ref, factor=factor)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise ScaleTransformError("scale destination must not be sliced")

    def apply_item(self, spec: ScaleSpec, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)
        scaled = select_tensor(src_sd[item.src_name], item.src_slice).clone()
        scaled.mul_(spec.factor)
        dst_sd[item.dst_name] = scaled


def _unit_test_scale_compile_rejects_non_numeric_factor() -> None:
    try:
        ScaleTransform().compile({"from": "x", "to": "y", "by": "nan?!"}, default_model="model")
    except TransformError as exc:
        assert "scale.by must be numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected scale numeric validation error")


def _unit_test_scale_compile_accepts_numeric_string_factor() -> None:
    spec = ScaleTransform().compile({"from": "x", "to": "y", "by": "2.5"}, default_model="model")
    assert spec.factor == 2.5


def _unit_test_scale_compile_rejects_sliced_destination() -> None:
    try:
        ScaleTransform().compile(
            {"from": "x", "to": "y::[:]", "by": 1.0},
            default_model="model",
        )
    except ScaleTransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced destination rejection")


def _unit_test_scale_apply_creates_scaled_tensor_from_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0, 4.0]), "z": torch.tensor([0.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = ScaleTransform().compile(
        {"from": "x::[1:3]", "to": "y", "by": 10.0},
        default_model="model",
    )
    ScaleTransform().apply(spec, provider)
    assert provider._state_dict["x"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert provider._state_dict["y"].tolist() == [20.0, 30.0]


def _unit_test_scale_apply_rejects_existing_destination() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "x": torch.tensor([1.0, 2.0]),
                "y": torch.tensor([0.0, 0.0]),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = ScaleTransform().compile(
        {"from": "x", "to": "y", "by": 2.0},
        default_model="model",
    )
    try:
        ScaleTransform().apply(spec, provider)
    except TransformError as exc:
        assert "destination already exists" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected destination-already-exists error")


__unit_tests__ = [
    _unit_test_scale_compile_rejects_non_numeric_factor,
    _unit_test_scale_compile_accepts_numeric_string_factor,
    _unit_test_scale_compile_rejects_sliced_destination,
    _unit_test_scale_apply_creates_scaled_tensor_from_slice,
    _unit_test_scale_apply_rejects_existing_destination,
]


register_transform(ScaleTransform())
