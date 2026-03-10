from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dtypes import parse_torch_dtype
from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import StateDictProvider, TransformError, ensure_mapping_payload, register_transform, require_nonempty_string, validate_payload_keys


class CastTransformError(TransformError):
    pass


@dataclass(frozen=True)
class CastSpec(BinaryMappingSpec):
    dtype: torch.dtype


class CastTransform(BinaryMappingTransform[CastSpec]):
    name = "cast"
    error_type = CastTransformError
    spec_type = CastSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "dtype"}
    required_keys = {"from", "to", "dtype"}
    progress_desc = "Applying cast transforms"
    help_text = (
        "Casts one or more source tensors to a different dtype and writes to new destinations.\n"
        "\n"
        "The source ('from') may be specified by name or pattern and may include slicing. "
        "Destination tensors ('to') must not already exist and must not be sliced.\n"
        "\n"
        "Examples:\n"
        "  cast: { from: ln_f.weight, to: ln_f_fp16.weight, dtype: float16 }\n"
        "  cast: { from: '.*weight', to: 'fp16.\\\\g<0>', dtype: bfloat16 }"
    )

    def compile(self, payload: dict, default_model: str | None) -> CastSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)

        raw_dtype = require_nonempty_string(payload, op_name=self.name, key="dtype")
        dtype = parse_torch_dtype(
            raw_dtype,
            error_type=CastTransformError,
            op_name=self.name,
            field_name="dtype",
        )
        return CastSpec(from_ref=from_ref, to_ref=to_ref, dtype=dtype)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise CastTransformError("cast destination must not be sliced")

    def apply_item(self, spec: CastSpec, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)
        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        dst_sd[item.dst_name] = src_view.to(dtype=spec.dtype)


def _unit_test_cast_compile_rejects_unknown_dtype() -> None:
    try:
        CastTransform().compile(
            {"from": "x", "to": "y", "dtype": "not_a_dtype"},
            default_model="model",
        )
    except CastTransformError as exc:
        assert "cast.dtype" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected cast dtype parse error")


def _unit_test_cast_compile_requires_dtype_key() -> None:
    try:
        CastTransform().compile({"from": "x", "to": "y"}, default_model="model")
    except TransformError as exc:
        assert "cast.dtype is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing dtype key error")


def _unit_test_cast_compile_rejects_sliced_destination() -> None:
    try:
        CastTransform().compile(
            {"from": "x", "to": "y::[:]", "dtype": "float16"},
            default_model="model",
        )
    except CastTransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced destination error")


def _unit_test_cast_apply_creates_new_tensor_with_new_dtype() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.ones((2,), dtype=torch.float32)}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = CastTransform().compile(
        {"from": "x", "to": "y", "dtype": "float16"},
        default_model="model",
    )
    CastTransform().apply(spec, provider)
    assert provider._state_dict["x"].dtype == torch.float32
    assert provider._state_dict["y"].dtype == torch.float16


def _unit_test_cast_apply_honors_source_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = CastTransform().compile(
        {"from": "x::[1:3]", "to": "y", "dtype": "float16"},
        default_model="model",
    )
    CastTransform().apply(spec, provider)
    assert provider._state_dict["y"].tolist() == [2.0, 3.0]
    assert provider._state_dict["y"].dtype == torch.float16


__unit_tests__ = [
    _unit_test_cast_compile_rejects_unknown_dtype,
    _unit_test_cast_compile_requires_dtype_key,
    _unit_test_cast_compile_rejects_sliced_destination,
    _unit_test_cast_apply_creates_new_tensor_with_new_dtype,
    _unit_test_cast_apply_honors_source_slice,
]


register_transform(CastTransform())
