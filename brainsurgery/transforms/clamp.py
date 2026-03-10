from __future__ import annotations

from dataclasses import dataclass

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import StateDictProvider, TransformError, ensure_mapping_payload, register_transform, require_numeric, validate_payload_keys


class ClampTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ClampSpec(BinaryMappingSpec):
    min_value: float | None
    max_value: float | None


class ClampTransform(BinaryMappingTransform[ClampSpec]):
    name = "clamp"
    error_type = ClampTransformError
    spec_type = ClampSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "min", "max"}
    required_keys = {"from", "to"}
    progress_desc = "Applying clamp transforms"
    help_text = (
        "Clamps source tensors into new destination tensors.\n"
        "\n"
        "Source references may be sliced. Destination tensors must not exist and "
        "must not be sliced. At least one of 'min' or 'max' is required.\n"
        "\n"
        "Example:\n"
        "  clamp: { from: x, to: x_clamped, min: -1.0, max: 1.0 }"
    )

    def compile(self, payload: dict, default_model: str | None) -> ClampSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)
        min_value, max_value = _parse_bounds(payload, self.name, ClampTransformError)
        return ClampSpec(from_ref=from_ref, to_ref=to_ref, min_value=min_value, max_value=max_value)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise ClampTransformError("clamp destination must not be sliced")

    def apply_item(self, spec: ClampSpec, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)
        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        dst_sd[item.dst_name] = src_view.clamp(min=spec.min_value, max=spec.max_value).clone()


def _parse_bounds(
    payload: dict,
    op_name: str,
    error_type: type[TransformError],
) -> tuple[float | None, float | None]:
    min_value = require_numeric(payload, op_name=op_name, key="min") if "min" in payload else None
    max_value = require_numeric(payload, op_name=op_name, key="max") if "max" in payload else None
    if min_value is None and max_value is None:
        raise error_type(f"{op_name} requires at least one of: min, max")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise error_type(f"{op_name}.min must be <= {op_name}.max")
    return min_value, max_value








register_transform(ClampTransform())
