from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import (
    StateDictProvider,
    TransformError,
    ensure_mapping_payload,
    register_transform,
    require_numeric,
    validate_payload_keys,
)
from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy


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


register_transform(ScaleTransform())
