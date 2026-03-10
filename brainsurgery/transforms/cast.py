from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dtypes import parse_torch_dtype
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import (
    ensure_mapping_payload,
    register_transform,
    require_nonempty_string,
    validate_payload_keys,
)
from ..transform_types import StateDictProvider, TransformError
from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy


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


register_transform(CastTransform())
