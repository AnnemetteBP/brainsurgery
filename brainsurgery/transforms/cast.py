from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dtypes import parse_torch_dtype
from ..targeting import resolve_target_names
from .unary import UnarySpec, UnaryTransform
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    register_transform,
    require_nonempty_string,
)


class CastTransformError(TransformError):
    pass


@dataclass(frozen=True)
class CastSpec(UnarySpec):
    dtype: torch.dtype


class CastTransform(UnaryTransform[CastSpec]):
    name = "cast"
    error_type = CastTransformError
    spec_type = CastSpec
    allowed_keys = {"target", "to"}
    required_keys = {"target", "to"}
    progress_desc = "Applying cast transforms"

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            raise CastTransformError("cast does not support tensor slices; cast the whole tensor")

    def build_spec(self, target_ref: TensorRef, payload: dict) -> CastSpec:
        raw_dtype = require_nonempty_string(payload, op_name=self.name, key="to")
        dtype = parse_torch_dtype(
            raw_dtype,
            error_type=CastTransformError,
            op_name=self.name,
            field_name="to",
        )
        return CastSpec(target_ref=target_ref, dtype=dtype)

    def resolve_targets(self, spec: CastSpec, provider: StateDictProvider) -> list[str]:
        return resolve_target_names(
            target_ref=spec.target_ref,
            provider=provider,
            op_name=self.name,
            error_type=CastTransformError,
        )

    def apply_to_target(self, spec: CastSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        sd[name] = sd[name].to(dtype=spec.dtype)


register_transform(CastTransform())
