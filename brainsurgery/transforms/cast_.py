from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dtypes import parse_torch_dtype
from ..refs import TensorRef, must_model
from .unary import UnarySpec, UnaryTransform
from ..transform import (
    StateDictProvider,
    TransformError,
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










register_transform(CastInPlaceTransform())
