from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils import parse_torch_dtype
from ..core import TensorRef, must_model
from .unary import UnarySpec
from ..core import (
    register_transform,
    require_nonempty_string,
)
from ..core import StateDictProvider, TransformError
from ..utils.transforms import DeclarativeUnaryTransform, Docs, UnaryRefs


@dataclass(frozen=True)
class CastInPlaceSpec(UnarySpec):
    dtype: torch.dtype


def _build_cast_in_place_spec(target_ref: TensorRef, payload: dict) -> CastInPlaceSpec:
    raw_dtype = require_nonempty_string(payload, op_name="cast_", key="to")
    dtype = parse_torch_dtype(
        raw_dtype,
        error_type=TransformError,
        op_name="cast_",
        field_name="to",
    )
    return CastInPlaceSpec(target_ref=target_ref, dtype=dtype)


def _cast_in_place_apply(
    spec: CastInPlaceSpec, name: str, provider: StateDictProvider
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    sd[name] = sd[name].to(dtype=spec.dtype)


class CastInPlaceTransform(DeclarativeUnaryTransform[CastInPlaceSpec]):
    name = "cast_"
    error_type = TransformError
    spec_type = CastInPlaceSpec
    allowed_keys = {"target", "to"}
    required_keys = {"target", "to"}
    docs = Docs(
        "Casts one or more tensors to a different dtype in-place.",
        notes=("The entire tensor is cast.",),
        examples=(
            "cast_: { target: ln_f.weight, to: float16 }",
            "cast_: { target: '.*weight', to: bfloat16 }",
        ),
    )
    refs = UnaryRefs()
    spec_builder = staticmethod(_build_cast_in_place_spec)
    apply_fn = staticmethod(_cast_in_place_apply)


register_transform(CastInPlaceTransform())
