from __future__ import annotations

from dataclasses import dataclass

from .binary import BinaryMappingSpec, DestinationPolicy
import torch
from ..core import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    register_transform,
    require_nonempty_string,
    select_tensor,
)
from ..utils import (
    BinaryRefs,
    DeclarativeBinaryTransform,
    Docs,
    parse_torch_dtype,
)


@dataclass(frozen=True)
class CastSpec(BinaryMappingSpec):
    dtype: torch.dtype


def _build_cast_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> CastSpec:
    raw_dtype = require_nonempty_string(payload, op_name="cast", key="dtype")
    dtype = parse_torch_dtype(
        raw_dtype,
        error_type=TransformError,
        op_name="cast",
        field_name="dtype",
    )
    return CastSpec(from_ref=from_ref, to_ref=to_ref, dtype=dtype)


def _cast_apply(
    spec: CastSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)
    src_view = select_tensor(src_sd[item.src_name], item.src_slice)
    dst_sd[item.dst_name] = src_view.to(dtype=spec.dtype)


class CastTransform(DeclarativeBinaryTransform[CastSpec]):
    name = "cast"
    error_type = TransformError
    spec_type = CastSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "dtype"}
    required_keys = {"from", "to", "dtype"}
    docs = Docs(
        "Casts source tensors to a different dtype and writes new destination tensors.",
        examples=(
            "cast: { from: ln_f.weight, to: ln_f_fp16.weight, dtype: float16 }",
            "cast: { from: '.*weight', to: 'fp16.\\\\g<0>', dtype: bfloat16 }",
        ),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_cast_spec)
    apply_fn = staticmethod(_cast_apply)


register_transform(CastTransform())
