from __future__ import annotations

import torch

from ..core import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
)
from ..core import select_tensor
from ..core import StateDictProvider, TransformError
from ..core import register_transform
from ..core import DeclarativeTernaryTransform, Docs, TernaryRefs


def _matmul_apply(
    _spec: TernaryMappingSpec,
    item: ResolvedTernaryMapping,
    provider: StateDictProvider,
) -> None:
    src_a_sd = provider.get_state_dict(item.src_a_model)
    src_b_sd = provider.get_state_dict(item.src_b_model)
    dst_sd = provider.get_state_dict(item.dst_model)

    src_a_view = select_tensor(src_a_sd[item.src_a_name], item.src_a_slice)
    src_b_view = select_tensor(src_b_sd[item.src_b_name], item.src_b_slice)

    if src_a_view.dtype != src_b_view.dtype:
        raise TransformError(
            f"dtype mismatch matmul {item.src_a_name} @ {item.src_b_name}: "
            f"{src_a_view.dtype} != {src_b_view.dtype}"
        )
    if src_a_view.device != src_b_view.device:
        raise TransformError(
            f"device mismatch matmul {item.src_a_name} @ {item.src_b_name}: "
            f"{src_a_view.device} != {src_b_view.device}"
        )

    try:
        result = torch.matmul(src_a_view, src_b_view)
    except RuntimeError as exc:
        raise TransformError(
            f"shape mismatch matmul {item.src_a_name} @ {item.src_b_name}: {exc}"
        ) from exc

    dst_sd[item.dst_name] = result.clone()


class MatmulTransform(DeclarativeTernaryTransform[TernaryMappingSpec]):
    name = "matmul"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    docs = Docs(
        "Computes matrix multiplication from 'from_a' and 'from_b' into 'to'.",
        examples=("matmul: { from_a: a.weight, from_b: b.weight, to: out.weight }",),
    )
    refs = TernaryRefs(from_a_slice=True, from_b_slice=True)
    apply_fn = staticmethod(_matmul_apply)


register_transform(MatmulTransform())
