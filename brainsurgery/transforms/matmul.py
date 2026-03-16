import torch

from ..core import (
    DeclarativeTernaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TernaryMappingSpec,
    TernaryRefs,
    TransformError,
    register_transform,
    state_dict_for_ref,
    view_for_ref_name,
)
from ..engine import emit_verbose_ternary_activity


def _matmul_apply(
    spec: TernaryMappingSpec,
    src_a_name: str,
    src_b_name: str,
    dst_name: str,
    provider: StateDictProvider,
) -> None:
    _src_a_sd, src_a_view = view_for_ref_name(provider, spec.from_a_ref, src_a_name)
    _src_b_sd, src_b_view = view_for_ref_name(provider, spec.from_b_ref, src_b_name)
    dst_sd = state_dict_for_ref(provider, spec.to_ref)

    if src_a_view.dtype != src_b_view.dtype:
        raise TransformError(
            f"dtype mismatch matmul {src_a_name} @ {src_b_name}: "
            f"{src_a_view.dtype} != {src_b_view.dtype}"
        )
    if src_a_view.device != src_b_view.device:
        raise TransformError(
            f"device mismatch matmul {src_a_name} @ {src_b_name}: "
            f"{src_a_view.device} != {src_b_view.device}"
        )

    try:
        result = torch.matmul(src_a_view, src_b_view)
    except RuntimeError as exc:
        raise TransformError(f"shape mismatch matmul {src_a_name} @ {src_b_name}: {exc}") from exc

    dst_sd[dst_name] = result.clone()
    emit_verbose_ternary_activity("matmul", src_a_name, src_b_name, dst_name)


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
