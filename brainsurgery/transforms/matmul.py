import torch

from ..core import (
    DestinationPolicy,
    TernaryMappingSpec,
    must_model,
    parse_slice,
)
from ..core import select_tensor
from ..core import StateDictProvider, TransformError
from ..core import register_transform
from ..core import DeclarativeTernaryTransform, Docs, TernaryRefs
from ..engine import emit_verbose_ternary_activity


def _matmul_apply(
    spec: TernaryMappingSpec,
    src_a_name: str,
    src_b_name: str,
    dst_name: str,
    provider: StateDictProvider,
) -> None:
    src_a_sd = provider.get_state_dict(must_model(spec.from_a_ref))
    src_b_sd = provider.get_state_dict(must_model(spec.from_b_ref))
    dst_sd = provider.get_state_dict(must_model(spec.to_ref))
    src_a_slice = parse_slice(spec.from_a_ref.slice_spec) if spec.from_a_ref.slice_spec is not None else None
    src_b_slice = parse_slice(spec.from_b_ref.slice_spec) if spec.from_b_ref.slice_spec is not None else None

    src_a_view = select_tensor(src_a_sd[src_a_name], src_a_slice)
    src_b_view = select_tensor(src_b_sd[src_b_name], src_b_slice)

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
        raise TransformError(
            f"shape mismatch matmul {src_a_name} @ {src_b_name}: {exc}"
        ) from exc

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
