from __future__ import annotations

from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DestinationPolicy,
    Docs,
    ResolvedMapping,
    ResolvedTernaryMapping,
    StateDictProvider,
    TernaryMappingSpec,
    TransformError,
    require_same_shape_dtype_device,
    select_tensor,
)
from ..core import require_same_shape_dtype_device3
from ..core import register_transform
from ..core import DeclarativeTernaryTransform, TernaryRefs
from ..engine import emit_verbose_binary_activity
from ..engine import emit_verbose_ternary_activity


def _subtract_apply(
    _spec: TernaryMappingSpec,
    item: ResolvedTernaryMapping,
    provider: StateDictProvider,
) -> None:
    src_a_sd = provider.get_state_dict(item.src_a_model)
    src_b_sd = provider.get_state_dict(item.src_b_model)
    dst_sd = provider.get_state_dict(item.dst_model)

    src_a_view = select_tensor(src_a_sd[item.src_a_name], item.src_a_slice)
    src_b_view = select_tensor(src_b_sd[item.src_b_name], item.src_b_slice)
    dst_view = select_tensor(dst_sd[item.dst_name], item.dst_slice)

    require_same_shape_dtype_device3(
        src_a_view,
        src_b_view,
        dst_view,
        op_name="subtracting",
        first_name=item.src_a_name,
        second_name=item.src_b_name,
        dest_name=item.dst_name,
        symbol="-",
    )
    dst_view.copy_(src_a_view - src_b_view)
    dst_sd.mark_write(item.dst_name)
    emit_verbose_ternary_activity("subtract", item)


class SubtractTransform(DeclarativeTernaryTransform[TernaryMappingSpec]):
    name = "subtract"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_EXIST
    docs = Docs(
        "Computes elementwise subtraction from 'from_a' minus 'from_b' into 'to'.",
        examples=("subtract: { from_a: a.weight, from_b: b.weight, to: out.weight }",),
    )
    refs = TernaryRefs(from_a_slice=True, from_b_slice=True, to_slice=True)
    apply_fn = staticmethod(_subtract_apply)


register_transform(SubtractTransform())


def _subtract_in_place_apply(
    _spec: BinaryMappingSpec,
    item: ResolvedMapping,
    provider: StateDictProvider,
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)

    src_view = select_tensor(src_sd[item.src_name], item.src_slice)
    dst_view = select_tensor(dst_sd[item.dst_name], item.dst_slice)
    require_same_shape_dtype_device(
        src_view,
        dst_view,
        op_name="subtracting",
        left_name=item.src_name,
        right_name=item.dst_name,
    )

    dst_view.sub_(src_view)
    dst_sd.mark_write(item.dst_name)
    emit_verbose_binary_activity("subtract_", item)


class SubtractInPlaceTransform(DeclarativeBinaryTransform[BinaryMappingSpec]):
    name = "subtract_"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_EXIST
    docs = Docs(
        "Subtracts source tensors from destination tensors in-place.",
        notes=("Computes: to <- to - from.",),
        examples=(
            "subtract_: { from: delta.weight, to: model.weight }",
            "subtract_: { from: 'a::[:, :10]', to: 'b::[:, :10]' }",
        ),
    )
    refs = BinaryRefs(from_slice=True, to_slice=True)
    apply_fn = staticmethod(_subtract_in_place_apply)


register_transform(SubtractInPlaceTransform())
