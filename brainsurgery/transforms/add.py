from __future__ import annotations

from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DestinationPolicy,
    Docs,
    ResolvedTernaryMapping,
    ResolvedMapping,
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


def _add_apply(
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
        op_name="adding",
        first_name=item.src_a_name,
        second_name=item.src_b_name,
        dest_name=item.dst_name,
        symbol="+",
    )
    dst_view.copy_(src_a_view + src_b_view)
    dst_sd.mark_write(item.dst_name)
    emit_verbose_ternary_activity("add", item)


class AddTransform(DeclarativeTernaryTransform[TernaryMappingSpec]):
    name = "add"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_EXIST
    docs = Docs(
        "Computes elementwise addition from 'from_a' and 'from_b' into 'to'.",
        examples=(
            "add: { from_a: a.weight, from_b: b.weight, to: out.weight }",
            "add: { from_a: '.*.weight', from_b: '.*.delta', to: '.*.weight' }",
        ),
    )
    refs = TernaryRefs(from_a_slice=True, from_b_slice=True, to_slice=True)
    apply_fn = staticmethod(_add_apply)


register_transform(AddTransform())


def _add_in_place_apply(
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
        op_name="adding",
        left_name=item.src_name,
        right_name=item.dst_name,
    )

    dst_view.add_(src_view)
    dst_sd.mark_write(item.dst_name)
    emit_verbose_binary_activity("add_", item)


class AddInPlaceTransform(DeclarativeBinaryTransform[BinaryMappingSpec]):
    name = "add_"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_EXIST
    docs = Docs(
        "Adds source tensors into destination tensors in-place.",
        notes=("Computes: to <- to + from.",),
        examples=(
            "add_: { from: delta.weight, to: model.weight }",
            "add_: { from: 'a::[:, :10]', to: 'b::[:, :10]' }",
        ),
    )
    refs = BinaryRefs(from_slice=True, to_slice=True)
    apply_fn = staticmethod(_add_in_place_apply)


register_transform(AddInPlaceTransform())
