from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DeclarativeTernaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TernaryMappingSpec,
    TernaryRefs,
    TransformError,
    binary_mapping_views,
    register_transform,
    require_same_shape_dtype_device,
    require_same_shape_dtype_device3,
    ternary_mapping_views,
)
from ..engine import emit_verbose_binary_activity, emit_verbose_ternary_activity


def _add_apply(
    spec: TernaryMappingSpec,
    src_a_name: str,
    src_b_name: str,
    dst_name: str,
    provider: StateDictProvider,
) -> None:
    src_a_sd, _src_b_sd, dst_sd, src_a_view, src_b_view, dst_view = ternary_mapping_views(
        provider,
        from_a_ref=spec.from_a_ref,
        from_b_ref=spec.from_b_ref,
        to_ref=spec.to_ref,
        src_a_name=src_a_name,
        src_b_name=src_b_name,
        dst_name=dst_name,
    )

    require_same_shape_dtype_device3(
        src_a_view,
        src_b_view,
        dst_view,
        op_name="adding",
        first_name=src_a_name,
        second_name=src_b_name,
        dest_name=dst_name,
        symbol="+",
    )
    dst_view.copy_(src_a_view + src_b_view)
    dst_sd.mark_write(dst_name)
    emit_verbose_ternary_activity("add", src_a_name, src_b_name, dst_name)


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
    spec: BinaryMappingSpec,
    src_name: str,
    dst_name: str,
    provider: StateDictProvider,
) -> None:
    _src_sd, dst_sd, src_view, dst_view = binary_mapping_views(
        provider,
        from_ref=spec.from_ref,
        to_ref=spec.to_ref,
        src_name=src_name,
        dst_name=dst_name,
    )
    require_same_shape_dtype_device(
        src_view,
        dst_view,
        op_name="adding",
        left_name=src_name,
        right_name=dst_name,
    )

    dst_view.add_(src_view)
    dst_sd.mark_write(dst_name)
    emit_verbose_binary_activity("add_", src_name, dst_name)


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
