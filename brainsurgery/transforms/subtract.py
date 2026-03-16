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


def _subtract_apply(
    spec: TernaryMappingSpec,
    src_a_name: str,
    src_b_name: str,
    dst_name: str,
    provider: StateDictProvider,
) -> None:
    _src_a_sd, _src_b_sd, dst_sd, src_a_view, src_b_view, dst_view = ternary_mapping_views(
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
        op_name="subtracting",
        first_name=src_a_name,
        second_name=src_b_name,
        dest_name=dst_name,
        symbol="-",
    )
    dst_view.copy_(src_a_view - src_b_view)
    dst_sd.mark_write(dst_name)
    emit_verbose_ternary_activity("subtract", src_a_name, src_b_name, dst_name)


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
        op_name="subtracting",
        left_name=src_name,
        right_name=dst_name,
    )

    dst_view.sub_(src_view)
    dst_sd.mark_write(dst_name)
    emit_verbose_binary_activity("subtract_", src_name, dst_name)


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
