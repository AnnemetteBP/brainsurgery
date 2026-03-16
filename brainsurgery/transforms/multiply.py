from ..core import (
    DeclarativeTernaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TernaryMappingSpec,
    TernaryRefs,
    TransformError,
    register_transform,
    require_same_shape_dtype_device3,
    ternary_mapping_views,
)
from ..engine import emit_verbose_ternary_activity


def _multiply_apply(
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
        op_name="multiplying",
        first_name=src_a_name,
        second_name=src_b_name,
        dest_name=dst_name,
        symbol="*",
    )
    dst_view.copy_(src_a_view * src_b_view)
    dst_sd.mark_write(dst_name)
    emit_verbose_ternary_activity("multiply", src_a_name, src_b_name, dst_name)


class MultiplyTransform(DeclarativeTernaryTransform[TernaryMappingSpec]):
    name = "multiply"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_EXIST
    docs = Docs(
        "Computes elementwise multiplication from 'from_a' and 'from_b' into 'to'.",
        examples=("multiply: { from_a: a.weight, from_b: b.weight, to: out.weight }",),
    )
    refs = TernaryRefs(from_a_slice=True, from_b_slice=True, to_slice=True)
    apply_fn = staticmethod(_multiply_apply)


register_transform(MultiplyTransform())
