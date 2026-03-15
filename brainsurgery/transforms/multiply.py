from ..core import (
    DestinationPolicy,
    TernaryMappingSpec,
    must_model,
    parse_slice,
)
from ..core import select_tensor
from ..core import require_same_shape_dtype_device3
from ..core import StateDictProvider, TransformError
from ..core import register_transform
from ..core import DeclarativeTernaryTransform, Docs, TernaryRefs
from ..engine import emit_verbose_ternary_activity


def _multiply_apply(
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
    dst_slice = parse_slice(spec.to_ref.slice_spec) if spec.to_ref.slice_spec is not None else None

    src_a_view = select_tensor(src_a_sd[src_a_name], src_a_slice)
    src_b_view = select_tensor(src_b_sd[src_b_name], src_b_slice)
    dst_view = select_tensor(dst_sd[dst_name], dst_slice)

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
