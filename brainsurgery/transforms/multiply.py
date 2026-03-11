from __future__ import annotations

from .ternary import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
)
from ..core import select_tensor
from ..utils import require_same_shape_dtype_device3
from ..core import (
    StateDictProvider,
    TransformError,
    register_transform,
)
from ..core import note_tensor_write
from ..utils import DeclarativeTernaryTransform, Docs, TernaryRefs


def _multiply_apply(
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
        op_name="multiplying",
        first_name=item.src_a_name,
        second_name=item.src_b_name,
        dest_name=item.dst_name,
        symbol="*",
    )
    dst_view.copy_(src_a_view * src_b_view)
    note_tensor_write(dst_sd, item.dst_name)


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
