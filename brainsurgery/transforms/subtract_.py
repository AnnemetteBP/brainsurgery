from __future__ import annotations

from .binary import BinaryMappingSpec, DestinationPolicy
from ..core import (
    ResolvedMapping,
    StateDictProvider,
    TransformError,
    note_tensor_write,
    register_transform,
    select_tensor,
)
from ..utils import (
    BinaryRefs,
    DeclarativeBinaryTransform,
    Docs,
    require_same_shape_dtype_device,
)


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
    note_tensor_write(dst_sd, item.dst_name)


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
