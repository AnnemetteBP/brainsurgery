from __future__ import annotations

import torch

from ..core import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..core import ResolvedMapping
from ..core import TensorRef, parse_slice, select_tensor
from ..core import require_same_shape_dtype_device
from ..core import register_transform
from ..core import StateDictProvider, TransformError


class AssignTransformError(TransformError):
    pass


class AssignTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "assign"
    error_type = AssignTransformError
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying assign transforms"
    help_text = (
        "Copies tensor values from 'from' into 'to'. The destination tensor must "
        "already exist. Source and destination (after slicing) must have identical shapes.\n"
        "\n"
        "Both references support slicing.\n"
        "\n"
        "Examples:\n"
        "  assign: { from: a.weight, to: b.weight }\n"
        "  assign: { from: 'a.weight::[:, :10]', to: 'b.weight::[:, :10]' }"
    )

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            parse_slice(to_ref.slice_spec)

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        dst_view = select_tensor(dst_sd[item.dst_name], item.dst_slice)
        require_same_shape_dtype_device(
            src_view,
            dst_view,
            op_name="assigning",
            left_name=item.src_name,
            right_name=item.dst_name,
        )

        dst_view.copy_(src_view)
        dst_sd.mark_write(item.dst_name)










register_transform(AssignTransform())
