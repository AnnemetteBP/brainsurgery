from __future__ import annotations

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..tensor_checks import require_same_shape_dtype_device
from ..transform import register_transform
from ..transform_types import StateDictProvider, TransformError, note_tensor_write


class SubtractInPlaceTransformError(TransformError):
    pass


class SubtractInPlaceTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "subtract_"
    error_type = SubtractInPlaceTransformError
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying subtract_ transforms"
    help_text = (
        "Subtracts source tensors from destination tensors in-place.\n"
        "\n"
        "Computes: to <- to - from. The destination tensor must already exist.\n"
        "Both references may include slicing.\n"
        "\n"
        "Examples:\n"
        "  subtract_: { from: delta.weight, to: model.weight }\n"
        "  subtract_: { from: 'a::[:, :10]', to: 'b::[:, :10]' }"
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
            error_type=SubtractInPlaceTransformError,
            op_name="subtracting",
            left_name=item.src_name,
            right_name=item.dst_name,
        )

        dst_view.sub_(src_view)
        note_tensor_write(dst_sd, item.dst_name)










register_transform(SubtractInPlaceTransform())
