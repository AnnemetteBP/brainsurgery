from __future__ import annotations

import torch

from ..ternary import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..refs import TensorRef, parse_slice, select_tensor
from ..tensor_checks import require_same_shape_dtype_device3
from ..transform import StateDictProvider, TransformError, register_transform


class AddTransformError(TransformError):
    pass


class AddTransform(TernaryMappingTransform[TernaryMappingSpec]):
    name = "add"
    error_type = AddTransformError
    spec_type = TernaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying add transforms"
    help_text = (
        "Computes elementwise addition from 'from_a' and 'from_b' into 'to'. "
        "Destination tensors must already exist.\n"
        "\n"
        "All references may include slicing and may be regex/structured mappings.\n"
        "\n"
        "Examples:\n"
        "  add: { from_a: a.weight, from_b: b.weight, to: out.weight }\n"
        "  add: { from_a: '.*.weight', from_b: '.*.delta', to: '.*.weight' }"
    )

    def validate_refs(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_a_ref.slice_spec is not None:
            parse_slice(from_a_ref.slice_spec)
        if from_b_ref.slice_spec is not None:
            parse_slice(from_b_ref.slice_spec)
        if to_ref.slice_spec is not None:
            parse_slice(to_ref.slice_spec)

    def apply_mapping(self, item: ResolvedTernaryMapping, provider: StateDictProvider) -> None:
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
            error_type=AddTransformError,
            op_name="adding",
            first_name=item.src_a_name,
            second_name=item.src_b_name,
            dest_name=item.dst_name,
            symbol="+",
        )
        dst_view.copy_(src_a_view + src_b_view)


register_transform(AddTransform())
