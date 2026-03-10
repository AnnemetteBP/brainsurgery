from __future__ import annotations

import torch

from ..ternary import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class MatmulTransformError(TransformError):
    pass


class MatmulTransform(TernaryMappingTransform[TernaryMappingSpec]):
    name = "matmul"
    error_type = MatmulTransformError
    spec_type = TernaryMappingSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    progress_desc = "Applying matmul transforms"
    help_text = (
        "Computes matrix multiplication from 'from_a' and 'from_b' into 'to'. "
        "Destination tensors must not already exist.\n"
        "\n"
        "All references may include slicing and may be regex/structured mappings.\n"
        "\n"
        "Example:\n"
        "  matmul: { from_a: a.weight, from_b: b.weight, to: out.weight }"
    )

    def validate_refs(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_a_ref.slice_spec is not None:
            parse_slice(from_a_ref.slice_spec)
        if from_b_ref.slice_spec is not None:
            parse_slice(from_b_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise MatmulTransformError("matmul destination must not be sliced")

    def apply_mapping(self, item: ResolvedTernaryMapping, provider: StateDictProvider) -> None:
        src_a_sd = provider.get_state_dict(item.src_a_model)
        src_b_sd = provider.get_state_dict(item.src_b_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_a_view = select_tensor(src_a_sd[item.src_a_name], item.src_a_slice)
        src_b_view = select_tensor(src_b_sd[item.src_b_name], item.src_b_slice)

        if src_a_view.dtype != src_b_view.dtype:
            raise MatmulTransformError(
                f"dtype mismatch matmul {item.src_a_name} @ {item.src_b_name}: "
                f"{src_a_view.dtype} != {src_b_view.dtype}"
            )
        if src_a_view.device != src_b_view.device:
            raise MatmulTransformError(
                f"device mismatch matmul {item.src_a_name} @ {item.src_b_name}: "
                f"{src_a_view.device} != {src_b_view.device}"
            )

        try:
            result = torch.matmul(src_a_view, src_b_view)
        except RuntimeError as exc:
            raise MatmulTransformError(
                f"shape mismatch matmul {item.src_a_name} @ {item.src_b_name}: {exc}"
            ) from exc

        dst_sd[item.dst_name] = result.clone()








register_transform(MatmulTransform())
