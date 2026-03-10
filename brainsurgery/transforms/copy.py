from __future__ import annotations

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class CopyTransformError(TransformError):
    pass


class CopyTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "copy"
    error_type = CopyTransformError
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    progress_desc = "Applying copy transforms"
    help_text = (
        "Copies tensors from 'from' to new names in 'to'. The destination must not "
        "already exist.\n"
        "\n"
        "The source reference supports slicing; the destination must not be sliced. "
        "Copied tensors are cloned and independent of the source.\n"
        "\n"
        "Examples:\n"
        "  copy: { from: ln_f.weight, to: ln_f_copy.weight }\n"
        "  copy: { from: a.weight[:, :10], to: b.partial_weight }"
    )

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise CopyTransformError("copy destination must not be sliced")

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        copied = select_tensor(src_sd[item.src_name], item.src_slice).clone()

        if item.dst_name in dst_sd:
            raise CopyTransformError(
                f"copy destination already exists during apply: "
                f"{item.dst_model}::{item.dst_name}"
            )

        dst_sd[item.dst_name] = copied










register_transform(CopyTransform())
