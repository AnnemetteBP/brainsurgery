from __future__ import annotations

from .binary import BinaryMappingSpec, BinaryMappingTransform
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    register_transform,
    require_dest_missing,
)


class MoveTransformError(TransformError):
    pass


class MoveTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "move"
    error_type = MoveTransformError
    spec_type = BinaryMappingSpec
    progress_desc = "Applying move transforms"

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            raise MoveTransformError("move source must not be sliced")
        if to_ref.slice_spec is not None:
            raise MoveTransformError("move destination must not be sliced")

    def validate_resolved_mappings(
        self,
        mappings: list[ResolvedMapping],
        provider: StateDictProvider,
    ) -> None:
        require_dest_missing(
            mappings=mappings,
            provider=provider,
            op_name=self.name,
        )

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        slot = src_sd.slot(item.src_name)

        if item.dst_name in dst_sd:
            raise MoveTransformError(
                f"move destination already exists during apply: "
                f"{item.dst_model}::{item.dst_name}"
            )

        dst_sd.bind_slot(item.dst_name, slot)
        del src_sd[item.src_name]


register_transform(MoveTransform())
