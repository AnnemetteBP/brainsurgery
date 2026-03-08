from dataclasses import dataclass

from .binary import BinaryMappingSpec, BinaryMappingTransform
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    parse_slice,
    register_transform,
    require_dest_present,
    select_tensor,
)

class AssignTransformError(TransformError):
    pass

@dataclass(frozen=True)
class AssignSpec(BinaryMappingSpec):
    pass

class AssignTransform(BinaryMappingTransform[AssignSpec]):
    name = "assign"
    error_type = AssignTransformError
    spec_type = AssignSpec
    progress_desc = "Applying assign transforms"

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            parse_slice(to_ref.slice_spec)

    def validate_resolved_mappings(
        self, mappings: list[ResolvedMapping], provider: StateDictProvider
    ) -> None:
        require_dest_present(
            mappings=mappings,
            provider=provider,
            op_name=self.name,
        )

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        dst_view = select_tensor(dst_sd[item.dst_name], item.dst_slice)

        if src_view.shape != dst_view.shape:
            raise AssignTransformError(
                f"shape mismatch assigning {item.src_name} -> {item.dst_name}: "
                f"{tuple(src_view.shape)} != {tuple(dst_view.shape)}"
            )

        dst_view.copy_(src_view)


register_transform(AssignTransform())
