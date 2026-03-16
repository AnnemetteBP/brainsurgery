from ..core import (
    BinaryMappingSpec,
    BinaryMappingTransform,
    DestinationPolicy,
    StateDictProvider,
    TensorRef,
    binary_mapping_views,
    parse_slice,
    register_transform,
    require_same_shape_dtype_device,
)
from ..engine import emit_verbose_binary_activity


class AssignTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "assign"
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

    def apply_mapping(
        self, spec: BinaryMappingSpec, src_name: str, dst_name: str, provider: StateDictProvider
    ) -> None:
        _src_sd, dst_sd, src_view, dst_view = binary_mapping_views(
            provider,
            from_ref=spec.from_ref,
            to_ref=spec.to_ref,
            src_name=src_name,
            dst_name=dst_name,
        )
        require_same_shape_dtype_device(
            src_view,
            dst_view,
            op_name="assigning",
            left_name=src_name,
            right_name=dst_name,
        )

        dst_view.copy_(src_view)
        dst_sd.mark_write(dst_name)
        emit_verbose_binary_activity(self.name, src_name, dst_name)


register_transform(AssignTransform())
