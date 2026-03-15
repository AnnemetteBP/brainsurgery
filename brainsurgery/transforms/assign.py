from ..core import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..core import TensorRef, must_model, parse_slice, select_tensor
from ..core import require_same_shape_dtype_device
from ..core import register_transform
from ..core import StateDictProvider
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

    def apply_mapping(self, spec: BinaryMappingSpec, src_name: str, dst_name: str, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(must_model(spec.from_ref))
        dst_sd = provider.get_state_dict(must_model(spec.to_ref))
        src_slice = parse_slice(spec.from_ref.slice_spec) if spec.from_ref.slice_spec is not None else None
        dst_slice = parse_slice(spec.to_ref.slice_spec) if spec.to_ref.slice_spec is not None else None

        src_view = select_tensor(src_sd[src_name], src_slice)
        dst_view = select_tensor(dst_sd[dst_name], dst_slice)
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
