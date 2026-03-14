from ..engine import get_runtime_flags
from ..core import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..core import ResolvedMapping
from ..core import TensorRef
from ..core import StateDictProvider, TransformError
from ..core import register_transform
from ..engine import emit_verbose_binary_activity


class MoveTransformError(TransformError):
    pass


class MoveTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "move"
    error_type = MoveTransformError
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    progress_desc = "Applying move transforms"
    help_text = (
        "Moves tensors from one name to another without copying.\n"
        "\n"
        "Both source ('from') and destination ('to') must refer to whole tensors "
        "(slicing is not allowed). Destination names must not already exist.\n"
        "\n"
        "Examples:\n"
        "  move: { from: ln_f.weight, to: ln_f_copy.weight }\n"
        "  move: { from: model::h.0.attn.c_proj.weight, to: model::proj.weight }"
    )

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            raise MoveTransformError("move source must not be sliced")
        if to_ref.slice_spec is not None:
            raise MoveTransformError("move destination must not be sliced")

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        if item.dst_name in dst_sd:
            raise MoveTransformError(
                f"move destination already exists during apply: "
                f"{item.dst_model}::{item.dst_name}"
            )

        if get_runtime_flags().dry_run:
            # In dry-run we still execute the transform flow, but with tensor-level overlay
            # semantics instead of slot rebinding.
            dst_sd[item.dst_name] = src_sd[item.src_name]
            del src_sd[item.src_name]
        else:
            slot = src_sd.slot(item.src_name)
            dst_sd.bind_slot(item.dst_name, slot)
            del src_sd[item.src_name]

        if item.dst_name not in dst_sd:
            raise MoveTransformError(
                f"move internal error: destination missing after move: "
                f"{item.dst_model}::{item.dst_name}"
            )
        if item.src_name in src_sd:
            raise MoveTransformError(
                f"move internal error: source still present after move: "
                f"{item.src_model}::{item.src_name}"
            )
        emit_verbose_binary_activity(self.name, item)










register_transform(MoveTransform())
