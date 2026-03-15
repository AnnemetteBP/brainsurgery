from ..engine import get_runtime_flags
from ..core import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..core import TensorRef, must_model
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

    def apply_mapping(self, spec: BinaryMappingSpec, src_name: str, dst_name: str, provider: StateDictProvider) -> None:
        src_model = must_model(spec.from_ref)
        dst_model = must_model(spec.to_ref)
        src_sd = provider.get_state_dict(src_model)
        dst_sd = provider.get_state_dict(dst_model)

        if dst_name in dst_sd:
            raise MoveTransformError(
                f"move destination already exists during apply: "
                f"{dst_model}::{dst_name}"
            )

        if get_runtime_flags().dry_run:
            # In dry-run we still execute the transform flow, but with tensor-level overlay
            # semantics instead of slot rebinding.
            dst_sd[dst_name] = src_sd[src_name]
            del src_sd[src_name]
        else:
            slot = src_sd.slot(src_name)
            dst_sd.bind_slot(dst_name, slot)
            del src_sd[src_name]

        if dst_name not in dst_sd:
            raise MoveTransformError(
                f"move internal error: destination missing after move: "
                f"{dst_model}::{dst_name}"
            )
        if src_name in src_sd:
            raise MoveTransformError(
                f"move internal error: source still present after move: "
                f"{src_model}::{src_name}"
            )
        emit_verbose_binary_activity(self.name, src_name, dst_name)










register_transform(MoveTransform())
