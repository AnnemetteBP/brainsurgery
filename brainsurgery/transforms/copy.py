from ..core import BinaryMappingSpec, DestinationPolicy
from ..core import StateDictProvider, TransformError, must_model, parse_slice, select_tensor
from ..core import register_transform
from ..core import BinaryRefs, DeclarativeBinaryTransform, Docs
from ..engine import emit_verbose_binary_activity


def _copy_apply(
    spec: BinaryMappingSpec, src_name: str, dst_name: str, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(must_model(spec.from_ref))
    dst_sd = provider.get_state_dict(must_model(spec.to_ref))
    src_slice = parse_slice(spec.from_ref.slice_spec) if spec.from_ref.slice_spec is not None else None

    copied = src_sd[src_name]
    if src_slice is not None:
        copied = select_tensor(copied, src_slice)
    copied = copied.clone()

    if dst_name in dst_sd:
        raise TransformError(
            f"copy destination already exists during apply: "
            f"{must_model(spec.to_ref)}::{dst_name}"
        )

    dst_sd[dst_name] = copied
    emit_verbose_binary_activity("copy", src_name, dst_name)


class CopyTransform(DeclarativeBinaryTransform[BinaryMappingSpec]):
    name = "copy"
    error_type = TransformError
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    docs = Docs(
        "Copies source tensors into new destination tensors.",
        notes=("Copied tensors are cloned and independent of the source.",),
        examples=(
            "copy: { from: ln_f.weight, to: ln_f_copy.weight }",
            "copy: { from: 'a::weight::[:, :10]', to: b.partial_weight }",
        ),
    )
    refs = BinaryRefs(from_slice=True)
    apply_fn = staticmethod(_copy_apply)


register_transform(CopyTransform())
