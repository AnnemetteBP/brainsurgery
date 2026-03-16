from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TransformError,
    must_model,
    register_transform,
    state_dict_for_ref,
    view_for_ref_name,
)
from ..engine import emit_verbose_binary_activity


def _copy_apply(
    spec: BinaryMappingSpec, src_name: str, dst_name: str, provider: StateDictProvider
) -> None:
    dst_sd = state_dict_for_ref(provider, spec.to_ref)
    _src_sd, src_view = view_for_ref_name(provider, spec.from_ref, src_name)
    copied = src_view.clone()

    if dst_name in dst_sd:
        raise TransformError(
            f"copy destination already exists during apply: {must_model(spec.to_ref)}::{dst_name}"
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
