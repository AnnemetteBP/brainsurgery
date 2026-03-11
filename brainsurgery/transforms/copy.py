from __future__ import annotations

from .binary import BinaryMappingSpec, DestinationPolicy
from ..core import (
    ResolvedMapping,
    StateDictProvider,
    TransformError,
    register_transform,
    select_tensor,
)
from ..utils import BinaryRefs, DeclarativeBinaryTransform, Docs


def _copy_apply(
    _spec: BinaryMappingSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)

    copied = src_sd[item.src_name]
    if item.src_slice is not None:
        copied = select_tensor(copied, item.src_slice)
    copied = copied.clone()

    if item.dst_name in dst_sd:
        raise TransformError(
            f"copy destination already exists during apply: "
            f"{item.dst_model}::{item.dst_name}"
        )

    dst_sd[item.dst_name] = copied


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
