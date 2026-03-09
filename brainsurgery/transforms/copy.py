from __future__ import annotations

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    parse_slice,
    register_transform,
    select_tensor,
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


def _unit_test_copy_compile_rejects_sliced_destination() -> None:
    try:
        CopyTransform().compile({"from": "a", "to": "b::[:]"}, default_model="model")
    except CopyTransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced destination error")


def _unit_test_copy_compile_accepts_sliced_source() -> None:
    spec = CopyTransform().compile({"from": "a::[:1]", "to": "b"}, default_model="model")
    assert spec.from_ref.slice_spec == "[:1]"
    assert spec.to_ref.slice_spec is None


def _unit_test_copy_apply_clones_tensor() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"src": torch.tensor([1.0, 2.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
    CopyTransform().apply_mapping(item, provider)
    assert torch.equal(provider._state_dict["src"], provider._state_dict["dst"])
    assert provider._state_dict["src"] is not provider._state_dict["dst"]


__unit_tests__ = [
    _unit_test_copy_compile_rejects_sliced_destination,
    _unit_test_copy_compile_accepts_sliced_source,
    _unit_test_copy_apply_clones_tensor,
]


register_transform(CopyTransform())
