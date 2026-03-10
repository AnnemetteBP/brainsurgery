from __future__ import annotations

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


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

        slot = src_sd.slot(item.src_name)

        if item.dst_name in dst_sd:
            raise MoveTransformError(
                f"move destination already exists during apply: "
                f"{item.dst_model}::{item.dst_name}"
            )

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


def _unit_test_move_compile_rejects_sliced_source() -> None:
    try:
        MoveTransform().compile({"from": "a::[:]", "to": "b"}, default_model="model")
    except MoveTransformError as exc:
        assert "source must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected move sliced source error")


def _unit_test_move_compile_rejects_sliced_destination() -> None:
    try:
        MoveTransform().compile({"from": "a", "to": "b::[:]"}, default_model="model")
    except MoveTransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected move sliced destination error")


def _unit_test_move_apply_moves_slot() -> None:
    class _StateDict(dict):
        def slot(self, key):
            return self[key]

        def bind_slot(self, key, slot):
            self[key] = slot

    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _StateDict({"src": object()})

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
    MoveTransform().apply_mapping(item, provider)
    assert "src" not in provider._state_dict
    assert "dst" in provider._state_dict


__unit_tests__ = [
    _unit_test_move_compile_rejects_sliced_source,
    _unit_test_move_compile_rejects_sliced_destination,
    _unit_test_move_apply_moves_slot,
]


register_transform(MoveTransform())
