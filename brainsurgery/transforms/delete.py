from __future__ import annotations

from .unary import UnarySpec, UnaryTransform
from ..refs import TensorRef, must_model
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class DeleteTransformError(TransformError):
    pass


class DeleteTransform(UnaryTransform[UnarySpec]):
    name = "delete"
    error_type = DeleteTransformError
    spec_type = UnarySpec
    progress_desc = "Applying delete transforms"
    help_text = (
        "Deletes one or more tensors selected by 'target'.\n"
        "\n"
        "Targets may be specified by name or pattern. The entire tensor is removed; "
        "slicing is not supported.\n"
        "\n"
        "Examples:\n"
        "  delete: { target: ln_f_copy.weight }\n"
        "  delete: { target: '.*_backup' }"
    )

    def apply_to_target(self, spec: UnarySpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if name not in sd:
            raise DeleteTransformError(f"delete target disappeared during apply: {model}::{name}")

        del sd[name]


def _unit_test_delete_compile_rejects_sliced_target() -> None:
    try:
        DeleteTransform().compile({"target": "x::[:]"}, default_model="model")
    except DeleteTransformError as exc:
        assert "target must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced target error")


def _unit_test_delete_apply_removes_target() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": object()}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = UnarySpec(target_ref=TensorRef(model="model", expr="x"))
    DeleteTransform().apply_to_target(spec, "x", provider)
    assert "x" not in provider._state_dict


def _unit_test_delete_apply_raises_for_missing_target() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    spec = UnarySpec(target_ref=TensorRef(model="model", expr="x"))
    try:
        DeleteTransform().apply_to_target(spec, "x", _Provider())
    except DeleteTransformError as exc:
        assert "disappeared" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing target error")


__unit_tests__ = [
    _unit_test_delete_compile_rejects_sliced_target,
    _unit_test_delete_apply_removes_target,
    _unit_test_delete_apply_raises_for_missing_target,
]


register_transform(DeleteTransform())
