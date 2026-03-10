from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..transform import (
    StateDictProvider,
    TypedTransform,
    TransformControl,
    TransformError,
    TransformResult,
    register_transform,
)


class ExitTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ExitSpec:
    def collect_models(self) -> set[str]:
        return set()


class ExitTransform(TypedTransform[ExitSpec]):
    name = "exit"
    error_type = ExitTransformError
    spec_type = ExitSpec
    completion_requires_payload = False
    help_text = (
        "Exits the current execution loop.\n"
        "\n"
        "In interactive mode, returns to the caller and stops further transforms. "
        "Does not modify tensors and takes no arguments.\n"
        "\n"
        "Examples:\n"
        "  exit"
    )

    def compile(self, payload: Any, default_model: str | None) -> ExitSpec:
        del default_model

        if payload is None:
            return ExitSpec()

        if isinstance(payload, dict) and not payload:
            return ExitSpec()

        raise ExitTransformError("exit does not take any payload")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        del provider

        self.require_spec(spec)

        return TransformResult(
            name=self.name,
            count=0,
            control=TransformControl.EXIT,
        )

    def infer_output_model(self, spec: object) -> str:
        del spec
        raise ExitTransformError("exit does not infer an output model")


def _unit_test_exit_compile_rejects_payload() -> None:
    try:
        ExitTransform().compile({"unexpected": 1}, default_model=None)
    except ExitTransformError as exc:
        assert "does not take any payload" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected exit payload error")


def _unit_test_exit_compile_accepts_none_and_empty_mapping() -> None:
    t = ExitTransform()
    assert isinstance(t.compile(None, default_model=None), ExitSpec)
    assert isinstance(t.compile({}, default_model=None), ExitSpec)


def _unit_test_exit_apply_returns_exit_control() -> None:
    result = ExitTransform().apply(ExitSpec(), provider=None)  # type: ignore[arg-type]
    assert result.control == TransformControl.EXIT
    assert result.count == 0


__unit_tests__ = [
    _unit_test_exit_compile_rejects_payload,
    _unit_test_exit_compile_accepts_none_and_empty_mapping,
    _unit_test_exit_apply_returns_exit_control,
]


register_transform(ExitTransform())
