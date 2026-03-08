from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..transform import (
    StateDictProvider,
    TransformControl,
    TransformError,
    TransformResult,
    register_transform,
)


class ExitTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ExitSpec:
    pass


class ExitTransform:
    name = "exit"
    error_type = ExitTransformError
    spec_type = ExitSpec
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

        if not isinstance(spec, ExitSpec):
            raise self.error_type(
                f"{self.name} expected {self.spec_type.__name__}, got {type(spec).__name__}"
            )

        return TransformResult(
            name=self.name,
            count=0,
            control=TransformControl.EXIT,
        )


register_transform(ExitTransform())
