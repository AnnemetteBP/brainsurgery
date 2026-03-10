from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..transform import (
    TypedTransform,
    TransformControl,
    TransformError,
    TransformResult,
    register_transform,
)
from ..transform_types import StateDictProvider


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

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False










register_transform(ExitTransform())
