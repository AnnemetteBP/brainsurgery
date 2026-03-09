from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

import yaml

from .transform import CompiledTransform, TransformError, get_transform


class PlanLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutputSpec:
    path: Path
    format: Optional[Literal["safetensors", "torch"]] = None
    shard: Optional[str] = None


@dataclass(frozen=True)
class SurgeryPlan:
    inputs: Dict[str, Path]
    output: OutputSpec | None
    transforms: List[CompiledTransform]


@runtime_checkable
class ModelCollectingSpec(Protocol):
    def collect_models(self) -> set[str]:
        ...


def load_plan(path: str | Path) -> SurgeryPlan:
    plan_path = Path(path)

    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanLoaderError(f"failed to read plan file: {plan_path}") from exc
    except yaml.YAMLError as exc:
        raise PlanLoaderError(f"failed to parse yaml from plan file: {plan_path}") from exc

    return compile_plan(raw)


def compile_plan(raw: Any) -> SurgeryPlan:
    if not isinstance(raw, dict):
        raise PlanLoaderError("plan must be a YAML mapping")

    inputs = parse_inputs(raw.get("inputs"))
    output = parse_output(raw.get("output"))
    transforms = parse_transforms(raw.get("transforms"), inputs)

    return SurgeryPlan(inputs=inputs, output=output, transforms=transforms)


def parse_inputs(raw: Any) -> Dict[str, Path]:
    if raw is None:
        return {}

    if not isinstance(raw, list):
        raise PlanLoaderError("inputs must be a list when provided")

    if not raw:
        return {}

    parsed: Dict[str, Path] = {}
    single_input = len(raw) == 1

    for item in raw:
        alias, path = parse_input_entry(item)

        if alias is None:
            if not single_input:
                raise PlanLoaderError(
                    f"input alias must not be empty when multiple inputs are provided: {item!r}"
                )
            alias = "model"

        if alias in parsed:
            raise PlanLoaderError(f"duplicate input alias: {alias!r}")
        parsed[alias] = path

    return parsed


def parse_input_entry(raw: Any) -> tuple[str | None, Path]:
    if not isinstance(raw, str) or not raw:
        raise PlanLoaderError("each inputs entry must be a non-empty string")

    if "::" in raw:
        alias, path_str = raw.split("::", 1)
        if not path_str:
            raise PlanLoaderError(f"input path must not be empty: {raw!r}")
        return alias or None, Path(path_str)

    return None, Path(raw)


def parse_output(raw: Any) -> OutputSpec | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        if not raw:
            return None
        return OutputSpec(path=Path(raw))

    if isinstance(raw, dict):
        if not raw:
            return None
        return parse_output_mapping(raw)

    raise PlanLoaderError("output must be either empty, a non-empty string, or a mapping")


def parse_output_mapping(raw: Dict[str, Any]) -> OutputSpec:
    allowed_keys = {"path", "format", "shard"}

    unknown = set(raw) - allowed_keys
    if unknown:
        raise PlanLoaderError(f"output received unknown keys: {sorted(unknown)}")

    if "path" not in raw:
        raise PlanLoaderError("output.path is required")

    path_value = raw["path"]
    if not isinstance(path_value, str) or not path_value:
        raise PlanLoaderError("output.path must be a non-empty string")

    format_value = raw.get("format")
    if format_value is not None:
        if not isinstance(format_value, str) or not format_value:
            raise PlanLoaderError("output.format must be a non-empty string when provided")
        if format_value not in {"safetensors", "torch"}:
            raise PlanLoaderError("output.format must be one of: 'safetensors', 'torch'")

    shard_value = raw.get("shard")
    if shard_value is not None:
        if not isinstance(shard_value, str) or not shard_value:
            raise PlanLoaderError("output.shard must be a non-empty string when provided")

    return OutputSpec(
        path=Path(path_value),
        format=format_value,
        shard=shard_value,
    )


def parse_transforms(raw: Any, inputs: Dict[str, Path]) -> List[CompiledTransform]:
    if raw is None:
        return []

    if not isinstance(raw, list):
        raise PlanLoaderError("transforms must be a list when provided")

    default_model: Optional[str] = None
    if len(inputs) == 1:
        default_model = next(iter(inputs.keys()))

    parsed: List[CompiledTransform] = []
    known_models = set(inputs.keys())
    for idx, item in enumerate(raw):
        compiled = parse_transform_entry(item, idx, known_models, default_model)
        parsed.append(compiled)

        try:
            output_model = compiled.transform.infer_output_model(compiled.spec)
        except TransformError:
            output_model = None
        if output_model:
            known_models.add(output_model)

    return parsed


def parse_transform_entry(
    raw: Any,
    index: int,
    known_models: set[str],
    default_model: Optional[str],
) -> CompiledTransform:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise PlanLoaderError(f"transform #{index} must be a single-key mapping")

    op_name, payload = next(iter(raw.items()))

    if not isinstance(op_name, str) or not op_name:
        raise PlanLoaderError(f"transform #{index}: operation name must be a non-empty string")

    try:
        transform = get_transform(op_name)
    except TransformError as exc:
        raise PlanLoaderError(f"transform #{index}: {exc}") from exc

    try:
        spec = transform.compile(payload, default_model)
    except TransformError as exc:
        raise PlanLoaderError(f"transform #{index}: {exc}") from exc

    validate_model_aliases(spec, known_models, index)

    return CompiledTransform(transform=transform, spec=spec)


def validate_model_aliases(spec: object, known_models: set[str], index: int) -> None:
    if not isinstance(spec, ModelCollectingSpec):
        raise PlanLoaderError(
            f"transform #{index}: internal error: spec type {type(spec).__name__} "
            "does not expose collect_models()"
        )

    unknown = sorted(model for model in spec.collect_models() if model not in known_models)
    if unknown:
        raise PlanLoaderError(f"transform #{index}: unknown model alias: {unknown[0]!r}")
