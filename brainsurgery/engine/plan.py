from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

from ..core import CompiledTransform, TransformError, get_transform
from .execution import _execute_transform_pairs


class PlanLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class _OutputSpec:
    path: Path
    format: Literal["safetensors", "torch", "dcp"] | None = None
    shard: str | None = None


@dataclass
class PlanStep:
    raw: dict[str, Any]
    compiled: CompiledTransform | None = None
    status: Literal["pending", "running", "done", "failed"] = "pending"
    error: str | None = None


@dataclass
class SurgeryPlan:
    inputs: dict[str, Path]
    output: _OutputSpec | None
    steps: list[PlanStep] = field(default_factory=list)
    raw_inputs: Any = field(default_factory=list)
    raw_output: Any = None

    @property
    def transforms(self) -> list[CompiledTransform]:
        return [step.compiled for step in self.steps if step.compiled is not None]

    @property
    def executed_raw_transforms(self) -> list[dict[str, Any]]:
        return [step.raw for step in self.steps if step.status == "done"]

    def append_raw_transform(self, raw: dict[str, Any]) -> PlanStep:
        step = PlanStep(raw=raw)
        self.steps.append(step)
        return step

    def append_raw_transforms(self, raws: list[dict[str, Any]]) -> None:
        for raw in raws:
            self.append_raw_transform(raw)

    def record_executed_raw(self, raw: dict[str, Any]) -> PlanStep:
        step = self.append_raw_transform(raw)
        step.status = "done"
        return step

    def compile_pending(
        self,
        *,
        extra_known_models: set[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        if default_model is None:
            default_model = _default_model_for_inputs(self.inputs)

        known_models = set(self.inputs.keys())
        if extra_known_models is not None:
            known_models.update(extra_known_models)

        for index, step in enumerate(self.steps):
            if step.compiled is None:
                raw_entry = parse_raw_transform_entry(step.raw, index=index)
                compiled_pairs = _compile_indexed_raw_transforms(
                    [(index, raw_entry)],
                    known_models=known_models,
                    default_model=default_model,
                )
                step.compiled = compiled_pairs[0][1]
                step.status = "pending"
                step.error = None
            elif step.compiled is not None:
                _register_inferred_output_model(step.compiled, known_models)

    def execute_pending(self, state_dict_provider: Any, *, interactive: bool) -> bool:
        pending_steps = [
            step for step in self.steps if step.compiled is not None and step.status == "pending"
        ]
        pairs = [(step.raw, step.compiled) for step in pending_steps if step.compiled is not None]

        for step in pending_steps:
            step.status = "running"
            step.error = None

        should_continue, executed = _execute_transform_pairs(
            pairs,
            state_dict_provider,
            interactive=interactive,
        )

        executed_count = len(executed)
        for index, step in enumerate(pending_steps):
            if index < executed_count:
                step.status = "done"
                continue
            if step.status == "running":
                step.status = "failed" if interactive else "pending"
                if interactive:
                    step.error = "interactive execution stopped on transform error"
            break

        return should_continue

    def to_raw_plan(self, *, executed_only: bool) -> dict[str, Any]:
        transforms = (
            self.executed_raw_transforms if executed_only else [step.raw for step in self.steps]
        )
        raw: dict[str, Any] = {"inputs": self.raw_inputs, "transforms": transforms}
        if self.raw_output is not None:
            raw["output"] = self.raw_output
        return raw


@runtime_checkable
class ModelCollectingSpec(Protocol):
    def collect_models(self) -> set[str]: ...


def _default_model_for_inputs(inputs: dict[str, Path]) -> str | None:
    if len(inputs) == 1:
        return next(iter(inputs.keys()))
    return None


def _register_inferred_output_model(compiled: CompiledTransform, known_models: set[str]) -> None:
    try:
        output_model = compiled.transform._infer_output_model(compiled.spec)
    except TransformError:
        output_model = None
    if output_model:
        known_models.add(output_model)


def _compile_indexed_raw_transforms(
    indexed_raw_transforms: list[tuple[int, dict[str, Any]]],
    *,
    known_models: set[str],
    default_model: str | None,
) -> list[tuple[int, CompiledTransform]]:
    parsed: list[tuple[int, CompiledTransform]] = []
    for index, raw in indexed_raw_transforms:
        compiled = parse_transform_entry(raw, index, known_models, default_model)
        parsed.append((index, compiled))
        _register_inferred_output_model(compiled, known_models)
    return parsed


def compile_plan(raw: Any) -> SurgeryPlan:
    if not isinstance(raw, dict):
        raise PlanLoaderError("plan must be a YAML mapping")

    inputs = parse_inputs(raw.get("inputs"))
    output = parse_output(raw.get("output"))
    raw_transforms = parse_raw_transforms(raw.get("transforms"))
    compiled_transforms = [
        compiled
        for _, compiled in _compile_indexed_raw_transforms(
            list(enumerate(raw_transforms)),
            known_models=set(inputs.keys()),
            default_model=_default_model_for_inputs(inputs),
        )
    ]

    steps = [
        PlanStep(raw=raw_transform, compiled=compiled_transform)
        for raw_transform, compiled_transform in zip(
            raw_transforms, compiled_transforms, strict=False
        )
    ]
    plan = SurgeryPlan(
        inputs=inputs,
        output=output,
        steps=steps,
        raw_inputs=raw.get("inputs", []),
        raw_output=raw.get("output"),
    )
    return plan


def parse_inputs(raw: Any) -> dict[str, Path]:
    if raw is None:
        return {}

    if not isinstance(raw, list):
        raise PlanLoaderError("inputs must be a list when provided")

    if not raw:
        return {}

    parsed: dict[str, Path] = {}
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


def parse_output(raw: Any) -> _OutputSpec | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        if not raw:
            return None
        return _OutputSpec(path=Path(raw))

    if isinstance(raw, dict):
        if not raw:
            return None
        return parse_output_mapping(raw)

    raise PlanLoaderError("output must be either empty, a non-empty string, or a mapping")


def parse_output_mapping(raw: dict[str, Any]) -> _OutputSpec:
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
        if format_value not in {"safetensors", "torch", "dcp"}:
            raise PlanLoaderError("output.format must be one of: 'safetensors', 'torch', 'dcp'")

    shard_value = raw.get("shard")
    if shard_value is not None:
        if not isinstance(shard_value, str) or not shard_value:
            raise PlanLoaderError("output.shard must be a non-empty string when provided")

    return _OutputSpec(
        path=Path(path_value),
        format=cast(Literal["safetensors", "torch", "dcp"] | None, format_value),
        shard=shard_value,
    )


def parse_raw_transforms(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []

    if not isinstance(raw, list):
        raise PlanLoaderError("transforms must be a list when provided")

    return [parse_raw_transform_entry(item, index=i) for i, item in enumerate(raw)]


def parse_raw_transform_entry(raw: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise PlanLoaderError(f"transform #{index} must be a single-key mapping")

    op_name, _payload = next(iter(raw.items()))
    if not isinstance(op_name, str) or not op_name:
        raise PlanLoaderError(f"transform #{index}: operation name must be a non-empty string")

    return raw


def parse_transform_entry(
    raw: dict[str, Any],
    index: int,
    known_models: set[str],
    default_model: str | None,
) -> CompiledTransform:
    op_name, payload = next(iter(raw.items()))

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
