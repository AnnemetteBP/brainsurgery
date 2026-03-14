from dataclasses import dataclass
from typing import Any

from ..core import StateDictProvider, TransformError
from ..core import TypedTransform, TransformResult, register_transform
from ..core import ensure_mapping_payload, validate_payload_keys
from ..engine import set_runtime_flag
from ..engine import emit_verbose_event


class SetTransformError(TransformError):
    pass


@dataclass(frozen=True)
class SetSpec:
    dry_run: bool | None = None
    verbose: bool | None = None

    def collect_models(self) -> set[str]:
        return set()


class SetTransform(TypedTransform[SetSpec]):
    name = "set"
    error_type = SetTransformError
    spec_type = SetSpec
    allowed_keys = {"dry-run", "verbose"}
    help_text = (
        "Sets runtime flags used by the execution session.\n"
        "\n"
        "Supported flags:\n"
        "  - dry-run (boolean)\n"
        "  - verbose (boolean)\n"
        "\n"
        "Boolean values may be: true/false, True/False, T/F.\n"
        "These flags are currently stored but have no behavior changes yet.\n"
        "\n"
        "Examples:\n"
        "  set: { dry-run: true }\n"
        "  set: { verbose: T }\n"
        "  set: { dry-run: False, verbose: true }"
    )

    def compile(self, payload: Any, default_model: str | None) -> SetSpec:
        del default_model

        try:
            payload = ensure_mapping_payload(payload, self.name)
        except TransformError as exc:
            raise SetTransformError(str(exc)) from exc
        try:
            validate_payload_keys(
                payload,
                op_name=self.name,
                allowed_keys=self.allowed_keys,
            )
        except TransformError as exc:
            raise SetTransformError(str(exc)) from exc
        if not payload:
            raise SetTransformError("set payload must include at least one flag")

        dry_run: bool | None = None
        verbose: bool | None = None
        if "dry-run" in payload:
            dry_run = _parse_bool(payload["dry-run"], field_name="dry-run")
        if "verbose" in payload:
            verbose = _parse_bool(payload["verbose"], field_name="verbose")

        return SetSpec(
            dry_run=dry_run,
            verbose=verbose,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        del provider

        typed = self.require_spec(spec)
        changed = 0

        if typed.dry_run is not None:
            set_runtime_flag("dry_run", typed.dry_run)
            changed += 1
        if typed.verbose is not None:
            set_runtime_flag("verbose", typed.verbose)
            changed += 1

        emit_verbose_event(
            self.name,
            f"dry-run={typed.dry_run!r}, verbose={typed.verbose!r}",
        )
        return TransformResult(name=self.name, count=changed)

    def infer_output_model(self, spec: object) -> str:
        del spec
        raise SetTransformError("set does not infer an output model")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor
        candidates = [f"{name}: " for name in sorted(self.allowed_keys)]
        return [candidate for candidate in candidates if candidate.startswith(prefix_text)]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key not in self.allowed_keys:
            return None
        candidates = ["false", "true", "F", "T", "False", "True"]
        return [candidate for candidate in candidates if candidate.startswith(prefix_text)]


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"t", "true"}:
            return True
        if lowered in {"f", "false"}:
            return False

    raise SetTransformError(
        f"set.{field_name} must be a boolean (T, true, True, F, false, False)"
    )


register_transform(SetTransform())
