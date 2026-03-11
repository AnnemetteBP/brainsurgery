from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core import TransformError, collect_ref_models, compile_tensor_ref_expr, format_ref, register_assert_expr, require_mapping_assert_payload, resolve_matches
from ..core import StateDictProvider
from ..core import ScalarComparison, parse_scalar_comparison


@dataclass(frozen=True)
class TensorAccessExpr:
    ref: Any
    field: str
    comparison: ScalarComparison

    def evaluate(self, provider: StateDictProvider) -> None:
        model = self.ref.model
        assert model is not None
        state_dict = provider.get_state_dict(model)
        access_counts = getattr(state_dict, "access_counts", None)
        if not callable(access_counts):
            raise TransformError(f"{self.field} assertions require an instrumented state_dict backend")

        matches = resolve_matches(self.ref, provider, op_name=self.field)
        if not matches:
            raise TransformError(f"{self.field} failed: {format_ref(self.ref)} matched zero tensors")

        expected = self.comparison.describe()
        for name in matches:
            actual = access_counts(name)[self.field]
            if not self.comparison.matches(actual):
                raise TransformError(
                    f"{self.field} failed: {model}::{name} had {self.field}={actual}, expected {expected}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


def _compile_tensor_access_expr(payload: Any, default_model: str | None, *, field: str) -> TensorAccessExpr:
    mapping = require_mapping_assert_payload(
        payload,
        op_name=field,
        allowed_keys={"of", "is", "ge", "gt", "le", "lt", "at_least", "at_most"},
        required_keys={"of"},
    )
    ref = compile_tensor_ref_expr(mapping["of"], default_model, f"{field}.of")
    comparison = parse_scalar_comparison(
        mapping,
        op_name=field,
        aliases={"at_least": "ge", "at_most": "le"},
    )
    return TensorAccessExpr(ref=ref, field=field, comparison=comparison)


@register_assert_expr(
    "reads",
    payload_kind="mapping",
    allowed_keys={"of", "is", "ge", "gt", "le", "lt", "at_least", "at_most"},
    required_keys={"of"},
    description="Succeeds if every matched tensor has a read count within the requested bounds.",
)
def compile_reads_expr(payload: Any, default_model: str | None) -> TensorAccessExpr:
    return _compile_tensor_access_expr(payload, default_model, field="reads")


@register_assert_expr(
    "writes",
    payload_kind="mapping",
    allowed_keys={"of", "is", "ge", "gt", "le", "lt", "at_least", "at_most"},
    required_keys={"of"},
    description="Succeeds if every matched tensor has a write count within the requested bounds.",
)
def compile_writes_expr(payload: Any, default_model: str | None) -> TensorAccessExpr:
    return _compile_tensor_access_expr(payload, default_model, field="writes")
