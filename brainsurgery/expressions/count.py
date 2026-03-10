from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expression import (
    AssertTransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    require_mapping_assert_payload,
    resolve_matches,
)
from ..refs import TensorRef
from ..transform import StateDictProvider


@dataclass(frozen=True)
class CountExpr:
    ref: TensorRef
    is_value: int

    def evaluate(self, provider: StateDictProvider) -> None:
        matches = resolve_matches(self.ref, provider, op_name="count.of")
        if len(matches) != self.is_value:
            raise AssertTransformError(
                f"count failed: {format_ref(self.ref)} matched {len(matches)} tensors, expected {self.is_value}"
            )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "count",
    payload_kind="mapping",
    allowed_keys={"of", "is"},
    required_keys={"of", "is"},
    description="Succeeds if the reference matches exactly the given number of tensors.",
)
def compile_count_expr(payload: Any, default_model: str | None) -> CountExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="count",
        allowed_keys={"of", "is"},
        required_keys={"of", "is"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "count.of")
    is_value = payload["is"]
    if not isinstance(is_value, int):
        raise AssertTransformError("count.is must be an integer")
    return CountExpr(ref=ref, is_value=is_value)


def _unit_test_count_compile_rejects_non_int_is() -> None:
    try:
        compile_count_expr({"of": "x", "is": "1"}, default_model="model")
    except AssertTransformError as exc:
        assert "count.is must be an integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected count.is integer validation error")


def _unit_test_count_evaluate_exact_match() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"a": object(), "b": object(), "c": object()}

    expr = CountExpr(ref=TensorRef(model="model", expr="[ab]"), is_value=2)
    expr.evaluate(_Provider())


def _unit_test_count_evaluate_mismatch_raises() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"a": object(), "b": object()}

    expr = CountExpr(ref=TensorRef(model="model", expr="a"), is_value=2)
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "matched 1 tensors, expected 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected count mismatch error")


__unit_tests__ = [
    _unit_test_count_compile_rejects_non_int_is,
    _unit_test_count_evaluate_exact_match,
    _unit_test_count_evaluate_mismatch_raises,
]
