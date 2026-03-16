from dataclasses import dataclass
from typing import Any

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    require_mapping_assert_payload,
    resolve_matches,
)


@dataclass(frozen=True)
class CountExpr:
    ref: TensorRef
    is_value: int

    def evaluate(self, provider: StateDictProvider) -> None:
        matches = resolve_matches(self.ref, provider, op_name="count.of")
        if len(matches) != self.is_value:
            raise TransformError(
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
        raise TransformError("count.is must be an integer")
    return CountExpr(ref=ref, is_value=is_value)
