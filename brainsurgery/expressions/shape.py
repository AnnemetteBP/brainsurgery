from dataclasses import dataclass
from typing import Any

from ..core import collect_ref_models, compile_shape, compile_tensor_ref_expr, format_ref, register_assert_expr, resolve_tensors, require_mapping_assert_payload
from ..core import TransformError
from ..core import TensorRef
from ..core import StateDictProvider


@dataclass(frozen=True)
class ShapeExpr:
    ref: TensorRef
    is_value: tuple[int, ...]

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="shape.of"):
            if tuple(tensor.shape) != self.is_value:
                raise TransformError(
                    f"shape failed: {format_ref(ref)} has shape {tuple(tensor.shape)}, expected {self.is_value}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "shape",
    payload_kind="mapping",
    allowed_keys={"of", "is"},
    required_keys={"of", "is"},
    description="Succeeds if the tensor has the given shape.",
)
def compile_shape_expr(payload: Any, default_model: str | None) -> ShapeExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="shape",
        allowed_keys={"of", "is"},
        required_keys={"of", "is"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "shape.of")
    return ShapeExpr(ref=ref, is_value=compile_shape(payload["is"]))
