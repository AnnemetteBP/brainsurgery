from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol

import torch

from .dtypes import parse_torch_dtype
from .targeting import format_target_ref, resolve_target_names
from .transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    parse_slice,
    select_tensor,
    validate_payload_keys,
)


class AssertTransformError(TransformError):
    pass


class AssertExpr(Protocol):
    def evaluate(self, provider: StateDictProvider) -> None: ...
    def collect_models(self) -> set[str]: ...


@dataclass(frozen=True)
class ExistsExpr:
    ref: TensorRef

    def evaluate(self, provider: StateDictProvider) -> None:
        matches = resolve_matches(self.ref, provider, op_name="exists")
        if not matches:
            raise AssertTransformError(f"exists failed: {format_ref(self.ref)} matched zero tensors")

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


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


@dataclass(frozen=True)
class DtypeExpr:
    ref: TensorRef
    is_value: torch.dtype

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="dtype.of")
        if tensor.dtype != self.is_value:
            raise AssertTransformError(
                f"dtype failed: {format_ref(self.ref)} has dtype {tensor.dtype}, expected {self.is_value}"
            )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@dataclass(frozen=True)
class ShapeExpr:
    ref: TensorRef
    is_value: tuple[int, ...]

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="shape.of")
        if tuple(tensor.shape) != self.is_value:
            raise AssertTransformError(
                f"shape failed: {format_ref(self.ref)} has shape {tuple(tensor.shape)}, expected {self.is_value}"
            )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@dataclass(frozen=True)
class DimensionsExpr:
    ref: TensorRef
    is_value: int

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="dimensions.of")
        if len(tensor.shape) != self.is_value:
            raise AssertTransformError(
                f"dimensions failed: {format_ref(self.ref)} has {len(tensor.shape)} dims, expected {self.is_value}"
            )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@dataclass(frozen=True)
class IsZeroExpr:
    ref: TensorRef

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="iszero")
        if not torch.all(tensor == 0):
            raise AssertTransformError(f"iszero failed: {format_ref(self.ref)} is not all zeros")

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@dataclass(frozen=True)
class EqualExpr:
    left: TensorRef
    right: TensorRef

    def evaluate(self, provider: StateDictProvider) -> None:
        left = resolve_single_tensor(self.left, provider, op_name="equal.left")
        right = resolve_single_tensor(self.right, provider, op_name="equal.right")

        if left.shape != right.shape:
            raise AssertTransformError(
                f"equal failed: shape mismatch {tuple(left.shape)} != {tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            raise AssertTransformError(f"equal failed: dtype mismatch {left.dtype} != {right.dtype}")
        if not torch.equal(left, right):
            raise AssertTransformError(f"equal failed: {format_ref(self.left)} != {format_ref(self.right)}")

    def collect_models(self) -> set[str]:
        return {must_model(self.left), must_model(self.right)}


@dataclass(frozen=True)
class NotExpr:
    expr: AssertExpr

    def evaluate(self, provider: StateDictProvider) -> None:
        try:
            self.expr.evaluate(provider)
        except AssertTransformError:
            return

        raise AssertTransformError(f"not failed: inner assertion succeeded: {format_expr(self.expr)}")

    def collect_models(self) -> set[str]:
        return self.expr.collect_models()


@dataclass(frozen=True)
class AllExpr:
    exprs: List[AssertExpr]

    def evaluate(self, provider: StateDictProvider) -> None:
        for expr in self.exprs:
            expr.evaluate(provider)

    def collect_models(self) -> set[str]:
        return collect_expr_models(self.exprs)


@dataclass(frozen=True)
class AnyExpr:
    exprs: List[AssertExpr]

    def evaluate(self, provider: StateDictProvider) -> None:
        errors: List[str] = []
        for expr in self.exprs:
            try:
                expr.evaluate(provider)
                return
            except AssertTransformError as exc:
                errors.append(str(exc))
        raise AssertTransformError("any failed: all alternatives failed:\n- " + "\n- ".join(errors))

    def collect_models(self) -> set[str]:
        return collect_expr_models(self.exprs)


def compile_assert_expr(raw: Any, default_model: str | None) -> AssertExpr:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise AssertTransformError("assert expression must be a single-key mapping")

    op, payload = next(iter(raw.items()))

    compilers = {
        "exists": lambda p: ExistsExpr(ref=compile_tensor_ref_expr(p, default_model, "exists")),
        "count": lambda p: compile_count_expr(p, default_model),
        "dtype": lambda p: compile_dtype_expr(p, default_model),
        "shape": lambda p: compile_shape_expr(p, default_model),
        "dimensions": lambda p: compile_dimensions_expr(p, default_model),
        "iszero": lambda p: IsZeroExpr(ref=compile_tensor_ref_expr(p, default_model, "iszero")),
        "equal": lambda p: compile_equal_expr(p, default_model),
        "not": lambda p: NotExpr(expr=compile_assert_expr(p, default_model)),
        "all": lambda p: compile_all_expr(p, default_model),
        "any": lambda p: compile_any_expr(p, default_model),
    }

    try:
        return compilers[op](payload)
    except KeyError as exc:
        raise AssertTransformError(f"unknown assert op: {op!r}") from exc


def compile_count_expr(payload: Any, default_model: str | None) -> CountExpr:
    payload = ensure_mapping_payload(payload, "count")
    validate_payload_keys(payload, op_name="count", allowed_keys={"of", "is"}, required_keys={"of", "is"})
    ref = compile_tensor_ref_expr(payload["of"], default_model, "count.of")
    is_value = payload["is"]
    if not isinstance(is_value, int):
        raise AssertTransformError("count.is must be an integer")
    return CountExpr(ref=ref, is_value=is_value)


def compile_dtype_expr(payload: Any, default_model: str | None) -> DtypeExpr:
    payload = ensure_mapping_payload(payload, "dtype")
    validate_payload_keys(payload, op_name="dtype", allowed_keys={"of", "is"}, required_keys={"of", "is"})
    ref = compile_tensor_ref_expr(payload["of"], default_model, "dtype.of")

    raw_dtype = payload["is"]
    if not isinstance(raw_dtype, str) or not raw_dtype:
        raise AssertTransformError("dtype.is must be a non-empty string")

    return DtypeExpr(
        ref=ref,
        is_value=parse_torch_dtype(
            raw_dtype,
            error_type=AssertTransformError,
            op_name="dtype",
            field_name="is",
        ),
    )


def compile_shape_expr(payload: Any, default_model: str | None) -> ShapeExpr:
    payload = ensure_mapping_payload(payload, "shape")
    validate_payload_keys(payload, op_name="shape", allowed_keys={"of", "is"}, required_keys={"of", "is"})
    ref = compile_tensor_ref_expr(payload["of"], default_model, "shape.of")
    return ShapeExpr(ref=ref, is_value=compile_shape(payload["is"]))


def compile_dimensions_expr(payload: Any, default_model: str | None) -> DimensionsExpr:
    payload = ensure_mapping_payload(payload, "dimensions")
    validate_payload_keys(payload, op_name="dimensions", allowed_keys={"of", "is"}, required_keys={"of", "is"})
    ref = compile_tensor_ref_expr(payload["of"], default_model, "dimensions.of")
    is_value = payload["is"]
    if not isinstance(is_value, int):
        raise AssertTransformError("dimensions.is must be an integer")
    return DimensionsExpr(ref=ref, is_value=is_value)


def compile_equal_expr(payload: Any, default_model: str | None) -> EqualExpr:
    payload = ensure_mapping_payload(payload, "equal")
    validate_payload_keys(payload, op_name="equal", allowed_keys={"left", "right"}, required_keys={"left", "right"})
    left = compile_tensor_ref_expr(payload["left"], default_model, "equal.left")
    right = compile_tensor_ref_expr(payload["right"], default_model, "equal.right")
    return EqualExpr(left=left, right=right)


def compile_all_expr(payload: Any, default_model: str | None) -> AllExpr:
    if not isinstance(payload, list) or not payload:
        raise AssertTransformError("all must be a non-empty list")
    return AllExpr(exprs=[compile_assert_expr(item, default_model) for item in payload])


def compile_any_expr(payload: Any, default_model: str | None) -> AnyExpr:
    if not isinstance(payload, list) or not payload:
        raise AssertTransformError("any must be a non-empty list")
    return AnyExpr(exprs=[compile_assert_expr(item, default_model) for item in payload])


def compile_tensor_ref_expr(raw: Any, default_model: str | None, op_name: str) -> TensorRef:
    if isinstance(raw, str):
        if not raw:
            raise AssertTransformError(
                f"{op_name} must be a non-empty string reference or non-empty list of strings"
            )
    elif isinstance(raw, list):
        if not raw or not all(isinstance(item, str) and item for item in raw):
            raise AssertTransformError(
                f"{op_name} must be a non-empty string reference or non-empty list of strings"
            )
    else:
        raise AssertTransformError(
            f"{op_name} must be a non-empty string reference or non-empty list of strings"
        )

    ref = parse_model_expr(raw, default_model=default_model)
    if ref.slice_spec is not None:
        parse_slice(ref.slice_spec)
    return ref


def compile_shape(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not all(isinstance(x, int) for x in raw):
        raise AssertTransformError("shape.is must be a list of integers")
    return tuple(raw)


def resolve_matches(
    ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
) -> List[str]:
    return resolve_target_names(
        target_ref=ref,
        provider=provider,
        op_name=op_name,
        error_type=AssertTransformError,
    )


def resolve_single_tensor(ref: TensorRef, provider: StateDictProvider, op_name: str) -> torch.Tensor:
    model = must_model(ref)
    sd = provider.get_state_dict(model)
    matches = resolve_matches(ref, provider, op_name=op_name)

    if len(matches) == 0:
        raise AssertTransformError(f"{op_name} failed: {format_ref(ref)} matched zero tensors")
    if len(matches) != 1:
        raise AssertTransformError(f"{op_name} failed: {format_ref(ref)} matched {len(matches)} tensors, expected 1")

    tensor = sd[matches[0]]
    slice_spec = parse_slice(ref.slice_spec) if ref.slice_spec else None
    return select_tensor(tensor, slice_spec)


def collect_ref_models(ref: TensorRef) -> set[str]:
    return {must_model(ref)}


def collect_expr_models(exprs: list[AssertExpr]) -> set[str]:
    models: set[str] = set()
    for expr in exprs:
        models.update(expr.collect_models())
    return models


def format_ref(ref: TensorRef) -> str:
    return format_target_ref(ref)


def format_expr(expr: AssertExpr) -> str:
    if isinstance(expr, ExistsExpr):
        return f"exists({format_ref(expr.ref)})"
    if isinstance(expr, CountExpr):
        return f"count({format_ref(expr.ref)}) == {expr.is_value}"
    if isinstance(expr, DtypeExpr):
        return f"dtype({format_ref(expr.ref)}) == {expr.is_value}"
    if isinstance(expr, ShapeExpr):
        return f"shape({format_ref(expr.ref)}) == {expr.is_value}"
    if isinstance(expr, DimensionsExpr):
        return f"dimensions({format_ref(expr.ref)}) == {expr.is_value}"
    if isinstance(expr, IsZeroExpr):
        return f"iszero({format_ref(expr.ref)})"
    if isinstance(expr, EqualExpr):
        return f"equal({format_ref(expr.left)}, {format_ref(expr.right)})"
    if isinstance(expr, NotExpr):
        return f"not({format_expr(expr.expr)})"
    if isinstance(expr, AllExpr):
        return f"all({', '.join(format_expr(e) for e in expr.exprs)})"
    if isinstance(expr, AnyExpr):
        return f"any({', '.join(format_expr(e) for e in expr.exprs)})"
    return repr(expr)
