from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import OmegaConf

from ..core import TensorRef
from .plan import SurgeryPlan

SummaryMode = Literal["raw", "resolve"]


def _normalize_summary_node(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {key: _normalize_summary_node(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_summary_node(item) for item in value]
    return value


def parse_summary_mode(mode: str) -> SummaryMode:
    mode_name = mode.strip().lower()
    if mode_name not in {"raw", "resolve"}:
        raise ValueError("summary mode must be one of: raw, resolve")
    return cast(SummaryMode, mode_name)


def _serialize_tensor_ref(ref: TensorRef) -> str | list[Any]:
    expr = ref.expr
    if isinstance(expr, str):
        name = expr
        if ref.slice_spec is not None:
            name = f"{name}::{ref.slice_spec}"
        if ref.model is not None:
            return f"{ref.model}::{name}"
        return name
    if isinstance(expr, list):
        return [_serialize_scalar(item) for item in expr]
    return str(expr)


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, TensorRef):
        return _serialize_tensor_ref(value)
    if is_dataclass(value):
        return {
            field.name: _serialize_scalar(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_serialize_scalar(item) for item in value]
    if isinstance(value, list):
        return [_serialize_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_scalar(item) for key, item in value.items()}
    if type(value).__name__ == "dtype" and type(value).__module__ == "torch":
        return str(value).replace("torch.", "")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize_scalar_comparison(comparison: Any) -> dict[str, int]:
    payload: dict[str, int] = {}
    exact = getattr(comparison, "exact", None)
    ge = getattr(comparison, "ge", None)
    gt = getattr(comparison, "gt", None)
    le = getattr(comparison, "le", None)
    lt = getattr(comparison, "lt", None)
    if exact is not None:
        payload["is"] = int(exact)
    if ge is not None:
        payload["ge"] = int(ge)
    if gt is not None:
        payload["gt"] = int(gt)
    if le is not None:
        payload["le"] = int(le)
    if lt is not None:
        payload["lt"] = int(lt)
    return payload


def _serialize_assert_expr(expr: Any) -> dict[str, Any]:
    class_name = type(expr).__name__
    if class_name == "AllExpr":
        return {"all": [_serialize_assert_expr(item) for item in getattr(expr, "exprs")]}
    if class_name == "AnyExpr":
        return {"any": [_serialize_assert_expr(item) for item in getattr(expr, "exprs")]}
    if class_name == "NotExpr":
        return {"not": _serialize_assert_expr(getattr(expr, "expr"))}
    if class_name == "ExistsExpr":
        return {"exists": _serialize_tensor_ref(getattr(expr, "ref"))}
    if class_name == "EqualExpr":
        payload: dict[str, Any] = {
            "left": _serialize_tensor_ref(getattr(expr, "left")),
            "right": _serialize_tensor_ref(getattr(expr, "right")),
        }
        eps = getattr(expr, "eps", None)
        if eps is not None:
            payload["eps"] = float(eps)
        return {"equal": payload}
    if class_name == "IsZeroExpr":
        payload_iszero: dict[str, Any] = {"of": _serialize_tensor_ref(getattr(expr, "ref"))}
        eps = getattr(expr, "eps", None)
        if eps is None:
            return {"iszero": payload_iszero["of"]}
        payload_iszero["eps"] = float(eps)
        return {"iszero": payload_iszero}
    if class_name == "DtypeExpr":
        return {
            "dtype": {
                "of": _serialize_tensor_ref(getattr(expr, "ref")),
                "is": _serialize_scalar(getattr(expr, "is_value")),
            }
        }
    if class_name == "ShapeExpr":
        return {
            "shape": {
                "of": _serialize_tensor_ref(getattr(expr, "ref")),
                "is": _serialize_scalar(getattr(expr, "is_value")),
            }
        }
    if class_name == "CountExpr":
        return {
            "count": {
                "of": _serialize_tensor_ref(getattr(expr, "ref")),
                "is": int(getattr(expr, "is_value")),
            }
        }
    if class_name == "DimensionsExpr":
        payload = {"of": _serialize_tensor_ref(getattr(expr, "ref"))}
        payload.update(_serialize_scalar_comparison(getattr(expr, "comparison")))
        return {"dimensions": payload}
    if class_name == "TensorAccessExpr":
        field = str(getattr(expr, "field"))
        payload = {"of": _serialize_tensor_ref(getattr(expr, "ref"))}
        payload.update(_serialize_scalar_comparison(getattr(expr, "comparison")))
        return {field: payload}

    if is_dataclass(expr):
        return {
            class_name.removesuffix("Expr").lower(): {
                field.name: _serialize_scalar(getattr(expr, field.name)) for field in fields(expr)
            }
        }
    return {"expr": _serialize_scalar(expr)}


def _serialize_fill_config(config: Any) -> dict[str, Any]:
    mode = str(getattr(config, "mode"))
    payload: dict[str, Any] = {"mode": mode}
    seed = getattr(config, "seed", None)
    if seed is not None:
        payload["seed"] = int(seed)
    if mode == "constant":
        payload["value"] = _serialize_scalar(getattr(config, "constant_value"))
        return payload
    if mode == "tensor":
        payload["values"] = _serialize_scalar(getattr(config, "values_payload"))
        return payload

    distribution = str(getattr(config, "distribution"))
    payload["distribution"] = distribution
    if distribution == "uniform":
        payload["low"] = _serialize_scalar(getattr(config, "low"))
        payload["high"] = _serialize_scalar(getattr(config, "high"))
    else:
        payload["mean"] = _serialize_scalar(getattr(config, "mean"))
        payload["std"] = _serialize_scalar(getattr(config, "std"))
    return payload


def _serialize_fill_spec(transform_name: str, spec: Any) -> dict[str, Any]:
    payload = _serialize_fill_config(getattr(spec, "config"))
    if transform_name == "fill":
        payload["from"] = _serialize_tensor_ref(getattr(spec, "from_ref"))
        payload["to"] = _serialize_tensor_ref(getattr(spec, "to_ref"))
    else:
        payload["target"] = _serialize_tensor_ref(getattr(spec, "target_ref"))
    return {transform_name: payload}


def _resolved_step_payload(step: Any) -> dict[str, Any]:
    compiled = step.compiled
    if compiled is None:
        return _normalize_summary_node(step.raw)

    transform = compiled.transform
    spec = compiled.spec
    transform_name = transform.name

    if transform_name == "assert" and hasattr(spec, "expr"):
        return {"assert": _serialize_assert_expr(spec.expr)}
    if transform_name in {"fill", "fill_"} and hasattr(spec, "config"):
        return _serialize_fill_spec(transform_name, spec)
    return _normalize_summary_node(step.raw)


def _resolved_summary_doc(plan: SurgeryPlan) -> dict[str, Any]:
    inputs = [f"{alias}::{path}" for alias, path in sorted(plan.inputs.items())]
    transforms = [_resolved_step_payload(step) for step in plan.steps if step.status == "done"]
    doc: dict[str, Any] = {
        "inputs": inputs,
        "transforms": transforms,
    }
    if plan.output is not None:
        output_doc: dict[str, Any] = {
            "path": str(plan.output.path),
        }
        if plan.output.format is not None:
            output_doc["format"] = plan.output.format
        if plan.output.shard is not None:
            output_doc["shard"] = plan.output.shard
        doc["output"] = output_doc
    return doc


def executed_plan_summary_doc(plan: SurgeryPlan, *, mode: SummaryMode = "raw") -> dict[str, Any]:
    if mode == "resolve":
        return _resolved_summary_doc(plan)
    return {
        "transforms": [_normalize_summary_node(item) for item in plan.executed_raw_transforms],
    }


def executed_plan_summary_yaml(plan: SurgeryPlan, *, mode: SummaryMode = "raw") -> str:
    return OmegaConf.to_yaml(executed_plan_summary_doc(plan, mode=mode))
