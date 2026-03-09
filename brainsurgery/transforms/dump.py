from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import typer
import torch

from .unary import UnarySpec, UnaryTransform
from ..model import tqdm
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    parse_slice,
    register_transform,
    select_tensor,
    validate_payload_keys,
)


class DumpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class DumpSpec(UnarySpec):
    format: str
    verbosity: str


class DumpTransform(UnaryTransform[DumpSpec]):
    name = "dump"
    error_type = DumpTransformError
    spec_type = DumpSpec
    allowed_keys = {"target", "format", "verbosity"}
    required_keys = set()
    slice_policy = "allow"
    progress_desc = "Dumping tensors"
    help_text = (
        "Displays tensors selected by 'target' without modifying them.\n"
        "\n"
        "Targets may be specified by name or pattern. Slices are written after '::', "
        "for example 'ln_f.weight::[:8]'. 'format' controls layout: 'tree' (default), "
        "'compact', or 'json'. 'verbosity' controls content: 'shape', 'stat' (default), "
        "or 'full'.\n"
        "\n"
        "Examples:\n"
        "  dump: { target: ln_f.weight }\n"
        "  dump: { target: '.*weight', format: compact, verbosity: shape }\n"
        "  dump: { target: 'h.0.attn.c_attn.weight::[:, :10]', format: json, verbosity: stat }\n"
        "  dump: { target: 'ln_f.weight::[:8]', format: tree, verbosity: full }"
    )

    def compile(self, payload: dict, default_model: str | None) -> DumpSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        if "target" not in payload:
            payload = dict(payload)
            payload["target"] = ".*"

        raw_target = self.require_target_expr(payload)
        target_ref = parse_model_expr(raw_target, default_model=default_model)

        self.validate_target_ref(target_ref)

        assert target_ref.model is not None
        return self.build_spec(target_ref=target_ref, payload=payload)

    def build_spec(self, target_ref: TensorRef, payload: dict) -> DumpSpec:
        raw_format = payload.get("format", "compact")
        if not isinstance(raw_format, str) or not raw_format:
            raise DumpTransformError("dump.format must be a non-empty string")

        fmt = raw_format.strip().lower()
        if fmt not in {"json", "tree", "compact"}:
            raise DumpTransformError("dump.format must be one of: json, tree, compact")

        raw_verbosity = payload.get("verbosity", "shape")
        if not isinstance(raw_verbosity, str) or not raw_verbosity:
            raise DumpTransformError("dump.verbosity must be a non-empty string")

        verbosity = raw_verbosity.strip().lower()
        if verbosity not in {"shape", "stat", "full"}:
            raise DumpTransformError("dump.verbosity must be one of: shape, stat, full")

        return DumpSpec(target_ref=target_ref, format=fmt, verbosity=verbosity)

    def apply_to_target(self, spec: DumpSpec, name: str, provider: StateDictProvider) -> None:
        raise AssertionError("DumpTransform overrides apply() and does not use apply_to_target()")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)

        model = must_model(typed.target_ref)
        sd = provider.get_state_dict(model)
        targets = self.resolve_targets(typed, provider)

        slice_spec = (
            parse_slice(typed.target_ref.slice_spec)
            if typed.target_ref.slice_spec is not None
            else None
        )

        tree: dict[str, Any] = {}

        for name in tqdm(targets, desc=self.progress_desc, unit="tensor"):
            tensor = sd[name]
            view = select_tensor(tensor, slice_spec)
            insert_into_tree(tree, name.split("."), summarize_tensor(view, verbosity=typed.verbosity))

        if typed.format == "json":
            typer.echo(json.dumps(tree, separators=(",", ":"), sort_keys=True))
        elif typed.format == "tree":
            typer.echo(render_tree(tree, compact=False))
        else:
            typer.echo(render_tree(tree, compact=True))

        return TransformResult(name=self.name, count=len(targets))


def summarize_tensor(tensor: torch.Tensor, *, verbosity: str) -> dict[str, Any]:
    t = tensor.detach()

    if verbosity == "shape":
        return {"shape": list(t.shape)}

    if verbosity == "full":
        values_tensor = t.cpu()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "values": values_tensor.tolist(),
        }

    if t.numel() == 0:
        return {
            "shape": list(t.shape),
            "min": None,
            "max": None,
            "mean": None,
        }

    stat_tensor = t
    if not stat_tensor.is_floating_point():
        stat_tensor = stat_tensor.to(torch.float32)

    return {
        "shape": list(t.shape),
        "min": float(stat_tensor.min().item()),
        "max": float(stat_tensor.max().item()),
        "mean": float(stat_tensor.mean().item()),
    }


def is_tensor_summary(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and set(node.keys())
        in (
            {"shape"},
            {"shape", "min", "max", "mean"},
            {"shape", "dtype", "device", "values"},
        )
    )


def shape_only(node: Any) -> Any:
    if is_tensor_summary(node):
        return {"shape": node["shape"]}

    if isinstance(node, dict):
        return {key: shape_only(value) for key, value in node.items()}

    if isinstance(node, list):
        return [None if value is None else shape_only(value) for value in node]

    return node


def format_summary(summary: dict[str, Any], *, compact: bool) -> str:
    del compact  # layout is handled by the tree renderer, not by summary content

    shape = summary["shape"]

    if set(summary.keys()) == {"shape"}:
        return f"shape={shape}"

    if set(summary.keys()) == {"shape", "dtype", "device", "values"}:
        return (
            f"shape={shape} "
            f"dtype={summary['dtype']} "
            f"device={summary['device']} "
            f"values={summary['values']!r}"
        )

    min_value = summary["min"]
    max_value = summary["max"]
    mean_value = summary["mean"]

    if min_value is None:
        return f"shape={shape} min=None max=None mean=None"

    return (
        f"shape={shape} "
        f"min={min_value:.6g} "
        f"max={max_value:.6g} "
        f"mean={mean_value:.6g}"
    )


def render_tree(tree: dict[str, Any], *, compact: bool) -> str:
    lines: list[str] = []
    items = list(tree.items())

    for index, (key, value) in enumerate(items):
        lines.extend(
            render_node(
                str(key),
                value,
                prefix="",
                is_last=index == len(items) - 1,
                compact=compact,
            )
        )

    return "\n".join(lines)


def render_node(
    name: str,
    node: Any,
    *,
    prefix: str,
    is_last: bool,
    compact: bool,
) -> list[str]:
    branch = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    if is_tensor_summary(node):
        return [f"{prefix}{branch}{name}  {format_summary(node, compact=compact)}"]

    lines = [f"{prefix}{branch}{name}"]

    if isinstance(node, dict):
        items = list(node.items())
        for index, (child_name, child_node) in enumerate(items):
            lines.extend(
                render_node(
                    str(child_name),
                    child_node,
                    prefix=child_prefix,
                    is_last=index == len(items) - 1,
                    compact=compact,
                )
            )
        return lines

    if isinstance(node, list):
        groups = list_group_entries(node, compact=compact)
        for index, (label, child_node) in enumerate(groups):
            lines.extend(
                render_node(
                    label,
                    child_node,
                    prefix=child_prefix,
                    is_last=index == len(groups) - 1,
                    compact=compact,
                )
            )
        return lines

    return [f"{prefix}{branch}{name}  {node!r}"]


def list_group_entries(node: list[Any], *, compact: bool) -> list[tuple[str, Any]]:
    present = [(index, value) for index, value in enumerate(node) if value is not None]
    if not present:
        return []

    if not compact:
        return [(format_index_range(index, index), value) for index, value in present]

    groups: list[tuple[int, int, Any]] = []

    start, first_value = present[0]
    prev = start
    current_value = first_value
    current_key = canonical_key(first_value)

    for index, value in present[1:]:
        value_key = canonical_key(value)

        if index == prev + 1 and value_key == current_key:
            prev = index
            continue

        groups.append((start, prev, current_value))
        start = index
        prev = index
        current_value = value
        current_key = value_key

    groups.append((start, prev, current_value))

    return [(format_index_range(start, end), value) for start, end, value in groups]


def format_index_range(start: int, end: int) -> str:
    if start == end:
        return f"[{start}]"
    return f"[{start}-{end}]"


def canonical_key(node: Any) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def insert_into_tree(tree: dict[str, Any], parts: list[str], leaf: Any) -> None:
    node: Any = tree

    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        next_is_index = i + 1 < len(parts) and parts[i + 1].isdigit()

        if part.isdigit():
            idx = int(part)

            if not isinstance(node, list):
                raise DumpTransformError("invalid tree structure while building dump")

            while len(node) <= idx:
                node.append(None)

            if is_last:
                node[idx] = leaf
                return

            child = node[idx]
            if child is None:
                child = [] if next_is_index else {}
                node[idx] = child
            elif not isinstance(child, (dict, list)):
                raise DumpTransformError("invalid tree structure while building dump")

            node = child
            continue

        if not isinstance(node, dict):
            raise DumpTransformError("invalid tree structure while building dump")

        if is_last:
            node[part] = leaf
            return

        child = node.get(part)
        if child is None:
            child = [] if next_is_index else {}
            node[part] = child
        elif not isinstance(child, (dict, list)):
            raise DumpTransformError("invalid tree structure while building dump")

        node = child


register_transform(DumpTransform())
