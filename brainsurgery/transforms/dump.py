from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from .unary import UnarySpec, UnaryTransform, resolve_target_names
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    must_model,
    parse_slice,
    register_transform,
    require_nonempty_string,
    select_tensor,
)


class DumpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class DumpSpec(UnarySpec):
    format: str


class DumpTransform(UnaryTransform[DumpSpec]):
    name = "dump"
    error_type = DumpTransformError
    spec_type = DumpSpec
    allowed_keys = {"target", "format"}
    required_keys = {"target"}
    progress_desc = None

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            parse_slice(target_ref.slice_spec)

    def build_spec(self, target_ref: TensorRef, payload: dict) -> DumpSpec:
        raw_format = payload.get("format", "tree")
        if not isinstance(raw_format, str) or not raw_format:
            raise DumpTransformError("dump.format must be a non-empty string")

        fmt = raw_format.strip().lower()
        if fmt not in {"json", "tree", "compact"}:
            raise DumpTransformError("dump.format must be one of: json, tree, compact")

        return DumpSpec(target_ref=target_ref, format=fmt)

    def resolve_targets(self, spec: DumpSpec, provider: StateDictProvider) -> list[str]:
        return resolve_target_names(
            target_ref=spec.target_ref,
            provider=provider,
            op_name=self.name,
            error_type=DumpTransformError,
        )

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

        for name in targets:
            tensor = sd[name]
            view = select_tensor(tensor, slice_spec)
            insert_into_tree(tree, name.split("."), summarize_tensor(view))

        if typed.format == "json":
            print(json.dumps(tree, separators=(",", ":"), sort_keys=True))
        elif typed.format == "tree":
            print(render_tree(tree, compact=False))
        else:
            print(render_tree(tree, compact=True))

        return TransformResult(name=self.name, count=len(targets))


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach()

    if t.numel() == 0:
        return {
            "shape": list(t.shape),
            "min": None,
            "max": None,
            "mean": None,
        }

    if not t.is_floating_point():
        t = t.to(torch.float32)

    return {
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
    }


def is_tensor_summary(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and set(node.keys()) in ({"shape", "min", "max", "mean"}, {"shape"})
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
    shape = summary["shape"]

    if compact or set(summary.keys()) == {"shape"}:
        return f"shape={shape}"

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
    current_value = shape_only(first_value)
    current_key = canonical_key(current_value)

    for index, value in present[1:]:
        value_shape = shape_only(value)
        value_key = canonical_key(value_shape)

        if index == prev + 1 and value_key == current_key:
            prev = index
            continue

        groups.append((start, prev, current_value))
        start = index
        prev = index
        current_value = value_shape
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
