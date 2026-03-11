from __future__ import annotations

import json
from typing import Any

import torch


def summarize_tensor(
    tensor: torch.Tensor,
    *,
    verbosity: str,
    access_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    t = tensor.detach()

    if verbosity == "shape":
        return {"shape": list(t.shape)}

    if verbosity == "full":
        values_tensor = t.cpu()
        summary = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "values": values_tensor.tolist(),
        }
        if access_counts is not None:
            summary["reads"] = access_counts["reads"]
            summary["writes"] = access_counts["writes"]
        return summary

    if t.numel() == 0:
        summary = {
            "shape": list(t.shape),
            "min": None,
            "max": None,
            "mean": None,
        }
        if access_counts is not None:
            summary["reads"] = access_counts["reads"]
            summary["writes"] = access_counts["writes"]
        return summary

    stat_tensor = t
    if not stat_tensor.is_floating_point():
        stat_tensor = stat_tensor.to(torch.float32)

    summary = {
        "shape": list(t.shape),
        "min": float(stat_tensor.min().item()),
        "max": float(stat_tensor.max().item()),
        "mean": float(stat_tensor.mean().item()),
    }
    if access_counts is not None:
        summary["reads"] = access_counts["reads"]
        summary["writes"] = access_counts["writes"]
    return summary


def _is_tensor_summary(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and set(node.keys())
        in (
            {"shape"},
            {"shape", "min", "max", "mean"},
            {"shape", "min", "max", "mean", "reads", "writes"},
            {"shape", "dtype", "device", "values"},
            {"shape", "dtype", "device", "values", "reads", "writes"},
        )
    )


def _shape_only(node: Any) -> Any:
    if _is_tensor_summary(node):
        return {"shape": node["shape"]}

    if isinstance(node, dict):
        return {key: _shape_only(value) for key, value in node.items()}

    if isinstance(node, list):
        return [None if value is None else _shape_only(value) for value in node]

    return node


def _format_summary(summary: dict[str, Any], *, compact: bool) -> str:
    del compact  # layout is handled by the tree renderer, not by summary content

    shape = summary["shape"]

    if set(summary.keys()) == {"shape"}:
        return f"shape={shape}"

    if {"dtype", "device", "values"}.issubset(summary.keys()):
        rendered = (
            f"shape={shape} "
            f"dtype={summary['dtype']} "
            f"device={summary['device']} "
            f"values={summary['values']!r}"
        )
        if "reads" in summary and "writes" in summary:
            rendered += f" reads={summary['reads']} writes={summary['writes']}"
        return rendered

    min_value = summary["min"]
    max_value = summary["max"]
    mean_value = summary["mean"]

    if min_value is None:
        rendered = f"shape={shape} min=None max=None mean=None"
        if "reads" in summary and "writes" in summary:
            rendered += f" reads={summary['reads']} writes={summary['writes']}"
        return rendered

    rendered = (
        f"shape={shape} "
        f"min={min_value:.6g} "
        f"max={max_value:.6g} "
        f"mean={mean_value:.6g}"
    )
    if "reads" in summary and "writes" in summary:
        rendered += f" reads={summary['reads']} writes={summary['writes']}"
    return rendered


def render_tree(tree: dict[str, Any], *, compact: bool) -> str:
    lines: list[str] = []
    items = list(tree.items())

    for index, (key, value) in enumerate(items):
        lines.extend(
            _render_node(
                str(key),
                value,
                prefix="",
                is_last=index == len(items) - 1,
                compact=compact,
            )
        )

    return "\n".join(lines)


def _render_node(
    name: str,
    node: Any,
    *,
    prefix: str,
    is_last: bool,
    compact: bool,
) -> list[str]:
    branch = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    if _is_tensor_summary(node):
        return [f"{prefix}{branch}{name}  {_format_summary(node, compact=compact)}"]

    lines = [f"{prefix}{branch}{name}"]

    if isinstance(node, dict):
        items = list(node.items())
        for index, (child_name, child_node) in enumerate(items):
            lines.extend(
                _render_node(
                    str(child_name),
                    child_node,
                    prefix=child_prefix,
                    is_last=index == len(items) - 1,
                    compact=compact,
                )
            )
        return lines

    if isinstance(node, list):
        groups = _list_group_entries(node, compact=compact)
        for index, (label, child_node) in enumerate(groups):
            lines.extend(
                _render_node(
                    label,
                    child_node,
                    prefix=child_prefix,
                    is_last=index == len(groups) - 1,
                    compact=compact,
                )
            )
        return lines

    return [f"{prefix}{branch}{name}  {node!r}"]


def _list_group_entries(node: list[Any], *, compact: bool) -> list[tuple[str, Any]]:
    present = [(index, value) for index, value in enumerate(node) if value is not None]
    if not present:
        return []

    if not compact:
        return [(_format_index_range(index, index), value) for index, value in present]

    groups: list[tuple[int, int, Any]] = []

    start, first_value = present[0]
    prev = start
    current_value = first_value
    current_key = _canonical_key(first_value)

    for index, value in present[1:]:
        value_key = _canonical_key(value)

        if index == prev + 1 and value_key == current_key:
            prev = index
            continue

        groups.append((start, prev, current_value))
        start = index
        prev = index
        current_value = value
        current_key = value_key

    groups.append((start, prev, current_value))

    return [(_format_index_range(start, end), value) for start, end, value in groups]


def _format_index_range(start: int, end: int) -> str:
    if start == end:
        return f"[{start}]"
    return f"[{start}-{end}]"


def _canonical_key(node: Any) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


__all__ = [
    "summarize_tensor",
    "render_tree",
]
