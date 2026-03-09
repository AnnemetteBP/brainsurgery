from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import typer

from ..render import render_tree, summarize_tensor
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


def _unit_test_dump_compile_rejects_unknown_format() -> None:
    try:
        DumpTransform().compile({"target": "x", "format": "xml"}, default_model="model")
    except DumpTransformError as exc:
        assert "dump.format must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dump.format validation error")


def _unit_test_dump_insert_tree_rejects_invalid_node_shape() -> None:
    try:
        insert_into_tree([], ["x"], 1)  # type: ignore[arg-type]
    except DumpTransformError as exc:
        assert "invalid tree structure" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid tree structure error")


def _unit_test_dump_compile_defaults() -> None:
    spec = DumpTransform().compile({}, default_model="model")
    assert spec.target_ref.expr == ".*"
    assert spec.format == "compact"
    assert spec.verbosity == "shape"


def _unit_test_dump_apply_to_target_is_not_used() -> None:
    try:
        DumpTransform().apply_to_target(
            DumpSpec(target_ref=TensorRef(model="model", expr="x"), format="tree", verbosity="shape"),
            "x",
            provider=None,  # type: ignore[arg-type]
        )
    except AssertionError as exc:
        assert "overrides apply()" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected apply_to_target guard assertion")


__unit_tests__ = [
    _unit_test_dump_compile_rejects_unknown_format,
    _unit_test_dump_insert_tree_rejects_invalid_node_shape,
    _unit_test_dump_compile_defaults,
    _unit_test_dump_apply_to_target_is_not_used,
]


register_transform(DumpTransform())
