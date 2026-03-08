from __future__ import annotations

import json
from dataclasses import dataclass

from ..matching import StructuredPathError, StructuredPathMatcher
from ..transform import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    register_transform,
    require_expr,
    validate_payload_keys,
)


class DumpTransformError(TransformError):
    pass


_MATCHER = StructuredPathMatcher()


@dataclass(frozen=True)
class DumpSpec:
    target_ref: TensorRef


class DumpTransform(BaseTransform):
    name = "dump"

    def compile(self, payload: dict, default_model: str | None) -> DumpSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"target"},
            required_keys={"target"},
        )

        raw_target = require_expr(payload, op_name=self.name, key="target")
        target_ref = parse_model_expr(raw_target, default_model=default_model)

        if target_ref.slice_spec is not None:
            raise DumpTransformError("dump target must not be sliced")

        assert target_ref.model is not None
        return DumpSpec(target_ref=target_ref)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, DumpSpec):
            raise DumpTransformError(f"dump received wrong spec type: {type(spec).__name__}")

        targets = resolve_dump_targets(spec, provider)
        tree = build_dump_tree(spec, targets, provider)
        print(json.dumps(tree, indent=2))
        return TransformResult(name=self.name, count=len(targets))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, DumpSpec):
            raise DumpTransformError(f"dump received wrong spec type: {type(spec).__name__}")

        model = spec.target_ref.model
        if model is None:
            raise DumpTransformError("dump output model missing")
        return model


def resolve_dump_targets(spec: DumpSpec, provider: StateDictProvider) -> list[str]:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    if isinstance(spec.target_ref.expr, str):
        import re

        try:
            matches = sorted(
                name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name)
            )
        except re.error as exc:
            raise DumpTransformError(
                f"dump invalid target regex {spec.target_ref.expr!r}: {exc}"
            ) from exc
    elif isinstance(spec.target_ref.expr, list):
        try:
            matches = sorted(
                name
                for name in sd.keys()
                if _MATCHER.match(spec.target_ref.expr, name) is not None
            )
        except StructuredPathError as exc:
            raise DumpTransformError(f"dump invalid structured target pattern: {exc}") from exc
    else:
        raise DumpTransformError(
            f"dump target expression has invalid type: {type(spec.target_ref.expr).__name__}"
        )

    if not matches:
        raise DumpTransformError(f"dump matched zero tensors: {format_target_ref(spec.target_ref)}")

    return matches


def build_dump_tree(
    spec: DumpSpec,
    targets: list[str],
    provider: StateDictProvider,
) -> dict[str, object]:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    root: dict[str, object] = {}
    for name in targets:
        tensor = sd[name]
        insert_path(root, split_key(name), list(tensor.shape), full_name=name)
    return root


def split_key(name: str) -> list[str | int]:
    parts: list[str | int] = []
    for part in name.split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return parts


def insert_path(
    root: dict[str, object],
    parts: list[str | int],
    value: list[int],
    *,
    full_name: str,
) -> None:
    if not parts:
        raise DumpTransformError(f"dump encountered empty key path for tensor {full_name!r}")

    node: dict[str, object] | list[object] = root

    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        next_part = None if is_last else parts[i + 1]

        if isinstance(part, str):
            if not isinstance(node, dict):
                raise DumpTransformError(
                    f"dump path conflict at {full_name!r}: expected object before {part!r}"
                )

            if is_last:
                existing = node.get(part)
                if existing is not None and existing != value:
                    raise DumpTransformError(
                        f"dump path conflict at {full_name!r}: leaf collides with existing node"
                    )
                node[part] = value
                return

            existing = node.get(part)
            expected_child: dict[str, object] | list[object] = (
                [] if isinstance(next_part, int) else {}
            )

            if existing is None:
                node[part] = expected_child
                node = expected_child
                continue

            if isinstance(existing, type(expected_child)):
                node = existing
                continue

            raise DumpTransformError(
                f"dump path conflict at {full_name!r}: incompatible node at {part!r}"
            )

        else:
            if not isinstance(node, list):
                raise DumpTransformError(
                    f"dump path conflict at {full_name!r}: expected array before index {part}"
                )

            while len(node) <= part:
                node.append(None)

            if is_last:
                existing = node[part]
                if existing is not None and existing != value:
                    raise DumpTransformError(
                        f"dump path conflict at {full_name!r}: leaf collides with existing node"
                    )
                node[part] = value
                return

            existing = node[part]
            expected_child = [] if isinstance(next_part, int) else {}

            if existing is None:
                node[part] = expected_child
                node = expected_child
                continue

            if isinstance(existing, type(expected_child)):
                node = existing
                continue

            raise DumpTransformError(
                f"dump path conflict at {full_name!r}: incompatible node at index {part}"
            )

    raise DumpTransformError(f"dump failed to insert tensor {full_name!r}")


def format_target_ref(ref: TensorRef) -> str:
    model = must_model(ref)
    if isinstance(ref.expr, str):
        expr = ref.expr
    elif isinstance(ref.expr, list):
        expr = "[" + ", ".join(repr(part) for part in ref.expr) + "]"
    else:
        expr = repr(ref.expr)

    if ref.slice_spec is None:
        return f"{model}::{expr}"
    return f"{model}::{expr}::{ref.slice_spec}"


register_transform(DumpTransform())
