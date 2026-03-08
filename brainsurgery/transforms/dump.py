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
    select_tensor,
)


class DumpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class DumpSpec(UnarySpec):
    pass


class DumpTransform(UnaryTransform[DumpSpec]):
    name = "dump"
    error_type = DumpTransformError
    spec_type = DumpSpec
    progress_desc = None

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            parse_slice(target_ref.slice_spec)

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

        print(json.dumps(tree, indent=2, sort_keys=True))
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
