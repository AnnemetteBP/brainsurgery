from __future__ import annotations

from typing import Any, Callable

import torch

from .mappings import ResolvedMapping, require_dest_present, resolve_name_mappings
from .refs import TensorRef, format_tensor_ref, must_model, parse_slice, select_tensor
from .transform_types import StateDictProvider, TransformError


def resolve_target_names(
    *,
    target_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
    match_names: Callable[..., list[str]],
    error_type: type[TransformError],
) -> list[str]:
    model = must_model(target_ref)
    sd = provider.get_state_dict(model)

    try:
        matches = match_names(
            expr=target_ref.expr,
            names=sd.keys(),
            op_name=op_name,
            role="target",
        )
    except TransformError as exc:
        raise error_type(str(exc)) from exc

    if not matches:
        raise error_type(f"{op_name} matched zero tensors: {format_tensor_ref(target_ref)}")

    return matches


def resolve_single_tensor(
    ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
    resolve_names: Any,
    error_type: type[TransformError],
) -> torch.Tensor:
    model = must_model(ref)
    sd = provider.get_state_dict(model)
    matches = resolve_names(ref, provider, op_name=op_name)

    if len(matches) == 0:
        raise error_type(f"{op_name} failed: {format_tensor_ref(ref)} matched zero tensors")
    if len(matches) != 1:
        raise error_type(
            f"{op_name} failed: {format_tensor_ref(ref)} matched {len(matches)} tensors, expected 1"
        )

    tensor = sd[matches[0]]
    slice_spec = parse_slice(ref.slice_spec) if ref.slice_spec else None
    return select_tensor(tensor, slice_spec)


def resolve_tensors(
    ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
    resolve_names: Any,
) -> list[tuple[TensorRef, torch.Tensor]]:
    model = must_model(ref)
    sd = provider.get_state_dict(model)
    matches = resolve_names(ref, provider, op_name=op_name)
    slice_spec = parse_slice(ref.slice_spec) if ref.slice_spec else None
    resolved: list[tuple[TensorRef, torch.Tensor]] = []

    for name in matches:
        resolved_ref = TensorRef(model=model, expr=name, slice_spec=ref.slice_spec)
        resolved.append((resolved_ref, select_tensor(sd[name], slice_spec)))

    return resolved


def resolve_tensor_mappings(
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
    error_type: type[TransformError],
) -> list[tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]]:
    try:
        mappings = resolve_name_mappings(
            from_ref=from_ref,
            to_ref=to_ref,
            provider=provider,
            op_name=op_name,
        )
        require_dest_present(mappings=mappings, provider=provider, op_name=op_name)
    except TransformError as exc:
        raise error_type(str(exc)) from exc

    resolved: list[tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]] = []
    for item in mappings:
        resolved.append(resolve_mapping_tensors(item, from_ref=from_ref, to_ref=to_ref, provider=provider))
    return resolved


def resolve_mapping_tensors(
    item: ResolvedMapping,
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
) -> tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)

    left_ref = TensorRef(model=item.src_model, expr=item.src_name, slice_spec=from_ref.slice_spec)
    right_ref = TensorRef(model=item.dst_model, expr=item.dst_name, slice_spec=to_ref.slice_spec)
    left = select_tensor(src_sd[item.src_name], item.src_slice)
    right = select_tensor(dst_sd[item.dst_name], item.dst_slice)
    return left_ref, left, right_ref, right
