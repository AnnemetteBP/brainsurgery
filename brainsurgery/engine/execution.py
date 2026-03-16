import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..core import (
    CompiledTransform,
    TensorRef,
    TransformControl,
    apply_transform,
    match_expr_names,
    resolve_name_mappings,
)
from .flags import get_runtime_flags
from .frontend import emit_line

logger = logging.getLogger("brainsurgery")

_LITERAL_EXPR_META = re.compile(r"[\\.^$*+?{}\[\]|()]")


@dataclass(frozen=True)
class _PreviewImpact:
    changed: set[str]
    created: set[str]
    deleted: set[str]


def _execute_transform_pairs(
    pairs: Iterable[tuple[dict[str, Any], CompiledTransform]],
    state_dict_provider: Any,
    *,
    interactive: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Execute (raw_transform, compiled_transform) pairs in order.

    Returns:
        should_continue:
            True if execution should continue.
            False if a transform requested orderly exit.

        executed_raw_transforms:
            The raw transform specs that actually executed successfully.

    Behavior:
        - In non-interactive mode, transform failures are raised.
        - In interactive mode, transform failures are logged and execution of the
          current submitted block stops, returning control to the prompt.
    """
    pair_list = list(pairs)
    total = len(pair_list)
    procedure_label = "Interactive procedure" if interactive else "Procedure"
    executed_raw_transforms: list[dict[str, Any]] = []

    session_changed: set[str] = set()
    session_created: set[str] = set()
    session_deleted: set[str] = set()
    preview_seen = False

    for transform_index, (raw_transform, compiled_transform) in enumerate(pair_list, start=1):
        logger.info(
            "%s %d/%d: positioning instruments for %s",
            procedure_label,
            transform_index,
            total,
            type(compiled_transform.spec).__name__,
        )
        flags = get_runtime_flags()
        preview_enabled = flags.preview
        impact: _PreviewImpact | None = None
        if preview_enabled:
            preview_seen = True
            try:
                impact = _preview_impact_for_transform(compiled_transform, state_dict_provider)
            except Exception as exc:
                emit_line(
                    f"preview {transform_index}/{total} {compiled_transform.transform.name}: "
                    f"could not infer impact ({exc})"
                )
            else:
                emit_line(
                    f"preview {transform_index}/{total} {compiled_transform.transform.name}: "
                    f"{_format_preview_impact(impact)}"
                )
                session_changed.update(impact.changed)
                session_created.update(impact.created)
                session_deleted.update(impact.deleted)
                if _should_skip_apply_for_preview(compiled_transform.transform.name, impact):
                    if interactive:
                        approved = _confirm_preview_apply(
                            transform_index=transform_index,
                            total=total,
                            transform_name=compiled_transform.transform.name,
                        )
                        if not approved:
                            emit_line(
                                f"preview {transform_index}/{total} "
                                f"{compiled_transform.transform.name}: no-go, apply skipped"
                            )
                            continue
                    else:
                        if flags.dry_run:
                            emit_line(
                                f"preview {transform_index}/{total} "
                                f"{compiled_transform.transform.name}: dry-run+preview, apply skipped"
                            )
                            continue

        try:
            transform_result = apply_transform(compiled_transform, state_dict_provider)
        except Exception as exc:
            if not interactive:
                raise

            logger.error(
                "%s %d/%d failed: %s",
                procedure_label,
                transform_index,
                total,
                exc,
            )
            return True, executed_raw_transforms

        logger.info(
            "%s %d/%d complete: %s affected %d site(s)",
            procedure_label,
            transform_index,
            total,
            transform_result.name,
            transform_result.count,
        )

        executed_raw_transforms.append(raw_transform)

        if transform_result.control != TransformControl.CONTINUE:
            logger.info("%s requested orderly exit", transform_result.name)
            if preview_seen:
                emit_line(
                    "preview session: "
                    f"changed[{len(session_changed)}], "
                    f"created[{len(session_created)}], "
                    f"deleted[{len(session_deleted)}]"
                )
            return False, executed_raw_transforms

    if preview_seen:
        emit_line(
            "preview session: "
            f"changed[{len(session_changed)}], "
            f"created[{len(session_created)}], "
            f"deleted[{len(session_deleted)}]"
        )

    return True, executed_raw_transforms


def execute_transform_pairs(
    pairs: Iterable[tuple[dict[str, Any], CompiledTransform]],
    state_dict_provider: Any,
    *,
    interactive: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    return _execute_transform_pairs(
        pairs,
        state_dict_provider,
        interactive=interactive,
    )


def _should_skip_apply_for_preview(transform_name: str, impact: _PreviewImpact) -> bool:
    if transform_name in {"set", "help", "dump", "diff", "assert", "exit"}:
        return False
    if transform_name in {"load", "save", "prefixes"}:
        return True
    return bool(impact.changed or impact.created or impact.deleted)


def preview_requires_confirmation(transform_name: str, impact: _PreviewImpact) -> bool:
    return _should_skip_apply_for_preview(transform_name, impact)


def _confirm_preview_apply(
    *,
    transform_index: int,
    total: int,
    transform_name: str,
) -> bool:
    prompt = (
        f"preview {transform_index}/{total} {transform_name}: go/no-go [go/no-go, default no-go]: "
    )
    while True:
        try:
            answer = input(prompt)
        except EOFError:
            return False
        normalized = answer.strip().lower()
        if normalized in {"go", "g", "yes", "y"}:
            return True
        if normalized in {"", "no-go", "nogo", "no", "n"}:
            return False
        emit_line("Please answer 'go' or 'no-go'.")


def _preview_impact_for_transform(
    compiled_transform: CompiledTransform,
    state_dict_provider: Any,
) -> _PreviewImpact:
    transform_name = compiled_transform.transform.name
    spec = compiled_transform.spec

    changed: set[str] = set()
    created: set[str] = set()
    deleted: set[str] = set()

    from_ref = _get_tensor_ref(spec, "from_ref")
    to_ref = _get_tensor_ref(spec, "to_ref")
    target_ref = _get_tensor_ref(spec, "target_ref")
    from_a_ref = _get_tensor_ref(spec, "from_a_ref")
    to_refs = _get_tensor_ref_list(spec, "to_refs")
    from_refs = _get_tensor_ref_list(spec, "from_refs")
    source_ref = _get_tensor_ref(spec, "source_ref")
    factor_a_ref = _get_tensor_ref(spec, "factor_a_ref")
    factor_b_ref = _get_tensor_ref(spec, "factor_b_ref")

    if from_ref is not None and to_ref is not None:
        mappings = resolve_name_mappings(
            from_ref=from_ref,
            to_ref=to_ref,
            provider=state_dict_provider,
            op_name=f"{transform_name}.preview",
        )
        for item in mappings:
            _classify_write(
                model=item.dst_model,
                tensor_name=item.dst_name,
                slice_spec=to_ref.slice_spec,
                provider=state_dict_provider,
                changed=changed,
                created=created,
            )
            if transform_name == "move":
                deleted.add(
                    _format_concrete_ref(item.src_model, item.src_name, from_ref.slice_spec)
                )

    if from_a_ref is not None and to_ref is not None:
        mappings = resolve_name_mappings(
            from_ref=from_a_ref,
            to_ref=to_ref,
            provider=state_dict_provider,
            op_name=f"{transform_name}.preview",
        )
        for item in mappings:
            _classify_write(
                model=item.dst_model,
                tensor_name=item.dst_name,
                slice_spec=to_ref.slice_spec,
                provider=state_dict_provider,
                changed=changed,
                created=created,
            )

    if source_ref is not None and factor_a_ref is not None:
        mappings = resolve_name_mappings(
            from_ref=source_ref,
            to_ref=factor_a_ref,
            provider=state_dict_provider,
            op_name=f"{transform_name}.preview.factor_a",
        )
        for item in mappings:
            _classify_write(
                model=item.dst_model,
                tensor_name=item.dst_name,
                slice_spec=factor_a_ref.slice_spec,
                provider=state_dict_provider,
                changed=changed,
                created=created,
            )

    if source_ref is not None and factor_b_ref is not None:
        mappings = resolve_name_mappings(
            from_ref=source_ref,
            to_ref=factor_b_ref,
            provider=state_dict_provider,
            op_name=f"{transform_name}.preview.factor_b",
        )
        for item in mappings:
            _classify_write(
                model=item.dst_model,
                tensor_name=item.dst_name,
                slice_spec=factor_b_ref.slice_spec,
                provider=state_dict_provider,
                changed=changed,
                created=created,
            )

    if target_ref is not None:
        target_names = _resolve_ref_names(
            ref=target_ref,
            provider=state_dict_provider,
            op_name=f"{transform_name}.preview.target",
            role="target",
        )
        for name in target_names:
            resolved = _format_concrete_ref(target_ref.model, name, target_ref.slice_spec)
            if transform_name == "delete":
                deleted.add(resolved)
            else:
                changed.add(resolved)

    if to_refs:
        for ref in to_refs:
            _classify_ref_write(
                ref=ref,
                provider=state_dict_provider,
                op_name=f"{transform_name}.preview.to",
                changed=changed,
                created=created,
            )

    if from_refs and to_ref is not None:
        _classify_ref_write(
            ref=to_ref,
            provider=state_dict_provider,
            op_name=f"{transform_name}.preview.to",
            changed=changed,
            created=created,
        )

    return _PreviewImpact(changed=changed, created=created, deleted=deleted)


def preview_impact_for_transform(
    compiled_transform: CompiledTransform,
    state_dict_provider: Any,
) -> _PreviewImpact:
    return _preview_impact_for_transform(compiled_transform, state_dict_provider)


def _get_tensor_ref(spec: object, attr_name: str) -> TensorRef | None:
    value = getattr(spec, attr_name, None)
    return value if isinstance(value, TensorRef) else None


def _get_tensor_ref_list(spec: object, attr_name: str) -> list[TensorRef]:
    value = getattr(spec, attr_name, None)
    if not isinstance(value, list):
        return []
    refs: list[TensorRef] = []
    for item in value:
        if isinstance(item, TensorRef):
            refs.append(item)
    return refs


def _format_concrete_ref(model: str | None, tensor_name: str, slice_spec: str | None) -> str:
    rendered = f"{model}::{tensor_name}"
    if slice_spec is None:
        return rendered
    return f"{rendered}::{slice_spec}"


def _resolve_ref_names(
    *,
    ref: TensorRef,
    provider: Any,
    op_name: str,
    role: str,
) -> list[str]:
    if ref.model is None:
        return []
    state_dict = provider.get_state_dict(ref.model)
    return match_expr_names(
        expr=ref.expr,
        names=state_dict.keys(),
        op_name=op_name,
        role=role,
    )


def _classify_write(
    *,
    model: str,
    tensor_name: str,
    slice_spec: str | None,
    provider: Any,
    changed: set[str],
    created: set[str],
) -> None:
    state_dict = provider.get_state_dict(model)
    rendered = _format_concrete_ref(model, tensor_name, slice_spec)
    if tensor_name in state_dict:
        changed.add(rendered)
    else:
        created.add(rendered)


def _classify_ref_write(
    *,
    ref: TensorRef,
    provider: Any,
    op_name: str,
    changed: set[str],
    created: set[str],
) -> None:
    model = ref.model
    if model is None:
        return
    state_dict = provider.get_state_dict(model)

    if isinstance(ref.expr, str) and _LITERAL_EXPR_META.search(ref.expr) is None:
        rendered = _format_concrete_ref(model, ref.expr, ref.slice_spec)
        if ref.expr in state_dict:
            changed.add(rendered)
        else:
            created.add(rendered)
        return

    names = _resolve_ref_names(ref=ref, provider=provider, op_name=op_name, role="destination")
    if names:
        for name in names:
            _classify_write(
                model=model,
                tensor_name=name,
                slice_spec=ref.slice_spec,
                provider=provider,
                changed=changed,
                created=created,
            )
        return
    created.add(_format_concrete_ref(model, str(ref.expr), ref.slice_spec))


def _format_preview_impact(impact: _PreviewImpact) -> str:
    parts: list[str] = []
    if impact.changed:
        parts.append(_format_preview_bucket("changed", impact.changed))
    if impact.created:
        parts.append(_format_preview_bucket("created", impact.created))
    if impact.deleted:
        parts.append(_format_preview_bucket("deleted", impact.deleted))
    if not parts:
        return "no tensor impact"
    return " | ".join(parts)


def format_preview_impact(impact: _PreviewImpact) -> str:
    return _format_preview_impact(impact)


def _format_preview_bucket(label: str, refs: set[str]) -> str:
    rendered = sorted(refs)
    return f"{label}[{len(rendered)}] " + ", ".join(rendered)
