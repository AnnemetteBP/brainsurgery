from __future__ import annotations

import logging
from typing import Any, Iterable

from ..core import CompiledTransform, TransformControl, _apply_transform

logger = logging.getLogger("brainsurgery")


def execute_transform_pairs(
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

    for transform_index, (raw_transform, compiled_transform) in enumerate(pair_list, start=1):
        logger.info(
            "%s %d/%d: positioning instruments for %s",
            procedure_label,
            transform_index,
            total,
            type(compiled_transform.spec).__name__,
        )

        try:
            transform_result = _apply_transform(compiled_transform, state_dict_provider)
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
            return False, executed_raw_transforms

    return True, executed_raw_transforms
