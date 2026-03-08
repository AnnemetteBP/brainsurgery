from __future__ import annotations

import logging
from typing import Any, Iterable

from .transform import CompiledTransform, TransformControl, apply_transform

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
            The raw transform specs that actually executed.
    """
    pair_list = list(pairs)
    total = len(pair_list)
    executed_raw_transforms: list[dict[str, Any]] = []

    for transform_index, (raw_transform, compiled_transform) in enumerate(pair_list, start=1):
        if interactive:
            logger.info(
                "Interactive procedure %d/%d: positioning instruments for %s",
                transform_index,
                total,
                type(compiled_transform.spec).__name__,
            )
        else:
            logger.info(
                "Procedure %d/%d: positioning instruments for %s",
                transform_index,
                total,
                type(compiled_transform.spec).__name__,
            )

        transform_result = apply_transform(compiled_transform, state_dict_provider)

        if interactive:
            logger.info(
                "Interactive procedure %d/%d complete: %s affected %d site(s)",
                transform_index,
                total,
                transform_result.name,
                transform_result.count,
            )
        else:
            logger.info(
                "Procedure %d/%d complete: %s affected %d site(s)",
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
