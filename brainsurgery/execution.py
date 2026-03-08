from __future__ import annotations

import logging
from typing import Any

from .transform import TransformControl, apply_transform

logger = logging.getLogger("brainsurgery")


def execute_transforms(
    transforms: list[Any],
    state_dict_provider: Any,
    *,
    interactive: bool,
) -> bool:
    total = len(transforms)

    for transform_index, transform in enumerate(transforms, start=1):
        if interactive:
            logger.info(
                "Interactive procedure %d/%d: positioning instruments for %s",
                transform_index,
                total,
                type(transform.spec).__name__,
            )
        else:
            logger.info(
                "Procedure %d/%d: positioning instruments for %s",
                transform_index,
                total,
                type(transform.spec).__name__,
            )

        transform_result = apply_transform(transform, state_dict_provider)

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

        if transform_result.control != TransformControl.CONTINUE:
            logger.info("%s requested orderly exit", transform_result.name)
            return False

    return True

