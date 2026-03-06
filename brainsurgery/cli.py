from __future__ import annotations

import logging
from pathlib import Path

import typer

from .model import InMemoryStateDictProvider
from .plan import load_plan
from .transform import apply_transform

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("brainsurgery")

app = typer.Typer(help="Brain surgery CLI.")


@app.command()
def run(plan: Path) -> None:
    """Load a plan, execute it, and save the rewritten output checkpoint."""
    logger.info("Scrubbing in with surgical plan %s", plan)
    surgery_plan = load_plan(plan)
    logger.info(
        "Plan loaded: %d input brains, %d transform(s), output path %s",
        len(surgery_plan.inputs),
        len(surgery_plan.transforms),
        surgery_plan.output.path,
    )
    provider = InMemoryStateDictProvider(surgery_plan.inputs)

    for transform_index, transform in enumerate(surgery_plan.transforms, start=1):
        logger.info(
            "Transform %d/%d: preparing %s",
            transform_index,
            len(surgery_plan.transforms),
            type(transform.spec).__name__,
        )
        transform_result = apply_transform(transform, provider)
        logger.info(
            "Transform %d/%d: %s complete, %d target(s) affected",
            transform_index,
            len(surgery_plan.transforms),
            transform_result.name,
            transform_result.count,
        )

    written_path = provider.save_output(surgery_plan)
    typer.echo(f"Wrote output checkpoint to {written_path}")


if __name__ == "__main__":
    app()
