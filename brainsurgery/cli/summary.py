import logging
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf

logger = logging.getLogger("brainsurgery")


def build_raw_plan(
    inputs: Any,
    output: Any,
    transforms: list[dict[str, Any]],
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "inputs": inputs,
        "transforms": transforms,
    }
    if output is not None:
        raw["output"] = output
    return raw


def _derive_summary_path(written_path: str | Path | None) -> Path:
    if written_path is None:
        return Path("brainsurgery-executed-plan.yaml")

    path = Path(written_path)

    if path.exists() and path.is_dir():
        return path / "executed-plan.yaml"

    if path.suffix:
        return path.with_suffix(f"{path.suffix}.executed.yaml")

    return path.parent / f"{path.name}.executed.yaml"


def write_executed_plan_summary(
    *,
    inputs: Any,
    output: Any,
    transforms: list[dict[str, Any]],
    destination: Path | None,
) -> None:
    summary_doc = build_raw_plan(inputs=inputs, output=output, transforms=transforms)

    if destination is None:
        typer.echo(OmegaConf.to_yaml(summary_doc))
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(summary_doc), f=str(destination))
    logger.info("Executed plan summary written to %s", destination)
