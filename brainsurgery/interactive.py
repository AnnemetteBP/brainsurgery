from __future__ import annotations

import logging
from typing import Any

import typer
from omegaconf import OmegaConf

from .history import add_history_entry

logger = logging.getLogger("brainsurgery")


def _normalize_single_transform_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        if len(raw) != 1:
            raise ValueError("each transform spec must be a mapping with exactly one key")
        return raw

    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            raise ValueError("transform name must be a non-empty string")
        return {name: {}}

    raise ValueError(
        "transform spec must be either a YAML mapping or a bare transform name"
    )


def normalize_transform_specs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, list):
        return [_normalize_single_transform_spec(item) for item in raw]

    return [_normalize_single_transform_spec(raw)]


def _rewrite_help_shorthand(block: str) -> str:
    lines = block.splitlines()
    if len(lines) != 1:
        return block

    line = lines[0].strip()
    if not line.startswith("help:"):
        return block

    parts = [part.strip() for part in line.split(":")]
    if len(parts) != 3:
        return block

    head, command, subcommand = parts
    if head != "help" or not command or not subcommand:
        return block

    return f"help: {{ {command}: {subcommand} }}"


def parse_transform_block(block: str) -> list[dict[str, Any]]:
    block = _rewrite_help_shorthand(block)

    try:
        loaded = OmegaConf.to_container(OmegaConf.create(block), resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc

    return normalize_transform_specs(loaded)


def prompt_interactive_transform() -> list[dict[str, Any]] | None:
    typer.echo("")
    typer.echo("Interactive mode.")
    typer.echo("Enter one transform as YAML, or a YAML list of transforms.")
    typer.echo("Finish input with an empty line.")
    typer.echo("Special transforms: help, exit")
    typer.echo("Example: copy: { from: ln_f.weight, to: ln_f_copy.weight }")

    while True:
        lines: list[str] = []
        prompt = "brainsurgery> "

        while True:
            try:
                line = input(prompt)
            except EOFError:
                return None

            if line.strip() == "":
                if not lines:
                    continue

                block = "\n".join(lines)
                try:
                    parsed = parse_transform_block(block)
                    add_history_entry(block)
                    return parsed
                except ValueError as exc:
                    logger.error("Interactive transform rejected: %s", exc)
                    typer.echo("Try again.")
                    break

            lines.append(line)
            prompt = "... "

