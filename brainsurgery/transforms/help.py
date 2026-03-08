from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer

from ..transform import (
    _REGISTRY,
    StateDictProvider,
    TransformControl,
    TransformError,
    TransformResult,
    register_transform,
)


class HelpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class HelpSpec:
    command: str | None = None


class HelpTransform:
    name = "help"
    error_type = HelpTransformError
    spec_type = HelpSpec

    def compile(self, payload: Any, default_model: str | None) -> HelpSpec:
        del default_model

        if payload is None:
            return HelpSpec(command=None)

        if isinstance(payload, str):
            cmd = payload.strip()
            if not cmd:
                raise HelpTransformError("help payload must be a non-empty string")
            return HelpSpec(command=cmd)

        if isinstance(payload, dict) and not payload:
            return HelpSpec(command=None)

        raise HelpTransformError("help payload must be a string or empty")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        del provider

        if not isinstance(spec, HelpSpec):
            raise self.error_type(
                f"{self.name} expected {self.spec_type.__name__}, got {type(spec).__name__}"
            )

        if spec.command is None:
            self._print_all_commands()
        else:
            self._print_command_help(spec.command)

        return TransformResult(
            name=self.name,
            count=0,
            control=TransformControl.CONTINUE,
        )

    def _print_all_commands(self) -> None:
        typer.echo("Available commands:")
        for name in sorted(_REGISTRY):
            typer.echo(f"  {name}")
        typer.echo("")
        typer.echo("For help on a specific command, run: help: <command>")

    def _print_command_help(self, command_name: str) -> None:
        transform = _REGISTRY.get(command_name)
        if transform is None:
            raise HelpTransformError(f"unknown command: {command_name}")

        allowed_keys = getattr(transform, "allowed_keys", None)
        required_keys = getattr(transform, "required_keys", None)
        help_text = getattr(transform, "help_text", None)

        typer.echo(f"Command: {command_name}")

        if help_text:
            typer.echo(help_text)

        if allowed_keys is None and required_keys is None:
            typer.echo("Key metadata: unavailable")
            return

        allowed = sorted(allowed_keys or set())
        required = sorted(required_keys or set())

        if required:
            typer.echo("Required keys:")
            for key in required:
                typer.echo(f"  - {key}")
        else:
            typer.echo("Required keys: none")

        optional = [key for key in allowed if key not in required]
        if optional:
            typer.echo("Optional keys:")
            for key in optional:
                typer.echo(f"  - {key}")
        else:
            typer.echo("Optional keys: none")

        if allowed:
            typer.echo("All allowed keys:")
            for key in allowed:
                typer.echo(f"  - {key}")
        else:
            typer.echo("All allowed keys: none")


register_transform(HelpTransform())
