from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer

from ..expressions import get_assert_expr_help, get_assert_expr_names
from ..transform import (
    StateDictProvider,
    TypedTransform,
    TransformControl,
    TransformError,
    TransformResult,
    get_transform,
    list_transforms,
    register_transform,
)


class HelpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class HelpSpec:
    command: str | None = None
    subcommand: str | None = None

    def collect_models(self) -> set[str]:
        return set()


class HelpTransform(TypedTransform[HelpSpec]):
    name = "help"
    error_type = HelpTransformError
    spec_type = HelpSpec
    help_text = (
        "Shows help for commands and assert expressions.\n"
        "\n"
        "Run without arguments to list all commands. You can request help for a specific "
        "command or, for 'assert', for an individual expression operator.\n"
        "\n"
        "Examples:\n"
        "  help\n"
        "  help: copy\n"
        "  help: assert\n"
        "  help: { assert: equal }"
    )

    def compile(self, payload: Any, default_model: str | None) -> HelpSpec:
        del default_model

        if payload is None:
            return HelpSpec(command=None, subcommand=None)

        if isinstance(payload, str):
            cmd = payload.strip()
            if not cmd:
                raise HelpTransformError("help payload must be a non-empty string")
            return HelpSpec(command=cmd, subcommand=None)

        if isinstance(payload, dict):
            if not payload:
                return HelpSpec(command=None, subcommand=None)

            if len(payload) != 1:
                raise HelpTransformError("help mapping payload must have exactly one key")

            command, subpayload = next(iter(payload.items()))
            if not isinstance(command, str) or not command.strip():
                raise HelpTransformError("help command must be a non-empty string")

            command = command.strip()

            if subpayload is None:
                return HelpSpec(command=command, subcommand=None)

            if not isinstance(subpayload, str) or not subpayload.strip():
                raise HelpTransformError(
                    "help subcommand must be a non-empty string when provided"
                )

            return HelpSpec(command=command, subcommand=subpayload.strip())

        raise HelpTransformError("help payload must be empty, a string, or a single-key mapping")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        del provider

        typed = self.require_spec(spec)

        if typed.command is None:
            self._print_all_commands()
        elif typed.command == "assert":
            if typed.subcommand is None:
                self._print_assert_help()
            else:
                self._print_assert_expr_help(typed.subcommand)
        else:
            if typed.subcommand is not None:
                raise HelpTransformError(
                    f"command {typed.command!r} does not support subcommand help"
                )
            self._print_command_help(typed.command)

        return TransformResult(
            name=self.name,
            count=0,
            control=TransformControl.CONTINUE,
        )

    def infer_output_model(self, spec: object) -> str:
        del spec
        raise HelpTransformError("help does not infer an output model")

    def _print_all_commands(self) -> None:
        typer.echo("Available commands:")
        for name in list_transforms():
            typer.echo(f"  {name}")
        typer.echo("")
        typer.echo("For help on a specific command, run: help: <command>")
        typer.echo("For help on a specific assert expression, run: help: { assert: <expr> }")

    def _print_command_help(self, command_name: str) -> None:
        try:
            transform = get_transform(command_name)
        except TransformError as exc:
            raise HelpTransformError(f"unknown command: {command_name}") from exc

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

    def _print_assert_help(self) -> None:
        try:
            transform = get_transform("assert")
        except TransformError:
            transform = None

        typer.echo("Command: assert")
        if transform is not None:
            help_text = getattr(transform, "help_text", None)
            if help_text:
                typer.echo(help_text)

        typer.echo("Supported assert expression operators:")
        for name in get_assert_expr_names():
            meta = get_assert_expr_help(name)
            if meta.description:
                typer.echo(f"  {name}: {meta.description}")
            else:
                typer.echo(f"  {name}")

        typer.echo("")
        typer.echo("For help on a specific assert expression, run: help: { assert: <expr> }")

    def _print_assert_expr_help(self, expr_name: str) -> None:
        meta = get_assert_expr_help(expr_name)

        typer.echo(f"Assert expression: {meta.name}")
        typer.echo(f"Payload: {meta.payload_kind}")

        if meta.description:
            typer.echo(meta.description)

        required = sorted(meta.required_keys or [])
        allowed = sorted(meta.allowed_keys or [])
        optional = [key for key in allowed if key not in required]

        if meta.required_keys is not None:
            if required:
                typer.echo("Required keys:")
                for key in required:
                    typer.echo(f"  - {key}")
            else:
                typer.echo("Required keys: none")

        if meta.allowed_keys is not None:
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
