from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expressions import get_assert_expr_help, get_assert_expr_names
from ..core import TransformError
from ..core import TypedTransform, TransformControl, TransformResult, get_transform, list_transforms, register_transform
from ..core import StateDictProvider
from ..engine.frontend import emit_line


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

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False

    def completion_payload_start_candidates(self, prefix_text: str) -> list[str] | None:
        candidates = [name for name in list_transforms() if name.startswith(prefix_text)]
        if not prefix_text:
            return candidates + ["{ "]
        return candidates

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor
        candidate = "assert: "
        if candidate.startswith(prefix_text):
            return [candidate]
        return []

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key == "help":
            return [name for name in list_transforms() if name.startswith(prefix_text)]
        if value_key == "assert":
            return [name for name in get_assert_expr_names() if name.startswith(prefix_text)]
        return None

    def completion_committed_next_candidates(self, value_key: str | None) -> list[str] | None:
        if value_key in {"help", "assert"}:
            return ["}"]
        return None

    def _print_all_commands(self) -> None:
        emit_line("Available commands:")
        for name in list_transforms():
            emit_line(f"  {name}")
        emit_line("")
        emit_line("For help on a specific command, run: help: <command>")
        emit_line("For help on a specific assert expression, run: help: { assert: <expr> }")

    def _print_command_help(self, command_name: str) -> None:
        try:
            transform = get_transform(command_name)
        except TransformError as exc:
            raise HelpTransformError(f"unknown command: {command_name}") from exc

        allowed_keys = getattr(transform, "allowed_keys", None)
        required_keys = getattr(transform, "required_keys", None)
        help_text = getattr(transform, "help_text", None)

        emit_line(f"Command: {command_name}")

        if help_text:
            emit_line(help_text)

        if allowed_keys is None and required_keys is None:
            emit_line("Key metadata: unavailable")
            return

        allowed = sorted(allowed_keys or set())
        required = sorted(required_keys or set())

        if required:
            emit_line("Required keys:")
            for key in required:
                emit_line(f"  - {key}")
        else:
            emit_line("Required keys: none")

        optional = [key for key in allowed if key not in required]
        if optional:
            emit_line("Optional keys:")
            for key in optional:
                emit_line(f"  - {key}")
        else:
            emit_line("Optional keys: none")

        if allowed:
            emit_line("All allowed keys:")
            for key in allowed:
                emit_line(f"  - {key}")
        else:
            emit_line("All allowed keys: none")

    def _print_assert_help(self) -> None:
        try:
            transform = get_transform("assert")
        except TransformError:
            transform = None

        emit_line("Command: assert")
        if transform is not None:
            help_text = getattr(transform, "help_text", None)
            if help_text:
                emit_line(help_text)

        emit_line("Supported assert expression operators:")
        for name in get_assert_expr_names():
            meta = get_assert_expr_help(name)
            if meta.description:
                emit_line(f"  {name}: {meta.description}")
            else:
                emit_line(f"  {name}")

        emit_line("")
        emit_line("For help on a specific assert expression, run: help: { assert: <expr> }")

    def _print_assert_expr_help(self, expr_name: str) -> None:
        meta = get_assert_expr_help(expr_name)

        emit_line(f"Assert expression: {meta.name}")
        emit_line(f"Payload: {meta.payload_kind}")

        if meta.description:
            emit_line(meta.description)

        required = sorted(meta.required_keys or [])
        allowed = sorted(meta.allowed_keys or [])
        optional = [key for key in allowed if key not in required]

        if meta.required_keys is not None:
            if required:
                emit_line("Required keys:")
                for key in required:
                    emit_line(f"  - {key}")
            else:
                emit_line("Required keys: none")

        if meta.allowed_keys is not None:
            if optional:
                emit_line("Optional keys:")
                for key in optional:
                    emit_line(f"  - {key}")
            else:
                emit_line("Optional keys: none")

            if allowed:
                emit_line("All allowed keys:")
                for key in allowed:
                    emit_line(f"  - {key}")
            else:
                emit_line("All allowed keys: none")










register_transform(HelpTransform())
