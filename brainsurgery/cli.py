from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf

from .arena import ArenaError, SegmentedFileBackedArena
from .model import parse_shard_size
from .plan import compile_plan
from .providers import ArenaStateDictProvider, InMemoryStateDictProvider
from .transform import TransformControl, apply_transform

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("brainsurgery")
_ALLOWED_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}

app = typer.Typer(help="Brain surgery CLI.")

_PATH_TOKEN_RE = re.compile(
    r"""
    ([^. \[\]]+)   # plain key
    |
    \[(\d+)\]      # numeric list index
    """,
    re.VERBOSE,
)

_INTERACTIVE_SPECIAL_TRANSFORMS = {"help", "exit"}


def configure_logging(log_level: str) -> None:
    level_name = log_level.strip().lower()
    if level_name not in _ALLOWED_LOG_LEVELS:
        raise typer.BadParameter(
            f"log-level must be one of: {', '.join(sorted(_ALLOWED_LOG_LEVELS))}"
        )

    level = getattr(logging, level_name.upper())
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


def is_yaml_file_arg(token: str) -> bool:
    path = Path(token)
    return path.suffix.lower() in {".yaml", ".yml"} and path.is_file()


def is_yaml_like_arg(token: str) -> bool:
    return Path(token).suffix.lower() in {".yaml", ".yml"}


def parse_override_path(path: str) -> list[str | int]:
    if not path:
        raise ValueError("override key must not be empty")

    tokens: list[str | int] = []
    pos = 0

    while pos < len(path):
        match = _PATH_TOKEN_RE.match(path, pos)
        if match is None:
            raise ValueError(f"invalid override path syntax: {path!r}")

        key, index = match.groups()
        if key is not None:
            tokens.append(key)
        else:
            tokens.append(int(index))

        pos = match.end()
        if pos < len(path):
            if path[pos] == ".":
                pos += 1
            elif path[pos] == "[":
                pass
            else:
                raise ValueError(f"invalid override path syntax: {path!r}")

    return tokens


def parse_override_value(raw_value: str) -> Any:
    # Use OmegaConf/YAML parsing for numbers, booleans, lists, dicts, quoted strings.
    # Fall back to the raw string for literals like *all that YAML would treat specially.
    try:
        parsed = OmegaConf.create({"_": raw_value})
        container = OmegaConf.to_container(parsed, resolve=True)
        assert isinstance(container, dict)
        return container["_"]
    except Exception:
        return raw_value


def build_override_fragment(tokens: list[str | int], value: Any) -> Any:
    node: Any = value

    for token in reversed(tokens):
        if isinstance(token, int):
            arr = [None] * (token + 1)
            arr[token] = node
            node = arr
        else:
            node = {token: node}

    return node


def deep_merge_lists(base: list[Any], patch: list[Any]) -> list[Any]:
    size = max(len(base), len(patch))
    out: list[Any] = []

    for i in range(size):
        has_base = i < len(base)
        has_patch = i < len(patch)

        if has_patch and patch[i] is not None:
            if has_base and isinstance(base[i], dict) and isinstance(patch[i], dict):
                out.append(deep_merge_dicts(base[i], patch[i]))
            elif has_base and isinstance(base[i], list) and isinstance(patch[i], list):
                out.append(deep_merge_lists(base[i], patch[i]))
            else:
                out.append(patch[i])
        elif has_base:
            out.append(base[i])
        else:
            out.append(None)

    return out


def deep_merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)

    for key, patch_value in patch.items():
        if key not in out:
            out[key] = patch_value
            continue

        base_value = out[key]
        if isinstance(base_value, dict) and isinstance(patch_value, dict):
            out[key] = deep_merge_dicts(base_value, patch_value)
        elif isinstance(base_value, list) and isinstance(patch_value, list):
            out[key] = deep_merge_lists(base_value, patch_value)
        else:
            out[key] = patch_value

    return out


def apply_override(config: dict[str, Any], token: str) -> dict[str, Any]:
    if "=" not in token:
        raise ValueError("override must have the form key=value")

    raw_path, raw_value = token.split("=", 1)
    tokens = parse_override_path(raw_path)
    value = parse_override_value(raw_value)
    fragment = build_override_fragment(tokens, value)

    if not isinstance(fragment, dict):
        raise ValueError(f"top-level override must produce a mapping: {token!r}")

    return deep_merge_dicts(config, fragment)


def load_cli_config(tokens: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for token in tokens:
        if is_yaml_file_arg(token):
            loaded = OmegaConf.to_container(OmegaConf.load(token), resolve=True)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise typer.BadParameter(f"YAML file must contain a mapping: {token}")
            merged = deep_merge_dicts(merged, loaded)
            continue

        if is_yaml_like_arg(token):
            raise typer.BadParameter(f"YAML file does not exist: {token}")

        try:
            merged = apply_override(merged, token)
        except Exception as exc:
            raise typer.BadParameter(f"Invalid override {token!r}: {exc}") from exc

    return merged


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


def parse_transform_block(block: str) -> list[dict[str, Any]]:
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

                try:
                    return parse_transform_block("\n".join(lines))
                except ValueError as exc:
                    logger.error("Interactive transform rejected: %s", exc)
                    typer.echo("Try again.")
                    break

            lines.append(line)
            prompt = "... "


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


def derive_summary_path(written_path: str | Path | None) -> Path:
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


def execute_transforms(
    transforms: list[Any],
    state_dict_provider: Any,
    *,
    interactive: bool,
) -> bool:
    """
    Execute transforms in order.

    Returns True if execution should continue.
    Returns False if a transform requested orderly exit.
    """
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

        if not transform_result.control == TransformControl.CONTINUE:
            logger.info(
                "%s requested orderly exit",
                transform_result.name,
            )
            return False

    return True


@app.command()
def run(
    config_items: list[str] = typer.Argument(
        None,
        help=(
            "Zero or more YAML files and/or overrides in any order. Existing .yaml/.yml "
            "files are loaded and merged; everything else is treated as an override."
        ),
    ),
    shard_size: str = typer.Option("5GB", help="Default shard size for directory outputs"),
    num_workers: int = typer.Option(8, help="Max number of parallel I/O workers"),
    provider: str = typer.Option("inmemory", help="State-dict provider: inmemory or arena"),
    arena_root: Path = typer.Option(
        Path(".brainsurgery"),
        help="Arena directory when using the arena provider",
    ),
    arena_segment_size: str = typer.Option(
        "1GB",
        help="Arena segment size, e.g. 1GB, 4GB, 512MB",
    ),
    interactive: bool = typer.Option(
        False,
        "-i",
        "--interactive",
        help="Run configured transforms, then enter an interactive prompt for additional transforms.",
    ),
    summarize: bool = typer.Option(
        True,
        "-s",
        "--summarize/--no-summarize",
        help="Write a YAML summary of the actually executed plan.",
    ),
    summarize_path: Path | None = typer.Option(
        None,
        help="Path for executed plan summary YAML. If not given, the summary is printed to standard output.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Log level: debug, info, warning, error, critical",
    ),
) -> None:
    """Load a plan, execute it, and save the rewritten output checkpoint."""
    configure_logging(log_level)
    raw_plan = load_cli_config(config_items or [])

    logger.info(
        "Scrubbing in. Surgical plan assembled from %d config item(s)",
        len(config_items or []),
    )
    surgery_plan = compile_plan(raw_plan)
    logger.info(
        "Surgical plan ready: %d brain(s) prepped, %d procedure(s) scheduled, preservation %s",
        len(surgery_plan.inputs),
        len(surgery_plan.transforms),
        surgery_plan.output.path if surgery_plan.output else None,
    )

    provider_name = provider.strip().lower()

    try:
        if provider_name == "inmemory":
            state_dict_provider = InMemoryStateDictProvider(
                surgery_plan.inputs,
                max_io_workers=num_workers,
            )
        elif provider_name == "arena":
            segment_size_bytes = parse_shard_size(arena_segment_size)
            if segment_size_bytes is None:
                raise typer.BadParameter("arena-segment-size must not be 'none'")

            arena = SegmentedFileBackedArena(
                arena_root,
                segment_size_bytes=segment_size_bytes,
            )
            state_dict_provider = ArenaStateDictProvider(
                surgery_plan.inputs,
                arena=arena,
                max_io_workers=num_workers,
            )
        else:
            raise typer.BadParameter("provider must be either 'inmemory' or 'arena'")
    except ArenaError as exc:
        raise typer.BadParameter(str(exc)) from exc

    executed_transforms: list[dict[str, Any]] = []
    written_path: str | Path | None = None

    try:
        should_continue = True

        for raw_transform, compiled_transform in zip(
            normalize_transform_specs(raw_plan.get("transforms")),
            surgery_plan.transforms,
            strict=False,
        ):
            logger.info(
                "Procedure %d/%d: positioning instruments for %s",
                len(executed_transforms) + 1,
                len(surgery_plan.transforms),
                type(compiled_transform.spec).__name__,
            )
            transform_result = apply_transform(compiled_transform, state_dict_provider)
            logger.info(
                "Procedure %d/%d complete: %s affected %d site(s)",
                len(executed_transforms) + 1,
                len(surgery_plan.transforms),
                transform_result.name,
                transform_result.count,
            )

            executed_transforms.append(raw_transform)

            if not transform_result.control == TransformControl.CONTINUE:
                logger.info("%s requested orderly exit", transform_result.name)
                should_continue = False
                break

        if should_continue and interactive:
            logger.info("Entering interactive mode after configured procedures")

            while True:
                extra_specs = prompt_interactive_transform()
                if extra_specs is None:
                    logger.info("Interactive session complete")
                    break

                interactive_raw_plan = build_raw_plan(
                    inputs=raw_plan.get("inputs", []),
                    output=raw_plan.get("output"),
                    transforms=extra_specs,
                )

                try:
                    interactive_plan = compile_plan(interactive_raw_plan)
                except Exception as exc:
                    logger.error("Could not compile interactive transform(s): %s", exc)
                    continue

                for raw_transform, compiled_transform in zip(
                    extra_specs,
                    interactive_plan.transforms,
                    strict=False,
                ):
                    logger.info(
                        "Interactive procedure %d/%d: positioning instruments for %s",
                        extra_specs.index(raw_transform) + 1,
                        len(interactive_plan.transforms),
                        type(compiled_transform.spec).__name__,
                    )
                    transform_result = apply_transform(compiled_transform, state_dict_provider)
                    logger.info(
                        "Interactive procedure %d/%d complete: %s affected %d site(s)",
                        extra_specs.index(raw_transform) + 1,
                        len(interactive_plan.transforms),
                        transform_result.name,
                        transform_result.count,
                    )

                    executed_transforms.append(raw_transform)

                    if transform_result.control == TransformControl.EXIT:
                        logger.info("%s requested orderly exit", transform_result.name)
                        should_continue = False
                        break

                if not should_continue:
                    logger.info("Leaving interactive mode")
                    break

        if surgery_plan.output is None:
            logger.info("No preservation requested; concluding operation without closure")
        else:
            written_path = state_dict_provider.save_output(
                surgery_plan,
                default_shard_size=shard_size,
                max_io_workers=num_workers,
            )
            logger.info("Operation complete. Brain preserved at %s", written_path)

        if summarize:
            write_executed_plan_summary(
                inputs=raw_plan.get("inputs", []),
                output=raw_plan.get("output"),
                transforms=executed_transforms,
                destination=summarize_path,
            )

    finally:
        state_dict_provider.close()


if __name__ == "__main__":
    app()
