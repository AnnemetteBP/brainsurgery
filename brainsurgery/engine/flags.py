from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RuntimeFlags:
    dry_run: bool = False
    verbose: bool = False


_runtime_flags = RuntimeFlags()


def get_runtime_flags() -> RuntimeFlags:
    return _runtime_flags


def set_runtime_flag(flag_name: str, flag_value: bool) -> RuntimeFlags:
    global _runtime_flags
    if flag_name == "dry_run":
        _runtime_flags = replace(_runtime_flags, dry_run=flag_value)
        return _runtime_flags
    if flag_name == "verbose":
        _runtime_flags = replace(_runtime_flags, verbose=flag_value)
        return _runtime_flags
    raise ValueError(f"unknown runtime flag: {flag_name!r}")


def reset_runtime_flags() -> RuntimeFlags:
    global _runtime_flags
    _runtime_flags = RuntimeFlags()
    return _runtime_flags


__all__ = [
    "RuntimeFlags",
    "get_runtime_flags",
    "set_runtime_flag",
    "reset_runtime_flags",
]
