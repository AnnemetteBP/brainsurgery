from enum import Enum

from .flags import get_runtime_flags, set_runtime_flag


class RuntimeFlagLifecycleScope(Enum):
    CLI_RUN = "cli_run"
    WEBCLI_RUN = "webcli_run"
    WEBUI_SESSION = "webui_session"


def reset_runtime_flags_for_scope(scope: RuntimeFlagLifecycleScope) -> object:
    del scope
    set_runtime_flag("dry_run", False)
    set_runtime_flag("preview", False)
    set_runtime_flag("verbose", False)
    return get_runtime_flags()


__all__ = [
    "RuntimeFlagLifecycleScope",
    "reset_runtime_flags_for_scope",
]
