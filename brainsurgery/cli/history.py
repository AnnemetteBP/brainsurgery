import atexit
import logging
from pathlib import Path

logger = logging.getLogger("brainsurgery")

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None


_HISTORY_FILE = Path.home() / ".brainsurgery_history"
_HISTORY_LENGTH = 1000


def _configure_history() -> None:
    if readline is None:
        return

    try:
        readline.parse_and_bind("set editing-mode emacs")
    except Exception:
        pass

    try:
        if _HISTORY_FILE.exists():
            readline.read_history_file(str(_HISTORY_FILE))
    except Exception:
        logger.debug("Could not read history file %s", _HISTORY_FILE, exc_info=True)

    try:
        readline.set_history_length(_HISTORY_LENGTH)
    except Exception:
        pass

    def _write_history() -> None:
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(_HISTORY_FILE))
        except Exception:
            logger.debug("Could not write history file %s", _HISTORY_FILE, exc_info=True)

    atexit.register(_write_history)


def _add_history_entry(entry: str) -> None:
    if readline is None:
        return

    text = entry.strip()
    if not text:
        return

    try:
        current_length = readline.get_current_history_length()
        if current_length > 0:
            last = readline.get_history_item(current_length)
            if last == text:
                return
        readline.add_history(text)
    except Exception:
        logger.debug("Could not add history entry", exc_info=True)
