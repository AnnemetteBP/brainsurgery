from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any


@dataclass
class SessionState:
    provider: Any
    lock: threading.Lock
    upload_root: Path


__all__ = ["SessionState"]
