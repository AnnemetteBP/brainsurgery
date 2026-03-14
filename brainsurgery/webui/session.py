from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any


@dataclass
class SessionState:
    provider: Any
    lock: threading.Lock
    upload_root: Path
    executed_transforms: list[dict[str, Any]] = field(default_factory=list)
    default_model: str | None = None


__all__ = ["SessionState"]
