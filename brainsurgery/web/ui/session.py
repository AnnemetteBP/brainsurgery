from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from brainsurgery.engine import SurgeryPlan


@dataclass
class _SessionState:
    provider: Any
    lock: threading.Lock
    upload_root: Path
    plan: SurgeryPlan = field(default_factory=lambda: SurgeryPlan(inputs={}, output=None))
    default_model: str | None = None
    progress_lock: threading.Lock = field(default_factory=threading.Lock)
    progress: dict[str, Any] = field(
        default_factory=lambda: {
            "active": False,
            "iterating": False,
            "transform": None,
            "desc": None,
            "unit": "item",
            "completed": 0,
            "total": None,
            "started_at": None,
            "updated_at": None,
            "error": None,
        }
    )
