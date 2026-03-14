from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from ..engine import SurgeryPlan


@dataclass
class _SessionState:
    provider: Any
    lock: threading.Lock
    upload_root: Path
    plan: SurgeryPlan = field(default_factory=lambda: SurgeryPlan(inputs={}, output=None))
    default_model: str | None = None
