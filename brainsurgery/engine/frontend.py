from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

OutputEmitter = Callable[[str], None]

_output_emitter: OutputEmitter | None = None


def emit_line(message: str = "") -> None:
    emitter = _output_emitter
    if emitter is None:
        print(message)
        return
    emitter(message)


def set_output_emitter(emitter: OutputEmitter | None) -> OutputEmitter | None:
    global _output_emitter
    previous = _output_emitter
    _output_emitter = emitter
    return previous


@contextmanager
def use_output_emitter(emitter: OutputEmitter | None):
    previous = set_output_emitter(emitter)
    try:
        yield
    finally:
        set_output_emitter(previous)
