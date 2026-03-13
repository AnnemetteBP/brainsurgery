from __future__ import annotations

import os
from pathlib import Path


def complete_filesystem_paths(
    prefix_text: str,
    *,
    include_files: bool = True,
    include_dirs: bool = True,
    limit: int = 200,
) -> list[str]:
    quote = ""
    raw = prefix_text
    if raw.startswith(("'", '"')):
        quote = raw[0]
        raw = raw[1:]

    typed = raw
    expanded = os.path.expanduser(typed)

    if typed.endswith("/"):
        display_dir = typed
        base_dir = Path(expanded)
        fragment = ""
    elif "/" in typed:
        display_dir, fragment = typed.rsplit("/", 1)
        display_dir = f"{display_dir}/"
        base_dir = Path(expanded).parent
    else:
        display_dir = ""
        fragment = typed
        base_dir = Path(".")

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    try:
        entries = sorted(base_dir.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name))
    except OSError:
        return []

    out: list[str] = []
    for entry in entries:
        name = entry.name
        if not name.startswith(fragment):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if is_dir and not include_dirs:
            continue
        if not is_dir and not include_files:
            continue
        candidate = f"{display_dir}{name}"
        if is_dir:
            candidate = f"{candidate}/"
        if quote:
            candidate = f"{quote}{candidate}"
        out.append(candidate)
        if len(out) >= limit:
            break

    return out
