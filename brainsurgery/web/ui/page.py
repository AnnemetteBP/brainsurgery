from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_PATH = _STATIC_DIR / "index.html"


_HTML_PAGE = _INDEX_PATH.read_text(encoding="utf-8")


__all__ = ["_HTML_PAGE", "_STATIC_DIR"]
