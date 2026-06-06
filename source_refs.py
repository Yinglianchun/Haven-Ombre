from __future__ import annotations

from pathlib import Path
from typing import Any


def source_ref_window(
    moment: dict,
    *,
    allowed_root: str = "",
    max_chars: int = 760,
    context_lines: int = 3,
) -> str:
    ref = _source_ref(moment)
    if not ref:
        return ""
    path = _safe_source_path(ref, allowed_root)
    if path is None or not path.exists() or not path.is_file():
        return ""
    start_line = _safe_int(ref.get("start_line"), 0)
    end_line = _safe_int(ref.get("end_line"), start_line)
    content_start_line = max(1, _safe_int(ref.get("content_start_line"), 1))
    if start_line <= 0 or end_line < start_line:
        return ""
    try:
        lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    except OSError:
        return ""
    start = max(content_start_line, start_line - max(0, int(context_lines)))
    end = min(len(lines), end_line + max(0, int(context_lines)))
    if start > end:
        return ""
    window = "\n".join(lines[start - 1 : end]).strip()
    return _clip_text(window, max_chars)


def _source_ref(moment: dict) -> dict[str, Any]:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    ref = meta.get("source_ref")
    return ref if isinstance(ref, dict) else {}


def _safe_source_path(ref: dict[str, Any], allowed_root: str) -> Path | None:
    raw_path = str(ref.get("path") or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    root = Path(allowed_root).resolve() if str(allowed_root or "").strip() else None
    if not candidate.is_absolute():
        if root is None:
            return None
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if root is not None and root not in (resolved, *resolved.parents):
        return None
    return resolved


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip_text(text: str, max_chars: int) -> str:
    compact = str(text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(1, int(max_chars))].rstrip() + "..."
