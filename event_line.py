"""
event_line.py -- 事件线：跨天持续事件追踪
只记录"一天内完成不了、持续多天"的长期事件（生病就医、旅行、生理周期、学术/项目进展等）。
存于 state/event_line.json，格式化后写入 system_standard.txt 的 EVENT_LINE 缓存区块。
事件结束时由调用方（gateway）归档为记忆桶，本模块只负责从事件线移除。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import cache_activity_zone

logger = logging.getLogger("ombre_brain.event_line")

BJ = timezone(timedelta(hours=8))
MAX_ENTRIES_PER_DAY = 3
STALE_DAYS = 1


def _bj_now() -> datetime:
    return datetime.now(BJ)


def _event_line_path(state_dir: str) -> Path:
    return Path(state_dir) / "event_line.json"


class EventLine:
    def __init__(self, state_dir: str, persona_file: str):
        self._state_dir = state_dir
        self._persona_file = persona_file
        self._path = _event_line_path(state_dir)

    # ── 文件 I/O ─────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("events"), list):
                    return data
            except Exception:
                pass
        return {"events": []}

    def _save(self, data: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("EventLine save failed | %s", exc)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create_event(self, title: str, first_text: str, progress: int | None = None) -> str:
        data = self._load()
        event_id = "evt_" + uuid.uuid4().hex[:10]
        today = _bj_now().strftime("%Y-%m-%d")
        data["events"].append(
            {
                "id": event_id,
                "title": str(title or "").strip()[:60] or "未命名事件",
                "status": "active",
                "created": today,
                "last_updated": today,
                "entries": [
                    {
                        "date": today,
                        "text": str(first_text or "").strip()[:200],
                        "progress": progress,
                    }
                ],
            }
        )
        self._save(data)
        self.sync()
        logger.info("EventLine created | id=%s title=%s", event_id, title)
        return event_id

    def append_entry(
        self, event_id: str, text: str, progress: int | None = None
    ) -> tuple[bool, str]:
        data = self._load()
        today = _bj_now().strftime("%Y-%m-%d")
        for ev in data["events"]:
            if ev.get("id") == event_id:
                today_count = sum(1 for e in ev.get("entries", []) if e.get("date") == today)
                if today_count >= MAX_ENTRIES_PER_DAY:
                    return False, f"今日已记录 {MAX_ENTRIES_PER_DAY} 条，达到上限"
                ev.setdefault("entries", []).append(
                    {
                        "date": today,
                        "text": str(text or "").strip()[:200],
                        "progress": progress,
                    }
                )
                ev["last_updated"] = today
                self._save(data)
                self.sync()
                logger.info("EventLine updated | id=%s", event_id)
                return True, ""
        return False, "event_id 不存在"

    def close_event(self, event_id: str, reason: str = "") -> dict | None:
        """关闭并从事件线移除，返回事件全量数据供调用方归档为记忆桶。"""
        data = self._load()
        closed = None
        remaining = []
        for ev in data["events"]:
            if ev.get("id") == event_id:
                closed = dict(ev)
                closed["close_reason"] = str(reason or "").strip()[:100]
                closed["closed_at"] = _bj_now().strftime("%Y-%m-%d")
            else:
                remaining.append(ev)
        if closed is None:
            return None
        data["events"] = remaining
        self._save(data)
        self.sync()
        logger.info("EventLine closed | id=%s reason=%s", event_id, reason)
        return closed

    def get_event(self, event_id: str) -> dict | None:
        for ev in self._load().get("events", []):
            if ev.get("id") == event_id:
                return ev
        return None

    def list_events(self) -> list[dict]:
        return self._load().get("events", [])

    def update_title(self, event_id: str, title: str) -> tuple[bool, str]:
        data = self._load()
        for ev in data.get("events", []):
            if ev.get("id") == event_id:
                ev["title"] = str(title or "").strip()[:60] or ev.get("title") or "未命名事件"
                self._save(data)
                self.sync()
                return True, ""
        return False, "event_id 不存在"

    def edit_entry(
        self,
        event_id: str,
        entry_index: int,
        *,
        text: str | None = None,
        progress: int | None | object = ...,
        date: str | None = None,
    ) -> tuple[bool, str]:
        data = self._load()
        for ev in data.get("events", []):
            if ev.get("id") != event_id:
                continue
            entries = list(ev.get("entries") or [])
            if entry_index < 0 or entry_index >= len(entries):
                return False, "entry_index 超出范围"
            entry = dict(entries[entry_index])
            if text is not None:
                entry["text"] = str(text or "").strip()[:200]
            if progress is not ...:
                entry["progress"] = progress
            if date is not None:
                entry["date"] = str(date or "").strip()[:10]
            entries[entry_index] = entry
            ev["entries"] = entries
            ev["last_updated"] = max(
                (str(e.get("date") or "") for e in entries),
                default=ev.get("last_updated") or _bj_now().strftime("%Y-%m-%d"),
            )
            self._save(data)
            self.sync()
            return True, ""
        return False, "event_id 不存在"

    def delete_entry(self, event_id: str, entry_index: int) -> tuple[bool, str]:
        data = self._load()
        for ev in data.get("events", []):
            if ev.get("id") != event_id:
                continue
            entries = list(ev.get("entries") or [])
            if entry_index < 0 or entry_index >= len(entries):
                return False, "entry_index 超出范围"
            entries.pop(entry_index)
            ev["entries"] = entries
            if entries:
                ev["last_updated"] = max(str(e.get("date") or "") for e in entries)
            self._save(data)
            self.sync()
            return True, ""
        return False, "event_id 不存在"

    def delete_event(self, event_id: str) -> bool:
        data = self._load()
        before = len(data.get("events", []))
        data["events"] = [ev for ev in data.get("events", []) if ev.get("id") != event_id]
        if len(data["events"]) == before:
            return False
        self._save(data)
        self.sync()
        logger.info("EventLine deleted | id=%s", event_id)
        return True

    def check_stale(self) -> list[dict]:
        """返回超过 STALE_DAYS 天未更新的进行中事件。"""
        today = _bj_now().date()
        stale = []
        for ev in self._load().get("events", []):
            try:
                last = datetime.strptime(ev.get("last_updated", ""), "%Y-%m-%d").date()
                if (today - last).days >= STALE_DAYS:
                    stale.append(ev)
            except Exception:
                continue
        return stale

    # ── 格式化 & 同步到缓存区 ───────────────────────────────────────────────────

    def format_section(self) -> str:
        events = self._load().get("events", [])
        if not events:
            return cache_activity_zone.default_block_text("EVENT_LINE")
        stale_ids = {e.get("id") for e in self.check_stale()}
        lines = ["【事件线】（跨天持续事件）"]
        for ev in events:
            entries_text = "；".join(
                f"{e.get('date', '')} {e.get('text', '')}"
                + (f"，进度{e['progress']}%" if e.get("progress") is not None else "")
                for e in ev.get("entries", [])
            )
            flag = ""
            if ev.get("id") in stale_ids:
                flag = " ⚠已超1天未更新，请判断是否继续记录或结束归档"
            lines.append(
                f"- [进行中][id:{ev.get('id', '')}] {ev.get('title', '')}：{entries_text}{flag}"
            )
        return "\n".join(lines)

    def sync(self) -> None:
        cache_activity_zone.update_block(
            self._persona_file, "EVENT_LINE", self.format_section()
        )

    # ── 归档格式化（供 gateway 写入记忆桶）──────────────────────────────────────

    @staticmethod
    def format_for_bucket(event: dict[str, Any]) -> str:
        """把已关闭的事件线打包成记忆桶正文（moment 叙事）。"""
        title = event.get("title", "")
        entries = event.get("entries", [])
        close_reason = event.get("close_reason", "")
        lines = [f"### moment", f"{title}的完整经过："]
        for e in entries:
            progress_part = f"（进度{e['progress']}%）" if e.get("progress") is not None else ""
            lines.append(f"- {e.get('date', '')}：{e.get('text', '')}{progress_part}")
        if close_reason:
            lines.append(f"最终：{close_reason}")
        return "\n".join(lines)
