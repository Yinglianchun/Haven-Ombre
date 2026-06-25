from __future__ import annotations

import os
import sqlite3
from typing import Any

from utils import now_iso


class TodoStore:
    """Derived todo state for followup/todo sections."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = str(config.get("todo_db_path") or os.path.join(state_dir, "todos.sqlite"))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id TEXT PRIMARY KEY,
                source_bucket_id TEXT NOT NULL,
                source_moment_id TEXT NOT NULL,
                source_section TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                updated TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                writeback_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_todos_status_active ON todos(status, active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_todos_bucket ON todos(source_bucket_id)")
        conn.commit()
        conn.close()

    def sync_from_entries(self, entries: list[dict[str, Any]]) -> None:
        now = now_iso()
        conn = self._connect()
        try:
            with conn:
                conn.execute("UPDATE todos SET active = 0 WHERE status = 'open'")
                for entry in entries or []:
                    todo_id = str(entry.get("id") or "").strip()
                    text = str(entry.get("text") or "").strip()
                    bucket_id = str(entry.get("bucket_id") or "").strip()
                    if not todo_id or not text or not bucket_id:
                        continue
                    values = {
                        "id": todo_id,
                        "source_bucket_id": bucket_id,
                        "source_moment_id": str(entry.get("moment_id") or ""),
                        "source_section": str(entry.get("section") or "followup"),
                        "source_hash": str(entry.get("source_hash") or ""),
                        "title": str(entry.get("title") or bucket_id),
                        "date": str(entry.get("date") or ""),
                        "updated": str(entry.get("updated") or ""),
                        "text": text,
                    }
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO todos (
                            id, source_bucket_id, source_moment_id, source_section, source_hash,
                            title, date, updated, text, status, active, created_at, updated_at
                        )
                        VALUES (
                            :id, :source_bucket_id, :source_moment_id, :source_section, :source_hash,
                            :title, :date, :updated, :text, 'open', 1, :now, :now
                        )
                        """,
                        {**values, "now": now},
                    )
                    conn.execute(
                        """
                        UPDATE todos
                        SET source_bucket_id = :source_bucket_id,
                            source_moment_id = :source_moment_id,
                            source_section = :source_section,
                            source_hash = :source_hash,
                            title = :title,
                            date = :date,
                            updated = :updated,
                            text = :text,
                            active = 1,
                            updated_at = :now
                        WHERE id = :id
                        """,
                        {**values, "now": now},
                    )
        finally:
            conn.close()

    def list(self, *, status: str = "open", limit: int = 50, include_inactive: bool = False) -> list[dict]:
        status = str(status or "open").strip().lower()
        params: list[Any] = []
        where = []
        if status and status != "all":
            where.append("status = ?")
            params.append(status)
        if not include_inactive:
            where.append("(active = 1 OR status != 'open')")
        sql = "SELECT * FROM todos"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY COALESCE(resolved_at, updated, updated_at, created_at) DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit or 50)))
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def get(self, todo_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM todos WHERE id = ?", (str(todo_id or ""),)).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def set_status(self, todo_id: str, status: str, *, resolved_at: str | None = None) -> dict | None:
        status = str(status or "").strip().lower()
        if status not in {"open", "done", "ignored"}:
            raise ValueError("invalid todo status")
        now = now_iso()
        resolved = resolved_at if status == "done" else None
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE todos
                    SET status = ?,
                        updated_at = ?,
                        resolved_at = ?
                    WHERE id = ?
                    """,
                    (status, now, resolved, str(todo_id or "")),
                )
            return self.get(todo_id)
        finally:
            conn.close()

    def mark_writeback(self, todo_id: str, *, writeback_at: str | None = None) -> dict | None:
        now = writeback_at or now_iso()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE todos SET writeback_at = ?, updated_at = ? WHERE id = ?",
                    (now, now, str(todo_id or "")),
                )
            return self.get(todo_id)
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "source_bucket_id": row["source_bucket_id"],
            "source_moment_id": row["source_moment_id"],
            "source_section": row["source_section"],
            "source_hash": row["source_hash"],
            "title": row["title"],
            "date": row["date"],
            "updated": row["updated"],
            "text": row["text"],
            "status": row["status"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "resolved_at": row["resolved_at"],
            "writeback_at": row["writeback_at"],
        }
