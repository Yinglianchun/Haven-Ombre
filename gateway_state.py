import os
import json
import sqlite3
from datetime import datetime
from typing import Any


class GatewayStateStore:
    """
    Tracks successful gateway rounds and which dynamic buckets were injected
    per session, so cooldown and recent-round skipping can work.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
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
            CREATE TABLE IF NOT EXISTS request_rounds (
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS injected_buckets (
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                bucket_id TEXT NOT NULL,
                injected_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_id, bucket_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_injected_lookup
            ON injected_buckets (session_id, bucket_id, injected_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS injection_debug (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_injection_debug_lookup
            ON injection_debug (session_id, id DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_context_injections (
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                injected_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recent_context_lookup
            ON recent_context_injections (session_id, injected_at DESC)
            """
        )
        conn.commit()
        conn.close()

    def record_success(
        self,
        session_id: str,
        bucket_ids: list[str],
        completed_at: datetime | None = None,
    ) -> int:
        completed_at = completed_at or datetime.now()
        completed_iso = completed_at.isoformat(timespec="seconds")
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(MAX(round_id), 0) AS current_round FROM request_rounds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_round = int(row["current_round"]) + 1
        conn.execute(
            "INSERT INTO request_rounds (session_id, round_id, completed_at) VALUES (?, ?, ?)",
            (session_id, next_round, completed_iso),
        )
        for bucket_id in bucket_ids:
            conn.execute(
                """
                INSERT OR REPLACE INTO injected_buckets
                (session_id, round_id, bucket_id, injected_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, next_round, bucket_id, completed_iso),
            )
        conn.commit()
        conn.close()
        return next_round

    def get_current_round(self, session_id: str) -> int:
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(MAX(round_id), 0) AS current_round FROM request_rounds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()
        return int(row["current_round"]) if row else 0

    def get_last_success_at(self, session_id: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT completed_at
            FROM request_rounds
            WHERE session_id = ?
            ORDER BY round_id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return datetime.fromisoformat(str(row["completed_at"]))
        except ValueError:
            return None

    def get_recent_bucket_ids(self, session_id: str, recent_rounds: int) -> set[str]:
        if recent_rounds <= 0:
            return set()
        conn = self._connect()
        current_round = self.get_current_round(session_id)
        if current_round <= 0:
            conn.close()
            return set()
        min_round = max(1, current_round - recent_rounds + 1)
        rows = conn.execute(
            """
            SELECT DISTINCT bucket_id
            FROM injected_buckets
            WHERE session_id = ? AND round_id >= ?
            """,
            (session_id, min_round),
        ).fetchall()
        conn.close()
        return {row["bucket_id"] for row in rows}

    def get_last_injected_at(self, session_id: str, bucket_id: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT injected_at
            FROM injected_buckets
            WHERE session_id = ? AND bucket_id = ?
            ORDER BY injected_at DESC
            LIMIT 1
            """,
            (session_id, bucket_id),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return datetime.fromisoformat(str(row["injected_at"]))
        except ValueError:
            return None

    def record_recent_context_injection(
        self,
        session_id: str,
        round_id: int,
        injected_at: datetime | None = None,
    ) -> None:
        injected_at = injected_at or datetime.now()
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO recent_context_injections
            (session_id, round_id, injected_at)
            VALUES (?, ?, ?)
            """,
            (session_id, int(round_id), injected_at.isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()

    def get_last_recent_context_at(self, session_id: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT injected_at
            FROM recent_context_injections
            WHERE session_id = ?
            ORDER BY injected_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return datetime.fromisoformat(str(row["injected_at"]))
        except ValueError:
            return None

    def record_injection_debug(
        self,
        session_id: str,
        round_id: int,
        payload: dict[str, Any],
        *,
        max_entries: int = 80,
    ) -> int:
        created_at = datetime.now().isoformat(timespec="seconds")
        body = json.dumps(payload, ensure_ascii=False)
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO injection_debug (session_id, round_id, created_at, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, int(round_id), created_at, body),
        )
        debug_id = int(cursor.lastrowid or 0)
        conn.execute(
            """
            DELETE FROM injection_debug
            WHERE id NOT IN (
                SELECT id FROM injection_debug ORDER BY id DESC LIMIT ?
            )
            """,
            (max(1, int(max_entries)),),
        )
        conn.commit()
        conn.close()
        return debug_id

    def list_injection_debug(
        self,
        *,
        session_id: str = "",
        limit: int = 20,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(100, int(limit)))
        conn = self._connect()
        if session_id:
            rows = conn.execute(
                """
                SELECT id, session_id, round_id, created_at, payload_json
                FROM injection_debug
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, session_id, round_id, created_at, payload_json
                FROM injection_debug
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        conn.close()

        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {"raw": row["payload_json"]}
            if isinstance(payload, dict) and not include_context:
                payload = dict(payload)
                payload.pop("stable_context", None)
                payload.pop("dynamic_context", None)
            items.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "round_id": row["round_id"],
                    "created_at": row["created_at"],
                    "payload": payload,
                }
            )
        return items

    def get_cooldown_multiplier(
        self,
        session_id: str,
        bucket_id: str,
        cooldown_hours: float,
        cooldown_floor: float,
        now: datetime | None = None,
    ) -> float:
        if cooldown_hours <= 0:
            return 1.0
        now = now or datetime.now()
        last_injected = self.get_last_injected_at(session_id, bucket_id)
        if not last_injected:
            return 1.0
        elapsed_hours = max(0.0, (now - last_injected).total_seconds() / 3600)
        if elapsed_hours >= cooldown_hours:
            return 1.0
        progress = elapsed_hours / cooldown_hours
        return round(cooldown_floor + (1.0 - cooldown_floor) * progress, 4)
