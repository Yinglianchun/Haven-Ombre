"""Derived shadow memory cards.

This index is deliberately separate from bucket storage and recall indexes.
Cards can be inspected and rebuilt, but they do not participate in candidate
generation, gating, diffusion, or prompt injection.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone


class ShadowMemoryCardStore:
    """SQLite store for non-authoritative memory-card previews."""

    def __init__(self, config: dict):
        config = config or {}
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = os.path.join(state_dir, "memory_cards_shadow.sqlite")
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
            CREATE TABLE IF NOT EXISTS memory_cards_shadow (
                bucket_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                generation_version TEXT NOT NULL,
                primary_abstraction TEXT NOT NULL,
                existing_moment TEXT NOT NULL DEFAULT '',
                primary_source TEXT NOT NULL,
                candidate_only INTEGER NOT NULL DEFAULT 0,
                cue_anchors_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                status TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_cards_shadow_status
            ON memory_cards_shadow(status, updated_at DESC)
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for field in ("cue_anchors_json", "evidence_refs_json"):
            target = field.removesuffix("_json")
            try:
                value = json.loads(item.pop(field) or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                value = []
            item[target] = value if isinstance(value, list) else []
        item["candidate_only"] = bool(item.get("candidate_only"))
        return item

    def upsert(self, card: dict) -> dict:
        bucket_id = str(card.get("bucket_id") or "").strip()
        source_hash = str(card.get("source_hash") or "").strip()
        generation_version = str(card.get("generation_version") or "").strip()
        if not bucket_id or not source_hash or not generation_version:
            raise ValueError("bucket_id, source_hash, and generation_version are required")

        existing = self.get(bucket_id)
        created_at = str((existing or {}).get("created_at") or card.get("created_at") or self._now())
        updated_at = self._now()
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO memory_cards_shadow (
                bucket_id, source_hash, generation_version, primary_abstraction,
                existing_moment, primary_source, candidate_only, cue_anchors_json,
                evidence_refs_json, status, model, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bucket_id) DO UPDATE SET
                source_hash=excluded.source_hash,
                generation_version=excluded.generation_version,
                primary_abstraction=excluded.primary_abstraction,
                existing_moment=excluded.existing_moment,
                primary_source=excluded.primary_source,
                candidate_only=excluded.candidate_only,
                cue_anchors_json=excluded.cue_anchors_json,
                evidence_refs_json=excluded.evidence_refs_json,
                status=excluded.status,
                model=excluded.model,
                updated_at=excluded.updated_at
            """,
            (
                bucket_id,
                source_hash,
                generation_version,
                str(card.get("primary_abstraction") or "").strip(),
                str(card.get("existing_moment") or "").strip(),
                str(card.get("primary_source") or "").strip(),
                1 if card.get("candidate_only") else 0,
                json.dumps(card.get("cue_anchors") or [], ensure_ascii=False, separators=(",", ":")),
                json.dumps(card.get("evidence_refs") or [], ensure_ascii=False, separators=(",", ":")),
                str(card.get("status") or "partial").strip(),
                str(card.get("model") or "").strip(),
                created_at,
                updated_at,
            ),
        )
        conn.commit()
        conn.close()
        return self.get(bucket_id) or {}

    def get(self, bucket_id: str) -> dict | None:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memory_cards_shadow WHERE bucket_id = ?",
            (bucket_id,),
        ).fetchone()
        conn.close()
        return self._decode(row)

    def list(self, limit: int = 100) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_cards_shadow
            ORDER BY updated_at DESC, bucket_id ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        conn.close()
        return [self._decode(row) or {} for row in rows]

    def delete(self, bucket_id: str) -> bool:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return False
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM memory_cards_shadow WHERE bucket_id = ?",
            (bucket_id,),
        )
        conn.commit()
        conn.close()
        return bool(cursor.rowcount)

    def is_current(self, bucket_id: str, source_hash: str, generation_version: str) -> bool:
        card = self.get(bucket_id)
        return bool(
            card
            and card.get("source_hash") == str(source_hash or "")
            and card.get("generation_version") == str(generation_version or "")
        )

    def stats(self) -> dict:
        conn = self._connect()
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM memory_cards_shadow GROUP BY status"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM memory_cards_shadow").fetchone()[0]
        conn.close()
        return {
            "total": int(total or 0),
            "by_status": {str(row["status"]): int(row["count"] or 0) for row in rows},
        }
