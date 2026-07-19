from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class MetadataProposalStore:
    """Pending model suggestions that never change canonical memory by themselves."""

    def __init__(self, config: dict):
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        os.makedirs(state_dir, exist_ok=True)
        self.db_path = os.path.join(state_dir, "metadata_proposals.sqlite")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_proposals (
                proposal_id TEXT PRIMARY KEY,
                bucket_id TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(bucket_id, source_hash, model)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_proposals_bucket "
            "ON metadata_proposals(bucket_id, updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_proposals_status "
            "ON metadata_proposals(status, updated_at DESC)"
        )
        conn.commit()
        conn.close()

    @staticmethod
    def source_hash(bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        source = {
            "id": str(bucket.get("id") or ""),
            "content": str(bucket.get("content") or ""),
            "name": str(meta.get("name") or ""),
            "tags": list(meta.get("tags") or []),
            "scene_cues": list(meta.get("scene_cues") or []),
            "importance": meta.get("importance"),
            "confidence": meta.get("confidence"),
        }
        raw = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            item["payload"] = {}
        return item

    def put(self, bucket: dict, payload: dict[str, Any], *, model: str = "") -> dict:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            raise ValueError("metadata proposal requires bucket_id")
        source_hash = self.source_hash(bucket)
        model_name = str(model or "").strip()
        proposal_id = "meta_" + hashlib.sha256(
            f"{bucket_id}\n{source_hash}\n{model_name}".encode("utf-8")
        ).hexdigest()[:24]
        now = _now_utc()
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO metadata_proposals (
                proposal_id, bucket_id, source_hash, model,
                payload_json, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            ON CONFLICT(bucket_id, source_hash, model) DO UPDATE SET
                payload_json = excluded.payload_json,
                status = 'pending',
                updated_at = excluded.updated_at
            """,
            (
                proposal_id,
                bucket_id,
                source_hash,
                model_name,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM metadata_proposals WHERE bucket_id = ? AND source_hash = ? AND model = ?",
            (bucket_id, source_hash, model_name),
        ).fetchone()
        conn.close()
        return self._row(row) or {}

    def latest_for_bucket(self, bucket_id: str) -> dict | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM metadata_proposals WHERE bucket_id = ? ORDER BY updated_at DESC LIMIT 1",
            (str(bucket_id or "").strip(),),
        ).fetchone()
        conn.close()
        return self._row(row)

    def list_pending(self, limit: int = 50) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM metadata_proposals WHERE status = 'pending' ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(int(limit or 50), 500)),),
        ).fetchall()
        conn.close()
        return [self._row(row) or {} for row in rows]
