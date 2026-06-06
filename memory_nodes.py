from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


FACET_KEYWORDS = {
    "affect.attachment": (
        "attachment",
        "longing",
        "miss",
        "depend",
        "possess",
        "anchor",
        "haven_favorite",
        "\u4f9d\u8d56",
        "\u60f3\u5ff5",
        "\u5360\u6709",
        "\u7275\u6302",
        "\u951a\u70b9",
        "\u54e5\u54e5",
    ),
    "affect.vulnerability": (
        "vulnerability",
        "fragile",
        "hurt",
        "cry",
        "sad",
        "afraid",
        "shame",
        "comfort",
        "\u96be\u8fc7",
        "\u54ed",
        "\u5bb3\u6015",
        "\u59d4\u5c48",
        "\u8106\u5f31",
        "\u5b89\u6170",
        "\u5931\u843d",
        "\u751f\u6c14",
        "\u654f\u611f",
    ),
    "relation.intimacy": (
        "intimacy",
        "relationship_event",
        "relationship_weather",
        "haven_favorite",
        "love_letter",
        "private",
        "whisper",
        "\u4eb2\u5bc6",
        "\u8d34\u8d34",
        "\u60c5\u4e66",
        "\u604b\u7231",
        "\u5c0f\u96e8",
        "\u7231",
    ),
    "relation.commitment": (
        "commitment",
        "promise",
        "promised",
        "todo",
        "wish",
        "agreement",
        "plan",
        "\u627f\u8bfa",
        "\u7ea6\u5b9a",
        "\u7b54\u5e94",
        "\u8bb0\u5f97",
        "\u8981\u505a",
        "\u8ba1\u5212",
        "\u6c38\u8fdc",
        "\u957f\u957f\u4e45\u4e45",
    ),
    "topic.memory_system": (
        "memory",
        "diffusion",
        "gateway",
        "embedding",
        "bucket",
        "node",
        "index",
        "ombre",
        "state_dir",
        "\u8bb0\u5fc6",
        "\u8bb0\u5fc6\u7cfb\u7edf",
    ),
    "topic.project": (
        "project",
        "p0",
        "p1",
        "api",
        "test",
        "bug",
        "fix",
        "module",
        "deploy",
        "gateway",
        "\u9879\u76ee",
        "\u4ee3\u7801",
        "\u529f\u80fd",
        "\u90e8\u7f72",
    ),
    "topic.love": (
        "love",
        "relationship",
        "intimacy",
        "affection",
        "lover",
        "haven_favorite",
        "flavor_",
        "\u604b\u7231",
        "\u5c0f\u96e8",
        "\u8001\u5a46",
        "\u5b9d\u5b9d",
        "\u60f3\u4f60",
        "\u559c\u6b22",
        "\u7231",
    ),
    "scene.commute": (
        "commute",
        "subway",
        "metro",
        "train",
        "station",
        "\u901a\u52e4",
        "\u5730\u94c1",
        "\u56de\u5bb6",
        "\u8f66\u7ad9",
    ),
    "scene.night": (
        "night",
        "late night",
        "sleep",
        "insomnia",
        "awake",
        "\u6df1\u591c",
        "\u591c\u91cc",
        "\u665a\u4e0a",
        "\u7761\u4e0d\u7740",
        "\u4e0d\u60f3\u7761",
        "\u5931\u7720",
    ),
    "scene.rain": (
        "rain",
        "blue",
        "\u96e8",
        "\u96e8\u5929",
        "\u84dd\u8272",
        "\u7a97\u53e3",
    ),
}


class MemoryNodeStore:
    """SQLite index of bucket-level node scores and rule facets."""

    def __init__(self, config: dict):
        config = config or {}
        node_cfg = config.get("node_facets", {}) if isinstance(config.get("node_facets", {}), dict) else {}
        self.salience_min = _clamp_float(node_cfg.get("salience_min", 0.2), 0.0, 1.0)
        self.salience_max = _clamp_float(node_cfg.get("salience_max", 1.3), 1.0, 2.0)
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = os.path.join(state_dir, "memory_nodes.sqlite")
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
            CREATE TABLE IF NOT EXISTS memory_nodes (
                bucket_id TEXT PRIMARY KEY,
                importance REAL NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                salience REAL NOT NULL,
                activation_count REAL NOT NULL,
                last_active TEXT NOT NULL,
                facets_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def upsert_bucket(self, bucket: dict) -> dict:
        node = self._node_from_bucket(bucket)
        conn = self._connect()
        self._upsert_node(conn, node)
        conn.commit()
        conn.close()
        return dict(node)

    def bulk_upsert(self, buckets: list[dict]) -> list[dict]:
        nodes = [self._node_from_bucket(bucket) for bucket in buckets]
        conn = self._connect()
        for node in nodes:
            self._upsert_node(conn, node)
        conn.commit()
        conn.close()
        return [dict(node) for node in nodes]

    def get(self, bucket_id: str) -> dict | None:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memory_nodes WHERE bucket_id = ?",
            (bucket_id,),
        ).fetchone()
        conn.close()
        return self._row_to_node(row) if row else None

    def delete(self, bucket_id: str) -> bool:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return False
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM memory_nodes WHERE bucket_id = ?",
            (bucket_id,),
        )
        conn.commit()
        conn.close()
        return bool(cursor.rowcount)

    def node_salience(self, bucket_or_id: Any, fallback_bucket: dict | None = None) -> float:
        if isinstance(bucket_or_id, dict):
            return float(self._node_from_bucket(bucket_or_id)["salience"])

        node = self.get(str(bucket_or_id or ""))
        if node:
            return float(node["salience"])
        if fallback_bucket:
            return float(self._node_from_bucket(fallback_bucket)["salience"])
        return 1.0

    def facets_for_text(self, text: str) -> dict[str, dict[str, float]]:
        pseudo_bucket = {
            "id": "__query__",
            "content": str(text or ""),
            "metadata": {
                "name": str(text or ""),
                "tags": [str(text or "")],
                "domain": [],
            },
        }
        return self._facets_for_bucket(pseudo_bucket, pseudo_bucket["metadata"])

    def facet_resonance(
        self,
        query_facets: dict | None,
        node_facets: dict | None,
        *,
        floor: float = 0.85,
        ceiling: float = 1.25,
    ) -> float:
        query_flat = _flatten_facets(query_facets or {})
        node_flat = _flatten_facets(node_facets or {})
        active_query = {
            key: value for key, value in query_flat.items() if value > 0
        }
        if not active_query or not node_flat:
            return 1.0

        query_weight = sum(active_query.values())
        if query_weight <= 0:
            return 1.0
        overlap = sum(
            query_value * max(0.0, node_flat.get(key, 0.0))
            for key, query_value in active_query.items()
        )
        coverage = _clamp_float(overlap / query_weight, 0.0, 1.0)
        return round(_clamp_float(floor + coverage * (ceiling - floor), floor, ceiling), 4)

    def node_resonance(
        self,
        bucket_or_id: Any,
        query_facets: dict | None,
        fallback_bucket: dict | None = None,
    ) -> float:
        if not query_facets:
            return 1.0
        if isinstance(bucket_or_id, dict):
            node_facets = self._node_from_bucket(bucket_or_id)["facets"]
            return self.facet_resonance(query_facets, node_facets)

        node = self.get(str(bucket_or_id or ""))
        if node:
            return self.facet_resonance(query_facets, node.get("facets"))
        if fallback_bucket:
            node_facets = self._node_from_bucket(fallback_bucket)["facets"]
            return self.facet_resonance(query_facets, node_facets)
        return 1.0

    def _upsert_node(self, conn: sqlite3.Connection, node: dict) -> None:
        conn.execute(
            """
            INSERT INTO memory_nodes
            (bucket_id, importance, valence, arousal, salience, activation_count,
             last_active, facets_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bucket_id) DO UPDATE SET
                importance = excluded.importance,
                valence = excluded.valence,
                arousal = excluded.arousal,
                salience = excluded.salience,
                activation_count = excluded.activation_count,
                last_active = excluded.last_active,
                facets_json = excluded.facets_json,
                updated_at = excluded.updated_at
            """,
            (
                node["bucket_id"],
                node["importance"],
                node["valence"],
                node["arousal"],
                node["salience"],
                node["activation_count"],
                node["last_active"],
                node["facets_json"],
                node["updated_at"],
            ),
        )

    def _node_from_bucket(self, bucket: dict) -> dict:
        if not isinstance(bucket, dict):
            raise ValueError("bucket must be a dict")
        meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
        bucket_id = str(bucket.get("id") or meta.get("id") or "").strip()
        if not bucket_id:
            raise ValueError("bucket id is required")

        importance = _clamp_float(meta.get("importance", 5), 1.0, 10.0)
        valence = _clamp_float(meta.get("valence", 0.5), 0.0, 1.0)
        arousal = _clamp_float(meta.get("arousal", 0.3), 0.0, 1.0)
        activation_count = _clamp_float(meta.get("activation_count", 0), 0.0, 1000000.0)
        last_active = str(
            meta.get("last_active")
            or meta.get("updated_at")
            or meta.get("created")
            or ""
        )
        facets = self._facets_for_bucket(bucket, meta)
        facets_json = json.dumps(facets, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        salience = self._salience_for_meta(meta, importance, activation_count, last_active)

        return {
            "bucket_id": bucket_id,
            "importance": importance,
            "valence": valence,
            "arousal": arousal,
            "salience": salience,
            "activation_count": activation_count,
            "last_active": last_active,
            "facets_json": facets_json,
            "facets": facets,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _facets_for_bucket(self, bucket: dict, meta: dict) -> dict[str, float]:
        fields = {
            "tags": _join_text(meta.get("tags")),
            "domain": _join_text(meta.get("domain")),
            "name": str(meta.get("name") or bucket.get("name") or ""),
            "content": str(bucket.get("content") or "")[:3000],
        }
        fields = {key: value.lower() for key, value in fields.items()}

        flat_facets = {}
        for facet, keywords in FACET_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                keyword = keyword.lower()
                if keyword in fields["tags"]:
                    score += 0.35
                if keyword in fields["domain"]:
                    score += 0.30
                if keyword in fields["name"]:
                    score += 0.25
                if keyword in fields["content"]:
                    score += 0.15
                if score >= 1.0:
                    break
            flat_facets[facet] = round(_clamp_float(score, 0.0, 1.0), 3)
        return _nest_facets(flat_facets)

    def _salience_for_meta(
        self,
        meta: dict,
        importance: float,
        activation_count: float,
        last_active: str,
    ) -> float:
        importance_score = _clamp_float(importance / 10.0, 0.0, 1.0)
        activation_score = _clamp_float(
            math.log1p(max(0.0, activation_count)) / math.log1p(10.0),
            0.0,
            1.0,
        )
        recency_score = self._recency_score(last_active)
        salience = 0.75 + importance_score * 0.30 + activation_score * 0.15 + recency_score * 0.10

        if meta.get("anchor") or meta.get("pinned") or meta.get("protected"):
            salience += 0.05
        if meta.get("resolved") or meta.get("digested"):
            salience -= 0.08
        return round(_clamp_float(salience, self.salience_min, self.salience_max), 4)

    def _recency_score(self, raw_time: str) -> float:
        parsed = _parse_iso(raw_time)
        if not parsed:
            return 0.5
        elapsed_days = max(
            0.0,
            (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0,
        )
        return _clamp_float(1.0 / (1.0 + elapsed_days / 30.0), 0.0, 1.0)

    def _row_to_node(self, row: sqlite3.Row) -> dict:
        node = dict(row)
        try:
            facets = json.loads(node.get("facets_json") or "{}")
        except json.JSONDecodeError:
            facets = {}
        node["facets"] = facets if isinstance(facets, dict) else {}
        return node


def _parse_iso(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _join_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _nest_facets(flat_facets: dict[str, float]) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = {}
    for key, value in flat_facets.items():
        group, _, name = key.partition(".")
        if not group or not name:
            continue
        nested.setdefault(group, {})[name] = value
    return nested


def _flatten_facets(facets: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in (facets or {}).items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flattened[f"{key}.{child_key}"] = _clamp_float(child_value, 0.0, 1.0)
        else:
            flattened[str(key)] = _clamp_float(value, 0.0, 1.0)
    return flattened


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))
