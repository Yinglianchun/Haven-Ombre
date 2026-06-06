import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

import yaml

from utils import now_iso, strip_affect_anchor, strip_wikilinks


PRIVATE_SCOPE = "private_relationship"
DEFAULT_EVIDENCE_TAGS = {"profile_fact", "haven_favorite", "favorite_memory"}


@dataclass(frozen=True)
class CanonicalNode:
    canonical: str
    scope: str
    group: str
    seed_aliases: tuple[str, ...]
    sensitivity: str = "private"


class IdentitySemanticStore:
    """Private canonical/alias overlay; aliases are stored only with evidence buckets."""

    def __init__(self, config: dict[str, Any]):
        cfg = config.get("identity_semantics", {}) if isinstance(config.get("identity_semantics", {}), dict) else {}
        state_dir = str(config.get("state_dir") or os.path.join(config.get("buckets_dir", "."), "state"))
        env_path = os.environ.get("OMBRE_IDENTITY_SEMANTICS_PATH", "").strip()
        self.private_config_path = str(cfg.get("private_config_path") or env_path or "").strip()
        self.enabled = bool(cfg.get("enabled", bool(self.private_config_path))) and bool(self.private_config_path)
        self.min_confidence = _float_between(cfg.get("min_confidence"), 0.78, 0.0, 1.0)
        self.evidence_tags = {
            str(item).strip()
            for item in (cfg.get("evidence_tags") or DEFAULT_EVIDENCE_TAGS)
            if str(item).strip()
        }
        self.db_path = str(cfg.get("db_path") or os.path.join(state_dir, "identity_semantics.sqlite"))
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_canonical (
                canonical TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                group_name TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_aliases (
                canonical TEXT NOT NULL,
                alias TEXT NOT NULL,
                scope TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(canonical, alias)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_alias_evidence (
                canonical TEXT NOT NULL,
                alias TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(canonical, alias, bucket_id)
            )
            """
        )
        conn.commit()
        conn.close()

    def load_private_nodes(self) -> list[CanonicalNode]:
        if not self.enabled:
            return []
        path = os.path.abspath(self.private_config_path)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                payload = json.load(f)
            else:
                payload = yaml.safe_load(f) or {}
        raw_nodes = payload.get("canonical") or payload.get("nodes") or {}
        if isinstance(raw_nodes, list):
            items = ((str(item), {}) for item in raw_nodes)
        elif isinstance(raw_nodes, dict):
            items = raw_nodes.items()
        else:
            items = []
        nodes: list[CanonicalNode] = []
        for canonical, raw in items:
            canonical_key = str(canonical or "").strip()
            if not canonical_key:
                continue
            cfg = raw if isinstance(raw, dict) else {}
            seed_aliases = tuple(
                dict.fromkeys(
                    alias
                    for alias in _list_text(cfg.get("seed_aliases") or cfg.get("aliases"))
                    if alias
                )
            )
            nodes.append(
                CanonicalNode(
                    canonical=canonical_key,
                    scope=str(cfg.get("scope") or PRIVATE_SCOPE).strip() or PRIVATE_SCOPE,
                    group=str(cfg.get("group") or cfg.get("group_name") or "shared").strip() or "shared",
                    seed_aliases=seed_aliases,
                    sensitivity=str(cfg.get("sensitivity") or "private").strip() or "private",
                )
            )
        return nodes

    def rebuild_alias_index(self, buckets: list[dict[str, Any]]) -> dict[str, int]:
        nodes = self.load_private_nodes()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM identity_alias_evidence")
            conn.execute("DELETE FROM identity_aliases")
            conn.execute("DELETE FROM identity_canonical")
            now = now_iso()
            for node in nodes:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity_canonical
                    (canonical, scope, group_name, sensitivity, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (node.canonical, node.scope, node.group, node.sensitivity, now),
                )
            for bucket in buckets:
                if not self._bucket_is_evidence(bucket):
                    continue
                self._index_evidence_bucket(conn, bucket, nodes)
            conn.commit()
        finally:
            conn.close()
        return self.stats()

    def aliases_for_canonical(self, canonical: str) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT a.canonical, a.alias, a.scope, a.confidence, a.source,
                       GROUP_CONCAT(e.bucket_id) AS evidence_bucket_ids
                FROM identity_aliases a
                LEFT JOIN identity_alias_evidence e
                  ON e.canonical = a.canonical AND e.alias = a.alias
                WHERE a.canonical = ?
                GROUP BY a.canonical, a.alias, a.scope, a.confidence, a.source
                ORDER BY a.confidence DESC, a.alias ASC
                """,
                (str(canonical or "").strip(),),
            ).fetchall()
            return [_alias_row(row) for row in rows]
        finally:
            conn.close()

    def list_aliases(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT a.canonical, a.alias, a.scope, a.confidence, a.source,
                       GROUP_CONCAT(e.bucket_id) AS evidence_bucket_ids
                FROM identity_aliases a
                LEFT JOIN identity_alias_evidence e
                  ON e.canonical = a.canonical AND e.alias = a.alias
                GROUP BY a.canonical, a.alias, a.scope, a.confidence, a.source
                ORDER BY a.canonical ASC, a.confidence DESC, a.alias ASC
                """
            ).fetchall()
            return [_alias_row(row) for row in rows]
        finally:
            conn.close()

    def role_edge_alias_config(self) -> dict[str, Any]:
        nodes = self._canonical_rows()
        aliases = self.list_aliases()
        groups = {"detail": set(), "context": set(), "relationship": set(), "shared": set()}
        for node in nodes:
            group = str(node.get("group_name") or "shared")
            groups.setdefault(group, set()).add(str(node.get("canonical") or ""))
        alias_map: dict[str, list[str]] = {}
        for row in aliases:
            alias_map.setdefault(row["canonical"], []).append(row["alias"])
        return {
            "enabled": bool(alias_map),
            "aliases": {key: tuple(values) for key, values in alias_map.items()},
            "detail_terms": frozenset(groups.get("detail", set())),
            "context_terms": frozenset(groups.get("context", set())),
            "relationship_terms": frozenset(groups.get("relationship", set())),
            "shared_terms": frozenset(groups.get("shared", set())),
        }

    def stats(self) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            canonical = conn.execute("SELECT COUNT(*) FROM identity_canonical").fetchone()[0]
            aliases = conn.execute("SELECT COUNT(*) FROM identity_aliases").fetchone()[0]
            evidence = conn.execute("SELECT COUNT(*) FROM identity_alias_evidence").fetchone()[0]
            return {"canonical": int(canonical), "aliases": int(aliases), "evidence": int(evidence)}
        finally:
            conn.close()

    def _canonical_rows(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM identity_canonical ORDER BY canonical ASC").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _index_evidence_bucket(
        self,
        conn: sqlite3.Connection,
        bucket: dict[str, Any],
        nodes: list[CanonicalNode],
    ) -> None:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return
        haystack = _bucket_text(bucket)
        now = now_iso()
        for node in nodes:
            for alias in node.seed_aliases:
                alias_text = str(alias or "").strip()
                if not alias_text or alias_text.lower() not in haystack:
                    continue
                confidence = max(self.min_confidence, 0.9 if node.sensitivity == "private" else 0.82)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity_aliases
                    (canonical, alias, scope, confidence, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (node.canonical, alias_text, node.scope, confidence, "evidence_bucket", now),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity_alias_evidence
                    (canonical, alias, bucket_id, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (node.canonical, alias_text, bucket_id, now),
                )

    def _bucket_is_evidence(self, bucket: dict[str, Any]) -> bool:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("resolved") or meta.get("digested") or meta.get("deprecated"):
            return False
        tags = {str(tag).strip() for tag in meta.get("tags", []) or [] if str(tag).strip()}
        return bool(
            meta.get("anchor")
            or meta.get("profile_kind")
            or "profile_fact" in tags
            or tags & self.evidence_tags
        )


def _alias_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    evidence = str(payload.pop("evidence_bucket_ids") or "")
    payload["evidence_bucket_ids"] = [item for item in evidence.split(",") if item]
    return payload


def _bucket_text(bucket: dict[str, Any]) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    parts = [
        str(meta.get("name") or ""),
        " ".join(_list_text(meta.get("tags"))),
        " ".join(_list_text(meta.get("domain"))),
        strip_wikilinks(strip_affect_anchor(str(bucket.get("content") or ""))),
    ]
    return " ".join(part for part in parts if part.strip()).lower()


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _float_between(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))
