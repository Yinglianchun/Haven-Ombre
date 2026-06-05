import itertools
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import jieba
import jieba.analyse

from utils import now_iso, strip_affect_anchor, strip_wikilinks


DEFAULT_WORD_MAP_STOPWORDS = {
    "一个",
    "一些",
    "一下",
    "不是",
    "为了",
    "他们",
    "你们",
    "我们",
    "什么",
    "今天",
    "刚才",
    "刚刚",
    "这个",
    "那个",
    "这些",
    "那些",
    "然后",
    "现在",
    "自己",
    "because",
    "from",
    "have",
    "that",
    "this",
    "with",
    "you",
    "active",
    "anchor",
    "archive",
    "archived",
    "comment",
    "commitment",
    "context",
    "current",
    "digested",
    "done",
    "dynamic",
    "emotional_echo",
    "event",
    "favorite",
    "feel",
    "haven_favorite",
    "memory",
    "pending",
    "permanent",
    "profile_fact",
    "project_event",
    "recent",
    "relationship_event",
    "resolved",
    "status",
    "task_status_signal",
    "todo",
    "上下文",
    "事件",
    "内容",
    "回忆",
    "当前",
    "最近",
    "状态",
    "记忆",
}

DEFAULT_STOPWORD_PREFIXES = ("flavor_", "profile_", "predicate_", "task_")


@dataclass(frozen=True)
class WordMapTerm:
    term: str
    source: str
    kind: str
    weight: float


class WordMapStore:
    """Derived word co-occurrence index for generic, non-private memory recall hints."""

    def __init__(self, config: dict[str, Any]):
        cfg = config.get("word_map", {}) if isinstance(config.get("word_map", {}), dict) else {}
        state_dir = str(config.get("state_dir") or os.path.join(config.get("buckets_dir", "."), "state"))
        self.enabled = bool(cfg.get("enabled", False))
        self.max_terms_per_bucket = _int_between(cfg.get("max_terms_per_bucket"), 16, 4, 80)
        self.edge_top_k = _int_between(cfg.get("edge_top_k"), 10, 2, 40)
        self.min_term_len = _int_between(cfg.get("min_term_len"), 2, 1, 12)
        self.db_path = str(cfg.get("db_path") or os.path.join(state_dir, "word_map.sqlite"))
        self.stopwords = {
            _normalize_term(item)
            for item in itertools.chain(
                DEFAULT_WORD_MAP_STOPWORDS,
                _identity_stopwords(config),
                cfg.get("stopwords", []) or [],
            )
            if _normalize_term(item)
        }
        self.stopword_prefixes = tuple(
            str(item).strip().lower()
            for item in itertools.chain(DEFAULT_STOPWORD_PREFIXES, cfg.get("stopword_prefixes", []) or [])
            if str(item).strip()
        )
        self.private_terms = {
            _normalize_term(item)
            for item in itertools.chain(
                cfg.get("private_terms", []) or [],
                reflection_identity_terms(config),
            )
            if _normalize_term(item)
        }
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS word_nodes (
                term TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                bucket_count INTEGER NOT NULL DEFAULT 0,
                weight REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS word_card_nodes (
                bucket_id TEXT NOT NULL,
                term TEXT NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                weight REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(bucket_id, term)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS word_edges (
                term_a TEXT NOT NULL,
                term_b TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                weight REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(term_a, term_b, bucket_id)
            )
            """
        )
        conn.commit()
        conn.close()

    def rebuild(self, buckets: list[dict[str, Any]]) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM word_edges")
            conn.execute("DELETE FROM word_card_nodes")
            conn.execute("DELETE FROM word_nodes")
            for bucket in buckets:
                self._write_bucket(conn, bucket)
            self._refresh_node_stats(conn)
            conn.commit()
        finally:
            conn.close()
        return self.stats()

    def upsert_bucket(self, bucket: dict[str, Any]) -> dict[str, int]:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return self.stats()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM word_edges WHERE bucket_id = ?", (bucket_id,))
            conn.execute("DELETE FROM word_card_nodes WHERE bucket_id = ?", (bucket_id,))
            self._write_bucket(conn, bucket)
            self._refresh_node_stats(conn)
            conn.commit()
        finally:
            conn.close()
        return self.stats()

    def extract_bucket_terms(self, bucket: dict[str, Any]) -> list[WordMapTerm]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        text = _bucket_text(bucket)
        terms: dict[str, WordMapTerm] = {}

        def add(raw: Any, source: str, kind: str, weight: float) -> None:
            term = self._clean_term(raw)
            if not term:
                return
            current = terms.get(term)
            incoming = WordMapTerm(term=term, source=source, kind=kind, weight=weight)
            if current is None or incoming.weight > current.weight:
                terms[term] = incoming

        for raw in _list_text(meta.get("subject")):
            add(raw, "subject", "subject", 1.0)
        name = str(meta.get("name") or "").strip()
        if 2 <= len(name) <= 32:
            add(name, "name", "subject", 0.9)
            jieba.add_word(name, freq=20000)
        for raw in _list_text(meta.get("keywords")):
            add(raw, "keyword", "keyword", 0.86)
            jieba.add_word(str(raw), freq=20000)
        for raw in _list_text(meta.get("tags")):
            add(raw, "tag", "keyword", 0.78)
            jieba.add_word(str(raw), freq=16000)
        for raw in _list_text(meta.get("domain")):
            add(raw, "domain", "keyword", 0.62)
            jieba.add_word(str(raw), freq=12000)

        for word, score in jieba.analyse.extract_tags(text, topK=max(self.max_terms_per_bucket * 2, 20), withWeight=True):
            add(word, "tfidf", "keyword", min(0.74, max(0.18, float(score) / 3.0)))

        return sorted(terms.values(), key=lambda item: (-item.weight, item.term))[: self.max_terms_per_bucket]

    def list_nodes(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT * FROM word_nodes
                ORDER BY bucket_count DESC, weight DESC, term ASC
                LIMIT ?
                """,
                (_int_between(limit, 50, 1, 500),),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def list_edges(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT term_a, term_b, COUNT(*) AS bucket_count, SUM(weight) AS weight
                FROM word_edges
                GROUP BY term_a, term_b
                ORDER BY bucket_count DESC, weight DESC, term_a ASC, term_b ASC
                LIMIT ?
                """,
                (_int_between(limit, 50, 1, 500),),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def cards_for_term(self, term: str, limit: int = 20) -> list[dict[str, Any]]:
        cleaned = self._clean_term(term)
        if not cleaned:
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT bucket_id, term, source, kind, weight, updated_at
                FROM word_card_nodes
                WHERE term = ?
                ORDER BY weight DESC, bucket_id ASC
                LIMIT ?
                """,
                (cleaned, _int_between(limit, 20, 1, 200)),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def stats(self) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            nodes = conn.execute("SELECT COUNT(*) FROM word_nodes").fetchone()[0]
            cards = conn.execute("SELECT COUNT(*) FROM word_card_nodes").fetchone()[0]
            edge_rows = conn.execute("SELECT COUNT(*) FROM word_edges").fetchone()[0]
            return {"nodes": int(nodes), "card_nodes": int(cards), "edge_evidence": int(edge_rows)}
        finally:
            conn.close()

    def _write_bucket(self, conn: sqlite3.Connection, bucket: dict[str, Any]) -> None:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return
        now = now_iso()
        terms = self.extract_bucket_terms(bucket)
        for term in terms:
            conn.execute(
                """
                INSERT OR REPLACE INTO word_card_nodes
                (bucket_id, term, source, kind, weight, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (bucket_id, term.term, term.source, term.kind, float(term.weight), now),
            )
        edge_terms = sorted(terms[: self.edge_top_k], key=lambda item: item.term)
        for left, right in itertools.combinations(edge_terms, 2):
            if left.term == right.term:
                continue
            term_a, term_b = sorted((left.term, right.term))
            conn.execute(
                """
                INSERT OR REPLACE INTO word_edges
                (term_a, term_b, bucket_id, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (term_a, term_b, bucket_id, round((left.weight + right.weight) / 2.0, 4), now),
            )

    def _refresh_node_stats(self, conn: sqlite3.Connection) -> None:
        now = now_iso()
        conn.execute("DELETE FROM word_nodes")
        rows = conn.execute(
            """
            SELECT term, kind, COUNT(DISTINCT bucket_id) AS bucket_count, SUM(weight) AS weight
            FROM word_card_nodes
            GROUP BY term, kind
            """
        ).fetchall()
        for term, kind, bucket_count, weight in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO word_nodes
                (term, kind, bucket_count, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (term, kind, int(bucket_count), float(weight or 0.0), now),
            )

    def _clean_term(self, value: Any) -> str:
        term = _normalize_term(value)
        if not term:
            return ""
        if term in self.stopwords or term in self.private_terms:
            return ""
        if any(term.startswith(prefix) for prefix in self.stopword_prefixes):
            return ""
        if len(term) < self.min_term_len or len(term) > 40:
            return ""
        if re.fullmatch(r"[a-z0-9_.:-]+", term) and len(term) < 3:
            return ""
        if re.fullmatch(r"[\d.:-]+", term):
            return ""
        return term


def _bucket_text(bucket: dict[str, Any]) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    parts = [
        str(meta.get("name") or ""),
        " ".join(_list_text(meta.get("tags"))),
        " ".join(_list_text(meta.get("domain"))),
        strip_wikilinks(strip_affect_anchor(str(bucket.get("content") or ""))),
    ]
    return " ".join(part for part in parts if part.strip())


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_term(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"'`“”‘’[]【】()（）")
    return text.lower() if re.fullmatch(r"[A-Za-z0-9_.:/ -]+", text) else text


def _identity_stopwords(config: dict[str, Any]) -> list[str]:
    identity = config.get("identity", {}) if isinstance(config.get("identity", {}), dict) else {}
    values = [
        identity.get("ai_name"),
        identity.get("user_name"),
        identity.get("user_display_name"),
    ]
    values.extend(identity.get("user_aliases") or [])
    return [str(item).strip() for item in values if str(item).strip()]


def reflection_identity_terms(config: dict[str, Any]) -> list[str]:
    reflection = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    role_edges = reflection.get("identity_role_edges", {}) if isinstance(reflection.get("identity_role_edges", {}), dict) else {}
    if not role_edges.get("enabled"):
        return []

    terms: list[str] = []
    for section in ("detail", "context", "relationship", "shared"):
        terms.extend(_collect_config_terms(role_edges.get(section)))
    return terms


def _collect_config_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        terms: list[str] = []
        for key, child in value.items():
            if str(key).strip():
                terms.append(str(key).strip())
            terms.extend(_collect_config_terms(child))
        return terms
    if isinstance(value, (list, tuple, set)):
        terms: list[str] = []
        for item in value:
            terms.extend(_collect_config_terms(item))
        return terms
    text = str(value).strip()
    return [text] if text else []


def _int_between(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def dumps_debug(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
