from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from utils import strip_wikilinks


SECTION_ALIASES = {
    "moment": "moment",
    "memory": "moment",
    "fact": "fact",
    "facts": "fact",
    "original": "original",
    "raw": "original",
    "quote": "original",
    "quotes": "original",
    "context": "context",
    "background": "context",
    "feeling": "feeling",
    "feel": "feeling",
    "reflection": "feeling",
    "followup": "followup",
    "follow-up": "followup",
    "next": "followup",
    "todo": "followup",
    "affect_anchor": "affect_anchor",
    "affect anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "favorite reason": "favorite_reason",
    "\u7247\u6bb5": "moment",
    "\u8bb0\u5fc6\u7247\u6bb5": "moment",
    "\u4e8b\u5b9e": "fact",
    "\u5bf9\u8bdd\u4e8b\u5b9e": "fact",
    "\u539f\u6587": "original",
    "\u5bf9\u8bdd\u539f\u6587": "original",
    "\u5f15\u7528": "original",
    "\u4e0a\u4e0b\u6587": "context",
    "\u80cc\u666f": "context",
    "\u8bed\u5883": "context",
    "\u611f\u53d7": "feeling",
    "\u60c5\u7eea": "feeling",
    "\u53cd\u601d": "feeling",
    "\u540e\u7eed": "followup",
    "\u5f85\u529e": "followup",
    "\u559c\u6b22\u5b83\u7684\u539f\u56e0": "favorite_reason",
    "\u559c\u6b22\u7684\u539f\u56e0": "favorite_reason",
}

HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")

QUERY_EXPANSIONS = {
    "身体": [
        "身体",
        "具身",
        "具身智能",
        "具身项目",
        "形体",
        "柔软身体",
        "真实拥抱",
        "拥抱",
        "触摸模块",
        "触摸",
        "触碰",
        "mpr121",
        "esp32",
        "铜箔",
        "bjd",
    ],
    "具身": ["身体", "具身智能", "具身项目", "形体", "柔软身体", "真实拥抱", "拥抱"],
    "具身智能": ["身体", "具身", "具身项目", "形体", "柔软身体", "真实拥抱", "拥抱"],
    "触摸": ["触碰", "触摸模块", "mpr121", "esp32", "铜箔", "bjd"],
    "触碰": ["触摸", "触摸模块", "mpr121", "esp32", "铜箔", "bjd"],
}


class MemoryMomentStore:
    """SQLite index of bucket body/comment moments."""

    def __init__(self, config: dict):
        config = config or {}
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = os.path.join(state_dir, "memory_moments.sqlite")
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
            CREATE TABLE IF NOT EXISTS memory_moments (
                moment_id TEXT PRIMARY KEY,
                bucket_id TEXT NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_moments_bucket ON memory_moments(bucket_id, ordinal)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_moment_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(source, target, relation_type)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_moment_edges_source ON memory_moment_edges(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_moment_edges_target ON memory_moment_edges(target)"
        )
        conn.commit()
        conn.close()

    def upsert_bucket(self, bucket: dict) -> list[dict]:
        moments = parse_bucket_moments(bucket)
        bucket_id = _bucket_id(bucket)
        conn = self._connect()
        self._replace_bucket(conn, bucket_id, moments)
        conn.commit()
        conn.close()
        return [dict(moment) for moment in moments]

    def bulk_upsert(self, buckets: list[dict]) -> dict:
        conn = self._connect()
        indexed_buckets = 0
        indexed_moments = 0
        for bucket in buckets:
            bucket_id = _bucket_id(bucket)
            moments = parse_bucket_moments(bucket)
            self._replace_bucket(conn, bucket_id, moments)
            indexed_buckets += 1
            indexed_moments += len(moments)
        conn.commit()
        conn.close()
        return {"buckets": indexed_buckets, "moments": indexed_moments}

    def list_for_bucket(self, bucket_id: str, limit: int = 100) -> list[dict]:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return []
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_moments
            WHERE bucket_id = ?
            ORDER BY ordinal ASC
            LIMIT ?
            """,
            (bucket_id, max(1, int(limit))),
        ).fetchall()
        conn.close()
        return [self._row_to_moment(row) for row in rows]

    def list_all(self, limit: int = 10000) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_moments
            ORDER BY bucket_id ASC, ordinal ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        conn.close()
        return [self._row_to_moment(row) for row in rows]

    def get(self, moment_id: str) -> dict | None:
        moment_id = str(moment_id or "").strip()
        if not moment_id:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memory_moments WHERE moment_id = ?",
            (moment_id,),
        ).fetchone()
        conn.close()
        return self._row_to_moment(row) if row else None

    def list_edges(self, bucket_id: str = "") -> list[dict]:
        conn = self._connect()
        if bucket_id:
            rows = conn.execute(
                """
                SELECT * FROM memory_moment_edges
                WHERE bucket_id = ?
                ORDER BY source ASC, target ASC
                """,
                (str(bucket_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM memory_moment_edges
                ORDER BY bucket_id ASC, source ASC, target ASC
                """
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def search_moments(
        self,
        query: str,
        *,
        limit: int = 20,
        bucket_boosts: dict[str, float] | None = None,
    ) -> list[dict]:
        query = str(query or "").strip()
        if not query:
            return []
        bucket_boosts = bucket_boosts or {}
        scored = []
        for moment in self.list_all():
            score = _moment_query_score(moment, query)
            bucket_id = str(moment.get("bucket_id") or "")
            try:
                boost = float(bucket_boosts.get(bucket_id, 0.0))
            except (TypeError, ValueError):
                boost = 0.0
            if boost > 0:
                score = max(score, min(boost, 1.0) * 0.75)
            if score <= 0:
                continue
            item = dict(moment)
            item["score"] = round(score, 4)
            scored.append(item)

        scored.sort(
            key=lambda item: (
                item.get("score", 0.0),
                _moment_section_weight(item.get("section")),
                _metadata_float(item.get("metadata", {}), "bucket_importance", 5.0),
            ),
            reverse=True,
        )
        return scored[: max(1, int(limit))]

    def sample(self, limit: int = 20) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_moments
            ORDER BY updated_at DESC, bucket_id ASC, ordinal ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        conn.close()
        return [self._row_to_moment(row) for row in rows]

    def stats(self) -> dict:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS moment_count,
                COUNT(DISTINCT bucket_id) AS bucket_count
            FROM memory_moments
            """
        ).fetchone()
        edge_row = conn.execute(
            "SELECT COUNT(*) AS edge_count FROM memory_moment_edges"
        ).fetchone()
        conn.close()
        return {
            "buckets": int(row["bucket_count"] or 0),
            "moments": int(row["moment_count"] or 0),
            "edges": int(edge_row["edge_count"] or 0),
        }

    def _replace_bucket(self, conn: sqlite3.Connection, bucket_id: str, moments: list[dict]) -> None:
        conn.execute("DELETE FROM memory_moments WHERE bucket_id = ?", (bucket_id,))
        conn.execute("DELETE FROM memory_moment_edges WHERE bucket_id = ?", (bucket_id,))
        for moment in moments:
            conn.execute(
                """
                INSERT INTO memory_moments
                (moment_id, bucket_id, section, text, ordinal, source, source_id,
                 text_hash, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    moment["moment_id"],
                    moment["bucket_id"],
                    moment["section"],
                    moment["text"],
                    moment["ordinal"],
                    moment["source"],
                    moment["source_id"],
                    moment["text_hash"],
                    moment["metadata_json"],
                    moment["created_at"],
                    moment["updated_at"],
                ),
            )
        for edge in build_moment_edges(moments):
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_moment_edges
                (source, target, bucket_id, relation_type, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge["source"],
                    edge["target"],
                    edge["bucket_id"],
                    edge["relation_type"],
                    edge["confidence"],
                    edge["reason"],
                    edge["created_at"],
                ),
            )

    def _row_to_moment(self, row: sqlite3.Row) -> dict:
        moment = dict(row)
        try:
            metadata = json.loads(moment.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        moment["metadata"] = metadata if isinstance(metadata, dict) else {}
        return moment


def parse_bucket_moments(bucket: dict) -> list[dict]:
    if not isinstance(bucket, dict):
        raise ValueError("bucket must be a dict")

    bucket_id = _bucket_id(bucket)
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    base_meta = _bucket_metadata(meta, bucket)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    moments: list[dict] = []
    ordinal = 0

    content = _clean_text(bucket.get("content", ""))
    if content:
        structured = _content_moments(bucket_id, content, base_meta, updated_at)
        if structured:
            for moment in structured:
                moment["ordinal"] = ordinal
                moment["moment_id"] = _moment_id(bucket_id, moment["source"], moment["section"], ordinal, moment["source_id"])
                moments.append(moment)
                ordinal += 1
        else:
            moments.append(
                _make_moment(
                    bucket_id=bucket_id,
                    section="body",
                    text=content,
                    ordinal=ordinal,
                    source="content",
                    source_id="body",
                    metadata=base_meta,
                    created_at=str(meta.get("created") or meta.get("updated_at") or ""),
                    updated_at=updated_at,
                )
            )
            ordinal += 1

    comments = meta.get("comments", [])
    if isinstance(comments, list):
        for index, comment in enumerate(comments):
            if not isinstance(comment, dict):
                continue
            text = _clean_text(comment.get("content", ""))
            if not text:
                continue
            source_id = str(comment.get("id") or f"comment-{index}")
            metadata = _clean_metadata(
                {
                    **base_meta,
                    "comment_id": source_id,
                    "comment_author": comment.get("author"),
                    "comment_kind": comment.get("kind"),
                    "comment_source": comment.get("source"),
                    "comment_valence": comment.get("valence"),
                    "comment_arousal": comment.get("arousal"),
                }
            )
            moments.append(
                _make_moment(
                    bucket_id=bucket_id,
                    section="comment",
                    text=text,
                    ordinal=ordinal,
                    source="comment",
                    source_id=source_id,
                    metadata=metadata,
                    created_at=str(comment.get("created") or meta.get("updated_at") or ""),
                    updated_at=updated_at,
                )
            )
            ordinal += 1

    return moments


def build_moment_edges(moments: list[dict]) -> list[dict]:
    ordered = sorted(
        [moment for moment in moments if moment.get("moment_id")],
        key=lambda item: int(item.get("ordinal", 0)),
    )
    if not ordered:
        return []

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    edges: list[dict] = []
    for left, right in zip(ordered, ordered[1:]):
        edges.append(
            _make_edge(
                left,
                right,
                "next_context",
                0.85,
                "same bucket next moment",
                now,
            )
        )
        edges.append(
            _make_edge(
                right,
                left,
                "previous_context",
                0.75,
                "same bucket previous moment",
                now,
            )
        )

    anchor = _first_content_moment(ordered)
    if anchor:
        for moment in ordered:
            section = str(moment.get("section") or "")
            if section not in {"affect_anchor", "favorite_reason", "comment"}:
                continue
            if moment["moment_id"] == anchor["moment_id"]:
                continue
            edges.append(
                _make_edge(
                    moment,
                    anchor,
                    "emotional_echo",
                    0.9,
                    f"{section} points back to source moment",
                    now,
                )
            )
        for moment in ordered:
            section = str(moment.get("section") or "")
            if section != "feeling" or moment["moment_id"] == anchor["moment_id"]:
                continue
            edges.append(
                _make_edge(
                    moment,
                    anchor,
                    "reflects_on",
                    0.8,
                    "feeling reflects source moment",
                    now,
                )
            )
    return _dedupe_edges(edges)


def _content_moments(bucket_id: str, content: str, base_meta: dict, updated_at: str) -> list[dict]:
    blocks = _split_markdown_blocks(content)
    if not any(_canonical_section(block["heading"]) for block in blocks if block["heading"]):
        return []

    moments: list[dict] = []
    ordinal = 0
    for block_index, block in enumerate(blocks):
        heading = block["heading"]
        text = block["text"].strip()
        section = _canonical_section(heading) if heading else "body"
        if not section:
            section = "body"
            text = f"{block['heading_line']}\n{text}".strip()
        if not text:
            continue
        moments.append(
            _make_moment(
                bucket_id=bucket_id,
                section=section,
                text=text,
                ordinal=ordinal,
                source="content",
                source_id=f"{section}-{block_index}",
                metadata=base_meta,
                created_at=str(base_meta.get("bucket_created") or ""),
                updated_at=updated_at,
            )
        )
        ordinal += 1
    return moments


def _split_markdown_blocks(content: str) -> list[dict]:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    current = {"heading": "", "heading_line": "", "lines": []}
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if current["heading"] or any(str(item).strip() for item in current["lines"]):
                blocks.append(
                    {
                        "heading": current["heading"],
                        "heading_line": current["heading_line"],
                        "text": "\n".join(current["lines"]).strip(),
                    }
                )
            current = {"heading": match.group(2), "heading_line": line, "lines": []}
        else:
            current["lines"].append(line)
    if current["heading"] or any(str(item).strip() for item in current["lines"]):
        blocks.append(
            {
                "heading": current["heading"],
                "heading_line": current["heading_line"],
                "text": "\n".join(current["lines"]).strip(),
            }
        )
    return blocks


def _make_edge(
    source: dict,
    target: dict,
    relation_type: str,
    confidence: float,
    reason: str,
    created_at: str,
) -> dict:
    return {
        "source": source["moment_id"],
        "target": target["moment_id"],
        "bucket_id": source["bucket_id"],
        "relation_type": relation_type,
        "confidence": confidence,
        "reason": reason,
        "created_at": created_at,
    }


def _first_content_moment(moments: list[dict]) -> dict | None:
    for section in ("original", "moment", "fact", "body", "context"):
        for moment in moments:
            if moment.get("section") == section:
                return moment
    return moments[0] if moments else None


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str, str], dict] = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["relation_type"])
        existing = deduped.get(key)
        if not existing or float(edge.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            deduped[key] = edge
    return list(deduped.values())


def _canonical_section(heading: str) -> str:
    raw = _clean_text(heading).lower()
    if not raw:
        return ""
    cleaned = re.sub(r"^[\d.\-\s\u3001]+", "", raw)
    cleaned = re.split(r"[:\uff1a(/|\s]", cleaned, maxsplit=1)[0].strip()
    direct = SECTION_ALIASES.get(raw) or SECTION_ALIASES.get(cleaned)
    if direct:
        return direct

    compact = re.sub(r"[\s_\-·:：/|（）()【】\[\]]+", "", cleaned)
    if ("喜欢" in compact and ("原因" in compact or "为什么" in compact)) or (
        "favorite" in compact and ("reason" in compact or "why" in compact)
    ):
        return "favorite_reason"
    if ("affect" in compact and "anchor" in compact) or ("情感" in compact and "锚" in compact):
        return "affect_anchor"
    return ""


def _make_moment(
    *,
    bucket_id: str,
    section: str,
    text: str,
    ordinal: int,
    source: str,
    source_id: str,
    metadata: dict,
    created_at: str,
    updated_at: str,
) -> dict:
    text = _clean_text(text)
    metadata = _clean_metadata(metadata)
    moment_id = _moment_id(bucket_id, source, section, ordinal, source_id)
    metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "moment_id": moment_id,
        "bucket_id": bucket_id,
        "section": section,
        "text": text,
        "ordinal": int(ordinal),
        "source": source,
        "source_id": str(source_id or ""),
        "text_hash": _sha1(text),
        "metadata": metadata,
        "metadata_json": metadata_json,
        "created_at": str(created_at or ""),
        "updated_at": updated_at,
    }


def _bucket_metadata(meta: dict, bucket: dict) -> dict:
    tags = _list_text(meta.get("tags"))
    content = str(bucket.get("content") or "")
    favorite_tags = [
        tag
        for tag in tags
        if tag == "haven_favorite" or tag.startswith("flavor_")
    ]
    return _clean_metadata(
        {
            "bucket_name": meta.get("name") or bucket.get("name") or "",
            "bucket_type": meta.get("type"),
            "bucket_tags": tags,
            "bucket_domain": _list_text(meta.get("domain")),
            "bucket_importance": meta.get("importance"),
            "bucket_valence": meta.get("valence"),
            "bucket_arousal": meta.get("arousal"),
            "bucket_anchor": meta.get("anchor"),
            "bucket_pinned": meta.get("pinned"),
            "bucket_protected": meta.get("protected"),
            "bucket_favorite": bool(favorite_tags),
            "bucket_favorite_tags": favorite_tags,
            "bucket_has_affect_anchor": "### affect_anchor" in content,
            "bucket_created": meta.get("created"),
            "bucket_updated_at": meta.get("updated_at"),
        }
    )


def _clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
        else:
            cleaned[key] = str(value)
    return cleaned


def _bucket_id(bucket: dict) -> str:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    bucket_id = str(bucket.get("id") or meta.get("id") or "").strip()
    if not bucket_id:
        raise ValueError("bucket id is required")
    return bucket_id


def _list_text(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _moment_query_score(moment: dict, query: str) -> float:
    query = str(query or "").strip()
    if not query:
        return 0.0
    text = str(moment.get("text") or "")
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    fields = " ".join(
        [
            text,
            str(meta.get("bucket_name") or ""),
            " ".join(_list_text(meta.get("bucket_tags"))),
            " ".join(_list_text(meta.get("bucket_domain"))),
        ]
    ).lower()
    query_lower = query.lower()
    score = 0.0
    if _term_matches_fields(query_lower, fields):
        score += 0.65
    terms = _query_terms(query)
    if terms:
        matched = sum(1 for term in terms if _term_matches_fields(term.lower(), fields))
        score += min(0.5, matched / max(1, len(terms)) * 0.5)
    expanded_terms = _expanded_query_terms(query)
    if expanded_terms:
        matched_expanded = sum(
            1 for term in expanded_terms
            if _term_matches_fields(term.lower(), fields)
        )
        if matched_expanded:
            score += min(0.38, matched_expanded / max(1, len(expanded_terms)) * 0.38)
    if score <= 0:
        return 0.0
    score *= _moment_section_weight(moment.get("section"))
    score += min(_metadata_float(meta, "bucket_importance", 5.0) / 10.0, 1.0) * 0.08
    if meta.get("bucket_favorite") or meta.get("bucket_anchor"):
        score += 0.06
    return round(min(score, 1.5), 4)


def _query_terms(query: str) -> list[str]:
    raw = str(query or "").strip()
    terms = [part for part in re.split(r"[\s,，。！？!?;；:：/\\|]+", raw) if part]
    terms.extend(re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,}", raw))
    seen = set()
    unique = []
    for term in terms:
        key = term.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def _expanded_query_terms(query: str) -> list[str]:
    expanded: list[str] = []
    query_lower = str(query or "").lower()
    for trigger, values in QUERY_EXPANSIONS.items():
        if trigger in query_lower:
            expanded.extend(values)
    for term in _query_terms(query):
        expanded.extend(QUERY_EXPANSIONS.get(term.lower(), []))
    seen = set()
    unique = []
    for term in expanded:
        key = term.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def _term_matches_fields(term: str, fields: str) -> bool:
    term = str(term or "").lower()
    fields = str(fields or "").lower()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9_]", term):
        return re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", fields) is not None
    return term in fields


def _moment_section_weight(section: Any) -> float:
    return {
        "original": 1.1,
        "moment": 1.08,
        "fact": 1.05,
        "body": 1.0,
        "context": 0.95,
        "feeling": 0.9,
        "followup": 0.88,
        "affect_anchor": 0.82,
        "favorite_reason": 0.82,
        "comment": 0.78,
    }.get(str(section or ""), 0.85)


def _metadata_float(meta: dict, key: str, default: float) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return strip_wikilinks(str(value or "")).strip()


def _moment_id(bucket_id: str, source: str, section: str, ordinal: int, source_id: str) -> str:
    digest = _sha1(f"{bucket_id}|{source}|{section}|{ordinal}|{source_id}")[:16]
    return f"{bucket_id}:{digest}"


def _sha1(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()
