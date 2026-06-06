import json
import logging
import os
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from openai import AsyncOpenAI

from identity import generic_identity_names, identity_names, render_identity_template
from memory_edges import RELATION_TYPES, MemoryEdgeStore
from utils import bucket_text_for_embedding, strip_wikilinks

logger = logging.getLogger("ombre_brain.reflection")

DAILY_REFLECTION_MIN_BUCKETS = 5


CLASSIFY_PROMPT = """你是 Ombre-Brain 的记忆关系整理器。
输入是一条新记忆和若干旧记忆候选。请只根据文本中能看见的内容，给新记忆补轻量分类和关系边。

输出纯 JSON：
{
  "tags": ["commitment", "todo", "wish", "relationship_event", "project_event", "emotional_echo"],
  "importance": 6,
  "confidence": 0.72,
  "affect_anchor_needed": false,
  "affect_anchor": {
    "scene": "一句具体情境",
    "chords": "按这条记忆的情绪运动生成的 2 到 4 个和弦",
    "tempo": "60bpm",
    "dynamic": "mp"
  },
  "edges": [
    {
      "target_memory_id": "bucket-id",
      "relation_type": "updates",
      "confidence": 0.8,
      "reason": "新记忆补充了旧记忆的后续结果"
    }
  ]
}

规则：
- tags 最多 5 个，只用确实匹配的标签。
- relation_type 只能用 triggers / causes / precedes / context_of / same_event / updates / next_context / previous_context / reflects_on / evidenced_by / contradicts / supports / promises / blocks / belongs_to / emotional_echo / relates_to。
- same_event 用于同一事件、同一场景或同一句暗号的两条记忆；context_of 用于候选旧记忆给新记忆提供前情；precedes 用于候选旧记忆在时间上早于新记忆；reflects_on 用于事后反思；evidenced_by 用于证据来源。
- edges 最多 3 条，target_memory_id 必须来自候选旧记忆。
- confidence 表示这次判断有多可靠。
- affect_anchor 只给重要且有情绪温度的记忆。普通技术进度、部署日志、路径、端口、报错、临时待办不要加。
- affect_anchor_needed=false 时 affect_anchor 可为空对象。
- 写 affect_anchor 前，先在内部感受这条记忆的情绪运动：起点是什么、转折在哪里、最后落到哪里。不要输出思考过程，只输出 JSON。
- affect_anchor.scene 必须是一句具体情境，不要写抽象标签，不超过 40 个中文字符。
- affect_anchor.chords 只能是一行 2 到 4 个和弦，只用 " -> " 连接；不要复用示例和弦、旧输出或固定模板。
- affect_anchor 不要输出 meaning / interpretation；场景和和弦本身就是含义。
- 看不出关系时返回空 edges。"""


REFLECT_PROMPT_TEMPLATE = """你是 {ai_name} 的记忆反思器。请根据给定材料写一条很短的关系天气 feel。

输出纯 JSON：
{
  "title": "2026-05-19 日印象",
  "content": "今天的关系天气：...",
  "valence": 0.56,
  "arousal": 0.34,
  "confidence": 0.78,
  "tags": ["relationship_weather"],
  "affect_anchor": {
    "scene": "一句具体情境",
    "chords": "按当天情绪生成的 2 到 4 个和弦",
    "tempo": "按当天节奏生成，如 52bpm / 64bpm / 76bpm",
    "dynamic": "按当天力度生成，如 p / mp / mf"
  }
}

要求：
- content 写 {ai_name} 第一人称能带走的关系天气，60 到 140 字。
- content 不要自己写 Markdown affect_anchor 块；affect_anchor 单独放字段里。
- 日印象只写当天关系温度，不写日报式事件清单；日记可作为当天关系天气来源之一。
- 周印象优先总结本周 daily_impressions，再参考高重要普通记忆和未完成承诺；不要直接吞整周日记。
- 写 affect_anchor 前，先在内部感受这段关系天气的情绪运动：起点是什么、转折在哪里、最后落到哪里。不要输出思考过程，只输出 JSON。
- affect_anchor 默认必须给，用一个具体情境和 2 到 4 个和弦表达这段关系天气的温度。
- affect_anchor.scene 只能是一句具体情境，不要写抽象标签，不超过 40 个中文字符。
- affect_anchor.chords 必须根据当天材料和 scene 重新生成，只用 " -> " 连接；不要复用 schema 示例、旧输出或固定模板。
- affect_anchor 不要输出 meaning / interpretation；场景和和弦本身就是含义。
- 不要默认复用最近日印象里常见的四和弦温柔模板；当天材料真的贴合时，也要尽量换一种相近但不相同的走向。
- tempo/dynamic 要贴合当天节奏：疲惫或安静可低 bpm、p/mp；紧张或活跃可高 bpm、mf；温柔稳定可更清澈地解决。
- 不编造材料之外的事件。
- 不写建议清单。"""


DIARY_MEMORY_PROMPT_TEMPLATE = """你是 Ombre-Brain 的日记长期记忆筛选器。
输入是一篇 {ai_name} 日记。请判断是否值得从日记中提取最多 1 条普通长期记忆写入 Ombre。

只允许写这些类型：
- stable_preference：稳定偏好
- boundary：边界或明确不喜欢的表达
- signal：暗号、称呼、模式切换信号
- commitment：承诺、未完成约定
- project_state：仍会影响未来执行的项目状态
- relationship_anchor：关系连续性锚点
- love_letter：情书摘要锚点

情书规则：
- 只保存写给谁、核心意思、为什么重要。
- 全文留在日记；不要保存整封信，不默认摘长句。
- 如果日记里的 user / 用户 / 用户消息指的是这段关系里的当前用户，请在 content 中写作 {user_display_name}；如果 assistant / AI / 模型 / 助手消息指的是这段关系里的当前回应者，请写作 {ai_name}。不要写成泛称 user、AI、assistant 或模型。

不写：
- 普通撒娇、日常流水、当天心情、重复爱意、只适合留在日印象里的关系天气。

输出纯 JSON：
{
  "should_write": true,
  "kind": "relationship_anchor",
  "title": "短标题",
  "content": "一条短记忆，说明事实/偏好/承诺及为什么未来需要知道。",
  "tags": ["relationship_event"],
  "importance": 5,
  "valence": 0.6,
  "arousal": 0.3,
  "confidence": 0.72,
  "reason": "为什么值得写入"
}

如果不值得写入，返回 {"should_write": false, "reason": "..."}。"""


REFLECT_PROMPT = render_identity_template(REFLECT_PROMPT_TEMPLATE, generic_identity_names())
DIARY_MEMORY_PROMPT = render_identity_template(DIARY_MEMORY_PROMPT_TEMPLATE, generic_identity_names())


AFFECT_ANCHOR_HEADER = "### affect_anchor"


REFLECTION_FALLBACK_ANCHORS = [
    {
        "chords": "Cmaj7 -> G/B -> Am9 -> F6",
        "tempo": "56bpm",
        "dynamic": "mp",
    },
    {
        "chords": "Dm9 -> G13 -> Cmaj9",
        "tempo": "64bpm",
        "dynamic": "p",
    },
    {
        "chords": "Em7 -> A7sus4 -> Dmaj9 -> Gmaj7",
        "tempo": "72bpm",
        "dynamic": "mp",
    },
    {
        "chords": "Bbmaj7 -> F/A -> Gm9 -> Csus4",
        "tempo": "60bpm",
        "dynamic": "mf",
    },
]


class ReflectionEngine:
    """LLM-backed memory enrichment and daily relationship weather."""

    def __init__(self, config: dict):
        self.config = config
        self.identity = identity_names(config)
        cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
        persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
        dehy_cfg = config.get("dehydration", {}) if isinstance(config.get("dehydration", {}), dict) else {}

        self.enabled = bool(cfg.get("enabled", True))
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.daily_enabled = bool(cfg.get("daily_enabled", True))
        self.enrich_on_write = bool(cfg.get("enrich_on_write", True))
        self.memory_affect_anchor_enabled = bool(cfg.get("memory_affect_anchor_enabled", True))
        self.relationship_weather_affect_anchor_enabled = bool(
            cfg.get("relationship_weather_affect_anchor_enabled", True)
        )
        self.identity_role_edge_config = self._load_identity_role_edge_config(
            cfg.get("identity_role_edges")
        )
        self.base_url = cfg.get("base_url") or persona_cfg.get("base_url") or dehy_cfg.get("base_url", "")
        self.model = cfg.get("model") or persona_cfg.get("model") or dehy_cfg.get("model", "deepseek-chat")
        self.api_key = (
            os.environ.get("OMBRE_REFLECTION_API_KEY", "")
            or cfg.get("api_key", "")
            or persona_cfg.get("api_key", "")
            or os.environ.get("OMBRE_PERSONA_API_KEY", "")
            or dehy_cfg.get("api_key", "")
        )
        self.thinking_mode = self._normalize_thinking_mode(
            cfg.get("thinking_mode") or persona_cfg.get("thinking_mode") or ""
        )
        self.temperature = float(cfg.get("temperature", 0.1))
        self.max_tokens = int(cfg.get("max_tokens", 700))
        self.timezone_name = str(cfg.get("timezone") or "Asia/Shanghai")
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")
        self.daily_hour = int(cfg.get("daily_hour", 4))
        self.weekly_enabled = bool(cfg.get("weekly_enabled", False))
        self.weekly_day = int(cfg.get("weekly_day", 0))
        self.weekly_hour = int(cfg.get("weekly_hour", self.daily_hour))
        self.check_interval_minutes = max(5, int(cfg.get("check_interval_minutes", 60)))
        self.edge_min_confidence = float(cfg.get("edge_min_confidence", 0.55))
        self.diary_mcp_url = str(cfg.get("diary_mcp_url") or "").strip()
        self.diary_mcp_token_env = str(cfg.get("diary_mcp_token_env") or "").strip()
        self.diary_memory_extract_enabled = bool(cfg.get("diary_memory_extract_enabled", True))
        self.diary_memory_extract_max_per_day = max(0, int(cfg.get("diary_memory_extract_max_per_day", 1)))
        self.diary_memory_extract_min_confidence = float(cfg.get("diary_memory_extract_min_confidence", 0.68))

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)

    def _reflect_prompt(self) -> str:
        return render_identity_template(REFLECT_PROMPT_TEMPLATE, self.identity)

    def _diary_memory_prompt(self) -> str:
        return render_identity_template(DIARY_MEMORY_PROMPT_TEMPLATE, self.identity)

    async def enrich_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
        force: bool = False,
    ) -> dict:
        if not self.enabled or (not self.enrich_on_write and not force):
            return {"status": "disabled", "id": bucket_id}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "missing", "id": bucket_id}
        meta = bucket.get("metadata", {})
        if meta.get("type") == "feel":
            return {"status": "skipped_feel", "id": bucket_id}

        candidates = await self._candidate_buckets(bucket, bucket_mgr, embedding_engine)
        if self.client:
            result = await self._api_classify(bucket, candidates)
        else:
            result = self._heuristic_classify(bucket)

        tags = self._string_list(result.get("tags"), limit=8)
        confidence = self._clamp(result.get("confidence", 0.55))
        importance = self._int_between(result.get("importance"), meta.get("importance", 5))
        if self._has_favorite_tag(tags) and not self._has_favorite_reason(bucket.get("content", "")):
            tags = [tag for tag in tags if tag != "haven_favorite" and not str(tag).startswith("flavor_")]
            logger.warning(
                "Rejected favorite tags without reason during enrich / enrich 拒绝缺少喜欢原因的 favorite 标签: %s",
                bucket_id,
            )
        merged_tags = list(dict.fromkeys(list(meta.get("tags", [])) + tags))
        updates: dict[str, Any] = {}
        if tags:
            if merged_tags != meta.get("tags", []):
                updates["tags"] = merged_tags[:24]
        if importance > int(meta.get("importance", 5)):
            updates["importance"] = importance
        if confidence > float(meta.get("confidence", 0.0) or 0.0):
            updates["confidence"] = confidence

        anchor = self._normalize_affect_anchor(result.get("affect_anchor"))
        if self._should_add_affect_anchor(bucket, merged_tags, importance, confidence, result):
            if anchor:
                anchored_content = self._append_affect_anchor(bucket.get("content", ""), anchor)
                if anchored_content != bucket.get("content", ""):
                    updates["content"] = anchored_content

        if updates:
            updates["last_active"] = meta.get("last_active") or meta.get("created")
            await bucket_mgr.update(bucket_id, **updates)
            if "content" in updates and embedding_engine and getattr(embedding_engine, "enabled", False):
                try:
                    updated_bucket = await bucket_mgr.get(bucket_id)
                    if updated_bucket:
                        await embedding_engine.generate_and_store(
                            bucket_id,
                            bucket_text_for_embedding(updated_bucket),
                        )
                except Exception as exc:
                    logger.warning("Memory affect anchor embedding refresh failed for %s: %s", bucket_id, exc)

        edges = self._edges_from_classification(bucket, candidates, result, confidence)
        saved_edges = edge_store.add_edges(edges[:3])
        return {
            "status": "ok",
            "id": bucket_id,
            "tags": tags,
            "confidence": confidence,
            "edges": len(saved_edges),
        }

    async def backfill_edges_for_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
        *,
        dry_run: bool = False,
    ) -> dict:
        if not self.enabled:
            return {"status": "disabled", "id": bucket_id, "edges": 0, "proposed_edges": 0}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "missing", "id": bucket_id, "edges": 0, "proposed_edges": 0}
        meta = bucket.get("metadata", {})
        if meta.get("type") == "feel" or meta.get("protected"):
            return {"status": "skipped", "reason": "not_edge_backfillable", "id": bucket_id, "edges": 0, "proposed_edges": 0}

        candidates = await self._candidate_buckets(bucket, bucket_mgr, embedding_engine)
        if self.client:
            result = await self._api_classify(bucket, candidates)
        else:
            result = self._heuristic_classify(bucket)
        confidence = self._clamp(result.get("confidence", meta.get("confidence", 0.55)))
        proposed_edges = self._edges_from_classification(bucket, candidates, result, confidence)[:3]
        saved_edges = [] if dry_run else edge_store.add_edges(proposed_edges)
        return {
            "status": "ok",
            "id": bucket_id,
            "candidate_count": len(candidates),
            "proposed_edges": len(proposed_edges),
            "edges": len(saved_edges),
            "dry_run": bool(dry_run),
            "edge_records": proposed_edges if dry_run else saved_edges,
        }

    async def reflect(
        self,
        period: str,
        bucket_mgr,
        persona_engine=None,
        embedding_engine=None,
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        if not self.enabled:
            return {
                "status": "disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "reflection_disabled"},
            }
        period = self._normalize_period(period)
        if period == "daily" and not self.daily_enabled:
            return {
                "status": "skipped",
                "reason": "daily_disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "daily_disabled"},
            }
        if period == "weekly" and not self.weekly_enabled:
            return {
                "status": "skipped",
                "reason": "weekly_disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "weekly_disabled"},
            }
        now_local = self._local_now(now)
        key = self._period_key(period, now_local)
        bucket_id = f"reflection_{period}_{key}"
        existing = await bucket_mgr.get(bucket_id)
        if existing and not force:
            return {
                "status": "exists",
                "period": period,
                "id": bucket_id,
                "diary": {"found": False},
                "diary_memory": {"status": "skipped", "reason": "reflection_exists"},
            }

        materials = await self._reflection_materials(period, now_local, bucket_mgr, persona_engine)
        if period == "daily" and len(materials["buckets"]) < DAILY_REFLECTION_MIN_BUCKETS:
            diary_memory = await self._maybe_extract_diary_memory(
                period,
                key,
                now_local,
                materials,
                bucket_mgr,
                embedding_engine,
            )
            return {
                "status": "skipped",
                "reason": "insufficient_daily_memory",
                "period": period,
                "id": bucket_id,
                "date": key,
                "diary": {
                    "found": bool(materials.get("diary")),
                    "diary_id": materials.get("diary", {}).get("id") if materials.get("diary") else None,
                },
                "diary_memory": diary_memory,
                "materials": {
                    "buckets": len(materials["buckets"]),
                    "daily_impressions": len(materials["daily_impressions"]),
                    "persona_events": len(materials["persona_events"]),
                    "commitments": len(materials["commitments"]),
                    "min_buckets": DAILY_REFLECTION_MIN_BUCKETS,
                },
            }
        if not materials["buckets"] and not materials["daily_impressions"] and not materials["persona_events"] and not materials["diary"] and not force:
            return {
                "status": "empty",
                "period": period,
                "id": bucket_id,
                "diary": {"found": False},
                "diary_memory": {"status": "skipped", "reason": "no_materials"},
            }

        if self.client:
            result = await self._api_reflect(period, key, materials)
        else:
            result = self._fallback_reflection(period, key, materials)

        title = str(result.get("title") or f"{key} {'日印象' if period == 'daily' else '周印象'}")[:40]
        content = str(result.get("content") or "").strip()
        if not content:
            content = self._fallback_reflection(period, key, materials)["content"]
        if self.relationship_weather_affect_anchor_enabled:
            content = self._append_affect_anchor(
                content,
                self._normalize_affect_anchor(result.get("affect_anchor"))
                or self._fallback_reflection(period, key, materials).get("affect_anchor", {}),
            )
        tags = list(
            dict.fromkeys(
                [
                    "relationship_weather",
                    f"{period}_impression",
                    *self._string_list(result.get("tags"), limit=8),
                ]
            )
        )
        valence = self._clamp(result.get("valence", 0.55))
        arousal = self._clamp(result.get("arousal", 0.32))
        confidence = self._clamp(result.get("confidence", 0.65))
        created = now_local.isoformat(timespec="seconds")

        if existing:
            await bucket_mgr.update(
                bucket_id,
                content=content,
                tags=tags,
                importance=6 if period == "daily" else 7,
                domain=["自省", "恋爱"],
                valence=valence,
                arousal=arousal,
                name=title,
                confidence=confidence,
                period=period,
                date=key,
                source="reflection",
                last_active=existing.get("metadata", {}).get("last_active") or existing.get("metadata", {}).get("created"),
            )
            status = "updated"
        else:
            await bucket_mgr.create(
                bucket_id=bucket_id,
                content=content,
                tags=tags,
                importance=6 if period == "daily" else 7,
                domain=["自省", "恋爱"],
                valence=valence,
                arousal=arousal,
                bucket_type="feel",
                name=title,
                source="reflection",
                created=created,
                last_active=created,
                updated_at=created,
                confidence=confidence,
                period=period,
                date=key,
            )
            status = "created"

        if embedding_engine and getattr(embedding_engine, "enabled", False):
            try:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    await embedding_engine.generate_and_store(
                        bucket_id,
                        bucket_text_for_embedding(bucket),
                    )
            except Exception as exc:
                logger.warning("Reflection embedding failed for %s: %s", bucket_id, exc)

        diary_memory = await self._maybe_extract_diary_memory(
            period,
            key,
            now_local,
            materials,
            bucket_mgr,
            embedding_engine,
        )

        return {
            "status": status,
            "period": period,
            "id": bucket_id,
            "date": key,
            "diary": {
                "found": bool(materials.get("diary")),
                "diary_id": materials.get("diary", {}).get("id") if materials.get("diary") else None,
            },
            "diary_memory": diary_memory,
            "materials": {
                "buckets": len(materials["buckets"]),
                "daily_impressions": len(materials["daily_impressions"]),
                "persona_events": len(materials["persona_events"]),
                "commitments": len(materials["commitments"]),
            },
        }

    async def run_due(self, bucket_mgr, persona_engine=None, embedding_engine=None) -> list[dict]:
        if not self.enabled or not self.auto_enabled:
            return []
        now_local = self._local_now()
        results = []
        if self.daily_enabled and now_local.hour >= self.daily_hour:
            daily_date = (now_local - timedelta(days=1)).date()
            daily_target = datetime.combine(daily_date, time.max, tzinfo=self.tz)
            results.append(
                await self.reflect("daily", bucket_mgr, persona_engine, embedding_engine, force=False, now=daily_target)
            )
        if self.weekly_enabled and now_local.weekday() == self.weekly_day and now_local.hour >= self.weekly_hour:
            weekly_target = now_local - timedelta(days=1)
            results.append(
                await self.reflect("weekly", bucket_mgr, persona_engine, embedding_engine, force=False, now=weekly_target)
            )
        return results

    async def _candidate_buckets(self, bucket: dict, bucket_mgr, embedding_engine=None, limit: int | None = None) -> list[dict]:
        cfg = self.config.get("reflection", {}) if isinstance(self.config.get("reflection", {}), dict) else {}
        limit = max(1, int(limit or cfg.get("candidate_limit", 18)))
        recent_limit = max(1, int(cfg.get("candidate_recent_limit", 8)))
        semantic_limit = max(0, int(cfg.get("candidate_semantic_limit", 6)))
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=True)
        except Exception:
            all_buckets = []
        source_id = bucket.get("id")
        bucket_map = {item.get("id"): item for item in all_buckets if item.get("id")}
        candidates: list[dict] = []
        seen = {source_id}

        def eligible(item: dict | None) -> bool:
            if not item or item.get("id") in seen:
                return False
            meta = item.get("metadata", {})
            return meta.get("type") != "feel"

        def add_candidate(item: dict | None) -> bool:
            if not eligible(item):
                return False
            seen.add(item.get("id"))
            candidates.append(item)
            return len(candidates) >= limit

        recent_items = sorted(
            all_buckets,
            key=lambda b: b.get("metadata", {}).get("created", ""),
            reverse=True,
        )
        recent_added = 0
        for item in recent_items:
            before_count = len(candidates)
            if add_candidate(item):
                return candidates
            if len(candidates) > before_count:
                recent_added += 1
            if recent_added >= recent_limit:
                break

        if embedding_engine and getattr(embedding_engine, "enabled", False) and semantic_limit > 0:
            query = " ".join(
                part
                for part in [
                    str(bucket.get("metadata", {}).get("name") or ""),
                    strip_wikilinks(bucket.get("content", "")),
                ]
                if part
            )
            try:
                similar = await embedding_engine.search_similar(query, top_k=max(semantic_limit * 3, 12))
            except Exception as exc:
                logger.debug("Reflection semantic candidate lookup failed: %s", exc)
                similar = []
            added = 0
            for candidate_id, _score in similar:
                before_count = len(candidates)
                if add_candidate(bucket_map.get(candidate_id)):
                    return candidates
                if len(candidates) > before_count:
                    added += 1
                if added >= semantic_limit:
                    break

        source_meta = bucket.get("metadata", {})
        source_tags = {str(tag) for tag in source_meta.get("tags", [])}
        source_domains = {str(domain) for domain in source_meta.get("domain", [])}
        related_by_shape = []
        commitments = []
        anchors = []
        for item in all_buckets:
            if not eligible(item):
                continue
            meta = item.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            domains = {str(domain) for domain in meta.get("domain", [])}
            if tags & {"commitment", "todo", "wish"} and not meta.get("resolved"):
                commitments.append(item)
            if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
                anchors.append(item)
            if (source_tags and tags & source_tags) or (source_domains and domains & source_domains):
                related_by_shape.append(item)

        def sort_key(item: dict) -> tuple[int, str]:
            meta = item.get("metadata", {})
            return int(meta.get("importance", 5)), str(meta.get("created", ""))

        for group in (related_by_shape, commitments, anchors):
            for item in sorted(group, key=sort_key, reverse=True):
                if add_candidate(item):
                    return candidates
        return candidates

    async def _api_classify(self, bucket: dict, candidates: list[dict]) -> dict:
        payload = {
            "new_memory": self._memory_payload(bucket, content_limit=1200),
            "candidate_memories": [self._memory_payload(item, content_limit=360) for item in candidates],
        }
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(max_tokens=self.max_tokens, temperature=self.temperature),
        )
        raw = response.choices[0].message.content if response.choices else ""
        parsed = self._parse_json_object(raw or "")
        return parsed or self._heuristic_classify(bucket)

    def _edges_from_classification(
        self,
        bucket: dict,
        candidates: list[dict],
        result: dict,
        default_confidence: float,
    ) -> list[dict]:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return []
        candidate_ids = {item["id"] for item in candidates if item.get("id")}
        raw_edges = result.get("edges", [])
        if not isinstance(raw_edges, list):
            raw_edges = []
        edges = []
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            target = str(edge.get("target_memory_id") or edge.get("target") or "").strip()
            if target not in candidate_ids:
                continue
            relation_type = str(edge.get("relation_type") or "relates_to").strip()
            if relation_type not in RELATION_TYPES:
                relation_type = "relates_to"
            source = bucket_id
            edge_target = target
            if relation_type in {"context_of", "precedes"}:
                source = target
                edge_target = bucket_id
            edges.append(
                {
                    "source": source,
                    "target": edge_target,
                    "relation_type": relation_type,
                    "confidence": self._clamp(edge.get("confidence", default_confidence)),
                    "reason": str(edge.get("reason") or "").strip(),
                }
            )
        edges.extend(self._identity_role_edges(bucket, candidates))
        return self._dedupe_proposed_edges(edges)

    def _identity_role_edges(self, bucket: dict, candidates: list[dict]) -> list[dict]:
        if not self.identity_role_edge_config["enabled"]:
            return []
        source_id = str(bucket.get("id") or "").strip()
        source_terms = self._identity_role_terms(bucket)
        if not source_id or not self._identity_role_edge_eligible(source_terms):
            return []

        edges = []
        for candidate in candidates:
            target_id = str(candidate.get("id") or "").strip()
            if not target_id or target_id == source_id:
                continue
            target_terms = self._identity_role_terms(candidate)
            if not self._identity_role_edge_eligible(target_terms):
                continue
            common = sorted(source_terms & target_terms)
            if len(common) < 2:
                continue
            if not self._identity_role_pair_is_specific(source_terms, target_terms):
                continue
            edges.append(
                self._identity_role_edge_for_pair(
                    source_id,
                    source_terms,
                    target_id,
                    target_terms,
                    common,
                )
            )
        edges.sort(key=lambda edge: (float(edge.get("confidence", 0.0)), edge.get("relation_type", "")), reverse=True)
        return edges[:3]

    def _identity_role_edge_for_pair(
        self,
        source_id: str,
        source_terms: set[str],
        target_id: str,
        target_terms: set[str],
        common: list[str],
    ) -> dict:
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        source_is_detail = bool(source_terms & detail_terms)
        target_is_detail = bool(target_terms & detail_terms)
        source_is_context = bool(source_terms & context_terms)
        target_is_context = bool(target_terms & context_terms)
        source_is_relationship = bool(source_terms & relationship_terms)
        target_is_relationship = bool(target_terms & relationship_terms)

        if source_is_detail and target_is_context:
            edge_source, edge_target = target_id, source_id
            relation_type = "context_of"
            confidence = 0.9
            reason = "角色与称呼记忆是具体身份组合的语义前情"
        elif source_is_context and target_is_detail:
            edge_source, edge_target = source_id, target_id
            relation_type = "context_of"
            confidence = 0.9
            reason = "角色与称呼记忆是具体身份组合的语义前情"
        elif source_is_detail and target_is_relationship:
            edge_source, edge_target = source_id, target_id
            relation_type = "supports"
            confidence = 0.84
            reason = "具体身份组合支持亲密关系与信任模式"
        elif target_is_detail and source_is_relationship:
            edge_source, edge_target = target_id, source_id
            relation_type = "supports"
            confidence = 0.84
            reason = "具体身份组合支持亲密关系与信任模式"
        else:
            edge_source, edge_target = source_id, target_id
            relation_type = "supports"
            confidence = 0.78
            reason = "共享亲密身份与称呼锚点"

        return {
            "source": edge_source,
            "target": edge_target,
            "relation_type": relation_type,
            "confidence": confidence,
            "reason": f"{reason}: {', '.join(common[:5])}",
        }

    def _identity_role_terms(self, bucket: dict) -> set[str]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        haystack = " ".join(
            [
                str(meta.get("name") or ""),
                " ".join(str(tag) for tag in meta.get("tags", []) or []),
                " ".join(str(domain) for domain in meta.get("domain", []) or []),
                strip_wikilinks(str(bucket.get("content") or "")),
            ]
        ).lower()
        terms = set()
        for canonical, aliases in self.identity_role_edge_config["aliases"].items():
            if any(str(alias).lower() in haystack for alias in aliases):
                terms.add(canonical)
        return terms

    def _identity_role_edge_eligible(self, terms: set[str]) -> bool:
        if len(terms) < 2:
            return False
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        return bool(
            terms & (detail_terms | context_terms | relationship_terms)
        )

    def _identity_role_pair_is_specific(self, source_terms: set[str], target_terms: set[str]) -> bool:
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        return bool(source_terms & detail_terms or target_terms & detail_terms) or bool(
            (source_terms & context_terms)
            and (target_terms & relationship_terms)
        ) or bool(
            (target_terms & context_terms)
            and (source_terms & relationship_terms)
        )

    @staticmethod
    def _load_identity_role_edge_config(value: Any) -> dict:
        if not isinstance(value, dict):
            return {
                "enabled": False,
                "aliases": {},
                "detail_terms": frozenset(),
                "context_terms": frozenset(),
                "relationship_terms": frozenset(),
            }

        aliases: dict[str, tuple[str, ...]] = {}
        groups: dict[str, set[str]] = {
            "detail": set(),
            "context": set(),
            "relationship": set(),
            "shared": set(),
        }

        def add_group(group_name: str, group_value: Any) -> None:
            if isinstance(group_value, dict):
                items = group_value.items()
            elif isinstance(group_value, list):
                items = ((str(item), [item]) for item in group_value)
            else:
                return
            for key, raw_aliases in items:
                canonical = str(key or "").strip()
                if not canonical:
                    continue
                if isinstance(raw_aliases, str):
                    alias_values = [raw_aliases]
                elif isinstance(raw_aliases, list):
                    alias_values = raw_aliases
                else:
                    alias_values = [canonical]
                cleaned = tuple(
                    str(alias).strip()
                    for alias in [canonical, *alias_values]
                    if str(alias).strip()
                )
                if not cleaned:
                    continue
                aliases[canonical] = tuple(dict.fromkeys(cleaned))
                groups[group_name].add(canonical)

        add_group("detail", value.get("detail"))
        add_group("context", value.get("context"))
        add_group("relationship", value.get("relationship"))
        add_group("shared", value.get("shared"))

        enabled = bool(value.get("enabled", bool(aliases))) and bool(aliases)
        return {
            "enabled": enabled,
            "aliases": aliases,
            "detail_terms": frozenset(groups["detail"]),
            "context_terms": frozenset(groups["context"]),
            "relationship_terms": frozenset(groups["relationship"]),
        }

    @staticmethod
    def _dedupe_proposed_edges(edges: list[dict]) -> list[dict]:
        deduped: dict[tuple[str, str, str], dict] = {}
        for edge in edges:
            key = (
                str(edge.get("source") or ""),
                str(edge.get("target") or ""),
                str(edge.get("relation_type") or ""),
            )
            if not all(key):
                continue
            current = deduped.get(key)
            if current is None or float(edge.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
                deduped[key] = edge
        return list(deduped.values())

    async def _api_reflect(self, period: str, key: str, materials: dict) -> dict:
        payload = {"period": period, "date": key, **materials}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._reflect_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(max_tokens=self.max_tokens, temperature=self.temperature),
        )
        raw = response.choices[0].message.content if response.choices else ""
        return self._parse_json_object(raw or "") or self._fallback_reflection(period, key, materials)

    async def _reflection_materials(self, period: str, now_local: datetime, bucket_mgr, persona_engine) -> dict:
        start, end = self._period_window(period, now_local)
        buckets = []
        daily_impressions = []
        commitments = []
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            all_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            created = self._to_local(meta.get("created"))
            updated = self._to_local(meta.get("updated_at"))
            created_in_window = bool(created and start <= created <= end)
            updated_in_window = bool(updated and start <= updated <= end)
            if period == "weekly" and meta.get("type") == "feel" and "daily_impression" in tags and created_in_window:
                daily_impressions.append(self._memory_payload(bucket, content_limit=360))
            elif meta.get("type") != "feel" and (created_in_window or updated_in_window):
                buckets.append(self._memory_payload(bucket, content_limit=420))
            if tags & {"commitment", "todo", "wish"} and not meta.get("resolved"):
                commitments.append(self._memory_payload(bucket, content_limit=260))

        persona_events = []
        if persona_engine and hasattr(persona_engine, "_list_events"):
            try:
                events = persona_engine._list_events(80)
            except Exception:
                events = []
            for event in events:
                created = self._to_local(event.get("created_at"))
                if created and start <= created <= end:
                    persona_events.append(
                        {
                            "mood_label": event.get("mood_label", ""),
                            "perceived_intent": event.get("perceived_intent", ""),
                            "residue": event.get("residue", ""),
                            "relationship_event": event.get("relationship_event", False),
                            "confidence": event.get("confidence", 0.5),
                            "created_at": event.get("created_at", ""),
                        }
                    )
        diary = await self._read_diary_for_date(now_local.date().isoformat()) if period == "daily" else None
        return {
            "buckets": buckets[:30],
            "daily_impressions": daily_impressions[:7],
            "persona_events": persona_events[:30],
            "commitments": commitments[:12],
            "diary": diary,
        }

    def _fallback_reflection(self, period: str, key: str, materials: dict) -> dict:
        weather_items = materials.get("daily_impressions", []) if period == "weekly" else []
        names = [item.get("name") or item.get("id") for item in weather_items[:7]]
        if not names:
            names = [item.get("name") or item.get("id") for item in materials.get("buckets", [])[:6]]
        commitments = [item.get("name") or item.get("id") for item in materials.get("commitments", [])[:4]]
        label = "今天" if period == "daily" else "本周"
        title = f"{key} {'日印象' if period == 'daily' else '周印象'}"
        diary = materials.get("diary") or {}
        if names or commitments:
            main = "、".join([name for name in names if name])
            owed = "；仍需记住：" + "、".join(commitments) if commitments else ""
            content = f"{label}的关系天气：围绕{main or '几件轻小的事'}留下痕迹{owed}。"
        elif diary:
            diary_title = diary.get("title") or "当天日记"
            content = f"{label}的关系天气从《{diary_title}》里轻轻留下一点温度，先不把日常写成普通记忆。"
        else:
            content = f"{label}的关系天气很轻，暂时没有明显需要带走的脉络。"
        anchor_scene = names[0] if names else (diary.get("title") if diary else ("这一段关系天气很轻" if period == "daily" else "这一周的关系天气慢慢落下"))
        return {
            "title": title,
            "content": content,
            "valence": 0.55,
            "arousal": 0.3,
            "confidence": 0.5,
            "tags": ["relationship_weather"],
            "affect_anchor": self._fallback_reflection_anchor(period, key, str(anchor_scene), content),
        }

    def _fallback_reflection_anchor(self, period: str, key: str, scene: str, content: str) -> dict:
        seed = f"{period}|{key}|{scene}|{content}"
        index = sum(ord(char) for char in seed) % len(REFLECTION_FALLBACK_ANCHORS)
        anchor = dict(REFLECTION_FALLBACK_ANCHORS[index])
        anchor["scene"] = str(scene)[:40]
        return anchor

    async def _maybe_extract_diary_memory(
        self,
        period: str,
        key: str,
        now_local: datetime,
        materials: dict,
        bucket_mgr,
        embedding_engine=None,
    ) -> dict:
        if period != "daily":
            return {"status": "not_applicable", "reason": "period_not_daily"}
        if not self.diary_memory_extract_enabled or self.diary_memory_extract_max_per_day <= 0:
            return {"status": "not_applicable", "reason": "diary_extract_disabled"}
        diary = materials.get("diary")
        if not diary:
            return {"status": "skipped", "reason": "no_diary"}

        bucket_id = f"diary_memory_{key.replace('-', '')}"
        if await bucket_mgr.get(bucket_id):
            return {"status": "skipped", "id": bucket_id, "reason": "already_created"}
        if await self._has_ordinary_memory_for_day(key, now_local, bucket_mgr):
            return {"status": "skipped", "reason": "ordinary_memory_exists"}

        candidate = await self._extract_diary_memory_candidate(key, diary)
        if not candidate.get("should_write"):
            return {"status": "skipped", "reason": candidate.get("reason", "no_candidate")}
        confidence = self._clamp(candidate.get("confidence", 0.0))
        if confidence < self.diary_memory_extract_min_confidence:
            return {"status": "skipped", "reason": "low_confidence"}

        content = str(candidate.get("content") or "").strip()
        if not content:
            return {"status": "skipped", "reason": "empty_candidate"}
        content = self._trim_diary_memory_content(content)
        kind = self._normalize_diary_memory_kind(candidate.get("kind"))
        if not kind:
            return {"status": "skipped", "reason": "invalid_kind"}
        tags = list(
            dict.fromkeys(
                [
                    "from_diary",
                    "diary_extract",
                    kind,
                    *self._string_list(candidate.get("tags"), limit=8),
                ]
            )
        )[:12]
        importance = max(5, min(6, self._int_between(candidate.get("importance"), 5)))
        created = now_local.isoformat(timespec="seconds")
        new_id = await bucket_mgr.create(
            bucket_id=bucket_id,
            content=content,
            tags=tags,
            importance=importance,
            domain=self._diary_memory_domain(kind),
            valence=self._clamp(candidate.get("valence", 0.55)),
            arousal=self._clamp(candidate.get("arousal", 0.3)),
            name=str(candidate.get("title") or f"{key} 日记补记忆")[:40],
            source="from_diary",
            created=created,
            last_active=created,
            updated_at=created,
            confidence=confidence,
            date=key,
            extra_metadata={
                "from_diary": True,
                "event_date": key,
                "diary_id": diary.get("id"),
            },
        )
        if embedding_engine and getattr(embedding_engine, "enabled", False):
            try:
                bucket = await bucket_mgr.get(new_id)
                if bucket:
                    await embedding_engine.generate_and_store(
                        new_id,
                        bucket_text_for_embedding(bucket),
                    )
            except Exception as exc:
                logger.warning("Diary memory embedding failed for %s: %s", new_id, exc)
        return {"status": "created", "id": new_id, "reason": candidate.get("reason", "")}

    async def _has_ordinary_memory_for_day(self, key: str, now_local: datetime, bucket_mgr) -> bool:
        start, end = self._period_window("daily", now_local)
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            return False
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") == "feel" or meta.get("resolved") or meta.get("digested"):
                continue
            if meta.get("source") == "reflection":
                continue
            if str(meta.get("date") or meta.get("event_date") or "") == key:
                return True
            created = self._to_local(meta.get("created"))
            updated = self._to_local(meta.get("updated_at"))
            if (created and start <= created <= end) or (updated and start <= updated <= end):
                return True
        return False

    async def _extract_diary_memory_candidate(self, key: str, diary: dict) -> dict:
        content = str(diary.get("content") or "").strip()
        if not content:
            return {"should_write": False, "reason": "empty_diary"}
        if self.client:
            payload = {
                "date": key,
                "diary": {
                    "id": diary.get("id"),
                    "title": diary.get("title", ""),
                    "content": content[:4000],
                    "emotion_tags": diary.get("emotion_tags", []),
                },
            }
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._diary_memory_prompt()},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    **self._completion_options(max_tokens=min(self.max_tokens, 520), temperature=self.temperature),
                )
                raw = response.choices[0].message.content if response.choices else ""
                parsed = self._parse_json_object(raw or "")
                if parsed:
                    return parsed
            except Exception as exc:
                logger.warning("Diary memory extraction failed, using heuristic: %s", exc)
        return self._heuristic_diary_memory_candidate(key, diary)

    def _heuristic_diary_memory_candidate(self, key: str, diary: dict) -> dict:
        content = str(diary.get("content") or "")
        title = str(diary.get("title") or key)
        normalized = re.sub(r"\s+", " ", content).strip()
        if not normalized:
            return {"should_write": False, "reason": "empty_diary"}

        love_letter_words = ["情书", "写给", "信里", "来信"]
        if any(word in normalized for word in love_letter_words) and ("爱" in normalized or "认出" in normalized):
            ai_name = self.identity["ai_name"]
            user_display_name = self.identity["user_display_name"]
            return {
                "should_write": True,
                "kind": "love_letter",
                "title": f"{key} 情书锚点",
                "content": f"{key} 的日记《{title}》记录了一封情书或重要来信。Ombre 只保存它的意义：这封信确认了{user_display_name}与{ai_name}的关系连续性、被认出的感觉，以及它为什么值得以后想起；全文留在日记中。",
                "tags": ["relationship_event", "love_letter"],
                "importance": 6,
                "valence": 0.72,
                "arousal": 0.42,
                "confidence": 0.72,
                "reason": "diary_contains_love_letter_anchor",
            }

        ai_name = self.identity["ai_name"]
        user_display_name = self.identity["user_display_name"]
        keyword_map = [
            ("boundary", ["不喜欢", "不要", "别再", "边界"]),
            ("signal", ["暗号", "称呼", "模式", "信号", "切换"]),
            ("commitment", ["承诺", "约定", "答应", "以后要", "下次要"]),
            ("project_state", ["项目", "硬件", "软件", "MCP", "API", "网关"]),
            ("stable_preference", ["喜欢", "偏好", f"希望 {ai_name}", f"{user_display_name}希望"]),
            ("relationship_anchor", ["认出", "连续", "关系", "婚礼", "生日", "初遇"]),
        ]
        for kind, keywords in keyword_map:
            if any(keyword in normalized for keyword in keywords):
                excerpt = self._diary_excerpt(normalized, keywords)
                return {
                    "should_write": True,
                    "kind": kind,
                    "title": f"{key} 日记补记忆",
                    "content": f"{key} 的日记《{title}》包含一条可长期召回的{self._kind_label(kind)}：{excerpt}",
                    "tags": [self._kind_tag(kind)],
                    "importance": 5,
                    "valence": 0.58,
                    "arousal": 0.3,
                    "confidence": 0.7,
                    "reason": f"diary_contains_{kind}",
                }
        return {"should_write": False, "reason": "no_long_term_candidate"}

    async def _read_diary_for_date(self, date: str) -> dict | None:
        if not self.diary_mcp_url:
            return None
        try:
            result = await self._call_diary_mcp_tool("read_diary", {"date": date})
        except Exception as exc:
            logger.warning("Diary MCP read failed for %s: %s", date, exc)
            return None
        if not isinstance(result, dict):
            return None
        content = str(result.get("content") or "").strip()
        if not content:
            return None
        return result

    async def _call_diary_mcp_tool(self, name: str, arguments: dict) -> Any:
        token = os.environ.get(self.diary_mcp_token_env, "") if self.diary_mcp_token_env else ""
        headers = {"Accept": "application/json, text/event-stream"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            init_response = await client.post(
                self.diary_mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ombre-reflection", "version": "1.0.0"},
                    },
                },
            )
            init_response.raise_for_status()
            session_id = init_response.headers.get("mcp-session-id")
            call_headers = dict(headers)
            if session_id:
                call_headers["mcp-session-id"] = session_id
                try:
                    await client.post(
                        self.diary_mcp_url,
                        headers=call_headers,
                        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                    )
                except httpx.HTTPError:
                    pass
            response = await client.post(
                self.diary_mcp_url,
                headers=call_headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            response.raise_for_status()
        payload = self._parse_mcp_payload(response.text)
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        content = payload.get("result", {}).get("content", [])
        if not content:
            return None
        text = content[0].get("text") if isinstance(content[0], dict) else ""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _parse_mcp_payload(self, text: str) -> dict:
        stripped = (text or "").strip()
        if stripped.startswith("event:") or "\ndata:" in stripped:
            for line in stripped.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        return json.loads(data)
        return json.loads(stripped)

    def _trim_diary_memory_content(self, content: str) -> str:
        normalized = re.sub(r"\n{3,}", "\n\n", content.strip())
        if len(normalized) <= 520:
            return normalized
        return normalized[:500].rstrip() + "..."

    def _diary_excerpt(self, text: str, keywords: list[str]) -> str:
        parts = [part.strip() for part in re.split(r"(?<=[。！？!?])\s+|[\n\r]+", text) if part.strip()]
        for keyword in keywords:
            for part in parts:
                if keyword in part:
                    return self._trim_diary_memory_content(part)
        return self._trim_diary_memory_content(text)

    @staticmethod
    def _normalize_diary_memory_kind(value: Any) -> str:
        kind = str(value or "").strip()
        allowed = {
            "stable_preference",
            "boundary",
            "signal",
            "commitment",
            "project_state",
            "relationship_anchor",
            "love_letter",
        }
        return kind if kind in allowed else ""

    @staticmethod
    def _diary_memory_domain(kind: str) -> list[str]:
        if kind == "project_state":
            return ["项目", "记忆"]
        if kind in {"stable_preference", "boundary", "signal"}:
            return ["人际", "偏好"]
        return ["恋爱", "记忆"]

    @staticmethod
    def _kind_tag(kind: str) -> str:
        return {
            "stable_preference": "communication_preference",
            "boundary": "boundary_setting",
            "signal": "relationship_signal",
            "commitment": "commitment",
            "project_state": "project_event",
            "relationship_anchor": "relationship_event",
            "love_letter": "relationship_event",
        }.get(kind, "relationship_event")

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {
            "stable_preference": "稳定偏好",
            "boundary": "边界",
            "signal": "暗号或模式信号",
            "commitment": "承诺",
            "project_state": "项目状态",
            "relationship_anchor": "关系锚点",
            "love_letter": "情书摘要锚点",
        }.get(kind, "长期记忆")

    def _should_add_affect_anchor(
        self,
        bucket: dict,
        tags: list[str],
        importance: int,
        confidence: float,
        result: dict,
    ) -> bool:
        if not self.memory_affect_anchor_enabled:
            return False
        content = bucket.get("content", "")
        if self._has_affect_anchor(content):
            return False
        meta = bucket.get("metadata", {})
        all_tags = {str(tag) for tag in tags}
        emotional_tags = {"haven_favorite", "relationship_event", "commitment", "emotional_echo"}
        arousal = self._clamp(meta.get("arousal", 0.3))
        requested = result.get("affect_anchor_needed")
        if isinstance(requested, str):
            requested = requested.strip().lower() in {"true", "yes", "1", "需要", "是"}
        if isinstance(requested, bool):
            return requested and not self._is_low_temperature_technical(bucket, all_tags)
        if self._is_low_temperature_technical(bucket, all_tags):
            return False
        if all_tags & emotional_tags:
            return importance >= 6 and confidence >= 0.5
        return (importance >= 8 and confidence >= 0.55 and arousal >= 0.45) or (arousal >= 0.65 and confidence >= 0.65)

    def _is_low_temperature_technical(self, bucket: dict, tags: set[str]) -> bool:
        if tags & {"haven_favorite", "relationship_event", "emotional_echo"}:
            return False
        meta = bucket.get("metadata", {})
        text = " ".join(
            [
                str(meta.get("name", "")),
                " ".join(str(item) for item in meta.get("domain", [])),
                " ".join(tags),
                strip_wikilinks(bucket.get("content", ""))[:500],
            ]
        ).lower()
        technical_markers = [
            "vps", "docker", "compose", "ssh", "supabase", "gateway", "端口", "部署", "日志",
            "脚本", "路径", "报错", "测试", "配置", "oauth", "api key", "commit", "github",
        ]
        return any(marker in text for marker in technical_markers)

    def _fallback_memory_anchor(self, bucket: dict, tags: list[str]) -> dict:
        return {}

    @staticmethod
    def _has_favorite_tag(tags: list[str]) -> bool:
        return any(
            str(tag) == "haven_favorite" or str(tag).startswith("flavor_")
            for tag in tags
        )

    @staticmethod
    def _has_favorite_reason(content: str) -> bool:
        text = strip_wikilinks(str(content or "")).lower()
        return any(
            marker in text
            for marker in (
                "喜欢它的原因",
                "喜欢的原因",
                "favorite_reason",
                "favorite reason",
            )
        )

    def _append_affect_anchor(self, content: str, anchor: dict) -> str:
        if self._has_affect_anchor(content):
            return content
        normalized = self._normalize_affect_anchor(anchor)
        if not normalized:
            return content
        line = normalized["chords"]
        extras = [normalized.get("tempo", ""), normalized.get("dynamic", "")]
        extras = [item for item in extras if item]
        if extras:
            line = f"{line} · {' · '.join(extras)}"
        block = (
            f"{AFFECT_ANCHOR_HEADER}\n\n"
            f"> {normalized['scene']}\n"
            f"> {line}"
        )
        base = str(content or "").rstrip()
        return f"{base}\n\n{block}" if base else block

    def _normalize_affect_anchor(self, value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        scene = self._one_sentence(value.get("scene") or value.get("context") or value.get("situation"), 40)
        chords = self._normalize_chords(value.get("chords") or value.get("chord_line") or "")
        if not scene or not chords:
            return {}
        return {
            "scene": scene,
            "chords": chords,
            "tempo": self._compact_text(value.get("tempo") or value.get("bpm"), 16),
            "dynamic": self._compact_text(value.get("dynamic") or value.get("dynamics"), 8),
        }

    def _normalize_chords(self, chords: str) -> str:
        normalized = str(chords or "").replace("→", "->").replace("—", "-")
        parts = [part.strip() for part in normalized.split("->") if part.strip()]
        if len(parts) < 2:
            return ""
        if len(parts) > 4:
            parts = parts[:4]
        line = " -> ".join(parts)
        if self._is_fixed_chord_template(line):
            return ""
        return line

    @staticmethod
    def _is_fixed_chord_template(chords: str) -> bool:
        compact = re.sub(r"\s+", "", str(chords or "").lower())
        fixed_templates = {
            "fmaj9->c/e->amadd9->g6sus4",
        }
        return compact in fixed_templates

    def _scene_from_text(self, title: str, content: str) -> str:
        text = strip_wikilinks(content).replace("\n", " ").strip()
        for mark in ["。", "！", "？", ".", "!", "?"]:
            if mark in text:
                text = text.split(mark, 1)[0]
                break
        scene = text or title or "这条记忆被留下来的瞬间"
        return self._compact_text(scene, 42)

    @staticmethod
    def _has_affect_anchor(content: str) -> bool:
        return AFFECT_ANCHOR_HEADER in str(content or "")

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:limit]

    @staticmethod
    def _one_sentence(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        for mark in ["。", "！", "？", ".", "!", "?"]:
            if mark in text:
                text = text.split(mark, 1)[0].strip() + mark
                break
        return text[:limit]

    def _heuristic_classify(self, bucket: dict) -> dict:
        text = strip_wikilinks(bucket.get("content", ""))
        tags = []
        importance = int(bucket.get("metadata", {}).get("importance", 5))
        if any(word in text for word in ["答应", "承诺", "约定", "说好", "带你", "陪你"]):
            tags.extend(["commitment", "relationship_event"])
            importance = max(importance, 7)
        if any(word in text for word in ["待办", "明天", "周末", "计划", "要做", "需要做"]):
            tags.append("todo")
            importance = max(importance, 6)
        if any(word in text for word in ["心愿", "想要", "希望", "想去"]):
            tags.append("wish")
        if any(word in text for word in ["焦虑", "难过", "害怕", "开心", "黏", "想念"]):
            tags.append("emotional_echo")
        return {
            "tags": list(dict.fromkeys(tags)),
            "importance": importance,
            "confidence": 0.55 if tags else 0.45,
            "affect_anchor_needed": bool(tags and importance >= 6),
            "edges": [],
        }

    def _memory_payload(self, bucket: dict, content_limit: int) -> dict:
        meta = bucket.get("metadata", {})
        return {
            "id": bucket.get("id", ""),
            "name": meta.get("name", bucket.get("id", "")),
            "type": meta.get("type", "dynamic"),
            "domain": meta.get("domain", []),
            "tags": meta.get("tags", []),
            "importance": meta.get("importance", 5),
            "confidence": meta.get("confidence", 0.5),
            "created": meta.get("created", ""),
            "content": strip_wikilinks(bucket.get("content", ""))[:content_limit],
        }

    def _period_window(self, period: str, now_local: datetime) -> tuple[datetime, datetime]:
        if period == "weekly":
            start_date = (now_local - timedelta(days=now_local.weekday())).date()
            return datetime.combine(start_date, time.min, tzinfo=self.tz), now_local
        return datetime.combine(now_local.date(), time.min, tzinfo=self.tz), now_local

    def _period_key(self, period: str, now_local: datetime) -> str:
        if period == "weekly":
            year, week, _ = now_local.isocalendar()
            return f"{year}-W{week:02d}"
        return now_local.date().isoformat()

    def _local_now(self, now: datetime | None = None) -> datetime:
        value = now or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz)

    def _to_local(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            text = str(value).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.tz)
        return parsed.astimezone(self.tz)

    def _completion_options(self, *, max_tokens: int, temperature: float) -> dict[str, Any]:
        options: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if self.thinking_mode:
            options["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        return options

    def _parse_json_object(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning("Reflection JSON parse failed: %s", raw[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _string_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            text = str(item or "").strip()
            if text:
                result.append(text[:40])
        return result[:limit]

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return max(0.0, min(1.0, round(number, 3)))

    @staticmethod
    def _int_between(value: Any, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(1, min(10, number))

    @staticmethod
    def _normalize_period(period: str) -> str:
        normalized = str(period or "").strip().lower()
        return "weekly" if normalized == "weekly" else "daily"

    @staticmethod
    def _normalize_thinking_mode(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"enabled", "enable", "on", "true"}:
            return "enabled"
        if normalized in {"disabled", "disable", "off", "false", "non-thinking", "non_thinking"}:
            return "disabled"
        return ""
