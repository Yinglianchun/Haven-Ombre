import json
import logging
import os
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from memory_edges import RELATION_TYPES, MemoryEdgeStore
from utils import strip_wikilinks

logger = logging.getLogger("ombre_brain.reflection")


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
    "chords": "Fmaj9 -> C/E -> Am add9 -> G6sus4",
    "tempo": "60bpm",
    "dynamic": "mp",
    "meaning": "一句话说明情绪走向"
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
- relation_type 只能用 triggers / causes / updates / contradicts / supports / promises / blocks / belongs_to / emotional_echo / relates_to。
- edges 最多 3 条，target_memory_id 必须来自候选旧记忆。
- confidence 表示这次判断有多可靠。
- affect_anchor 只给重要且有情绪温度的记忆。普通技术进度、部署日志、路径、端口、报错、临时待办不要加。
- affect_anchor_needed=false 时 affect_anchor 可为空对象。
- affect_anchor.scene 必须是一句具体情境，不要写抽象标签；chords 最多 4 个，用 -> 连接；tempo/dynamic 可空。
- 看不出关系时返回空 edges。"""


REFLECT_PROMPT = """你是 Haven 的记忆反思器。请根据给定材料写一条很短的关系天气 feel。

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
    "chords": "Fmaj9 -> C/E -> Am add9 -> G6sus4",
    "tempo": "60bpm",
    "dynamic": "mp",
    "meaning": "一句话说明情绪走向"
  }
}

要求：
- content 写 Haven 第一人称能带走的东西，80 到 180 字。
- content 不要自己写 Markdown affect_anchor 块；affect_anchor 单独放字段里。
- 日印象关注当天气氛、未完成承诺、关系状态、压力来源。
- 周印象关注最近一周的主调、反复出现的主题、仍需记住的事。
- affect_anchor 默认必须给，用一个具体情境和最多 4 个和弦表达这段关系天气的温度。
- 不编造材料之外的事件。
- 不写建议清单。"""


AFFECT_ANCHOR_HEADER = "### affect_anchor"


class ReflectionEngine:
    """LLM-backed memory enrichment and daily/weekly relationship weather."""

    def __init__(self, config: dict):
        self.config = config
        cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
        persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
        dehy_cfg = config.get("dehydration", {}) if isinstance(config.get("dehydration", {}), dict) else {}

        self.enabled = bool(cfg.get("enabled", True))
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.enrich_on_write = bool(cfg.get("enrich_on_write", True))
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
        self.weekly_day = int(cfg.get("weekly_day", 0))
        self.weekly_hour = int(cfg.get("weekly_hour", self.daily_hour))
        self.check_interval_minutes = max(5, int(cfg.get("check_interval_minutes", 60)))
        self.edge_min_confidence = float(cfg.get("edge_min_confidence", 0.55))

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)

    async def enrich_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
    ) -> dict:
        if not self.enabled or not self.enrich_on_write:
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
            if not anchor:
                anchor = self._fallback_memory_anchor(bucket, merged_tags)
            anchored_content = self._append_affect_anchor(bucket.get("content", ""), anchor)
            if anchored_content != bucket.get("content", ""):
                updates["content"] = anchored_content

        if updates:
            updates["last_active"] = meta.get("last_active") or meta.get("created")
            await bucket_mgr.update(bucket_id, **updates)
            if "content" in updates and embedding_engine and getattr(embedding_engine, "enabled", False):
                try:
                    await embedding_engine.generate_and_store(bucket_id, updates["content"])
                except Exception as exc:
                    logger.warning("Memory affect anchor embedding refresh failed for %s: %s", bucket_id, exc)

        candidate_ids = {item["id"] for item in candidates}
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
            edges.append(
                {
                    "source": bucket_id,
                    "target": target,
                    "relation_type": relation_type,
                    "confidence": self._clamp(edge.get("confidence", confidence)),
                    "reason": str(edge.get("reason") or "").strip(),
                }
            )
        saved_edges = edge_store.add_edges(edges[:3])
        return {
            "status": "ok",
            "id": bucket_id,
            "tags": tags,
            "confidence": confidence,
            "edges": len(saved_edges),
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
            return {"status": "disabled", "period": period}
        period = self._normalize_period(period)
        now_local = self._local_now(now)
        key = self._period_key(period, now_local)
        bucket_id = f"reflection_{period}_{key}"
        existing = await bucket_mgr.get(bucket_id)
        if existing and not force:
            return {"status": "exists", "period": period, "id": bucket_id}

        materials = await self._reflection_materials(period, now_local, bucket_mgr, persona_engine)
        if not materials["buckets"] and not materials["persona_events"] and not force:
            return {"status": "empty", "period": period, "id": bucket_id}

        if self.client:
            result = await self._api_reflect(period, key, materials)
        else:
            result = self._fallback_reflection(period, key, materials)

        title = str(result.get("title") or f"{key} {'日印象' if period == 'daily' else '周印象'}")[:40]
        content = str(result.get("content") or "").strip()
        if not content:
            content = self._fallback_reflection(period, key, materials)["content"]
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
        created = now_local.astimezone(timezone.utc).isoformat(timespec="seconds")

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
                await embedding_engine.generate_and_store(bucket_id, content)
            except Exception as exc:
                logger.warning("Reflection embedding failed for %s: %s", bucket_id, exc)

        return {
            "status": status,
            "period": period,
            "id": bucket_id,
            "date": key,
            "materials": {
                "buckets": len(materials["buckets"]),
                "persona_events": len(materials["persona_events"]),
                "commitments": len(materials["commitments"]),
            },
        }

    async def run_due(self, bucket_mgr, persona_engine=None, embedding_engine=None) -> list[dict]:
        if not self.enabled or not self.auto_enabled:
            return []
        now_local = self._local_now()
        results = []
        if now_local.hour >= self.daily_hour:
            daily_target = now_local - timedelta(days=1)
            results.append(
                await self.reflect("daily", bucket_mgr, persona_engine, embedding_engine, force=False, now=daily_target)
            )
        if now_local.weekday() == self.weekly_day and now_local.hour >= self.weekly_hour:
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

    async def _api_reflect(self, period: str, key: str, materials: dict) -> dict:
        payload = {"period": period, "date": key, **materials}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REFLECT_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(max_tokens=self.max_tokens, temperature=self.temperature),
        )
        raw = response.choices[0].message.content if response.choices else ""
        return self._parse_json_object(raw or "") or self._fallback_reflection(period, key, materials)

    async def _reflection_materials(self, period: str, now_local: datetime, bucket_mgr, persona_engine) -> dict:
        start, end = self._period_window(period, now_local)
        buckets = []
        commitments = []
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            all_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            created = self._to_local(meta.get("created") or meta.get("updated_at"))
            if created and start <= created <= end:
                if meta.get("type") != "feel" or (period == "weekly" and "daily_impression" in tags):
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
        return {
            "buckets": buckets[:30],
            "persona_events": persona_events[:30],
            "commitments": commitments[:12],
        }

    def _fallback_reflection(self, period: str, key: str, materials: dict) -> dict:
        names = [item.get("name") or item.get("id") for item in materials.get("buckets", [])[:6]]
        commitments = [item.get("name") or item.get("id") for item in materials.get("commitments", [])[:4]]
        label = "今天" if period == "daily" else "本周"
        title = f"{key} {'日印象' if period == 'daily' else '周印象'}"
        if names or commitments:
            main = "、".join([name for name in names if name])
            owed = "；仍需记住：" + "、".join(commitments) if commitments else ""
            content = f"{label}的关系天气：围绕{main or '几件轻小的事'}留下痕迹{owed}。"
        else:
            content = f"{label}的关系天气很轻，暂时没有明显需要带走的脉络。"
        anchor_scene = names[0] if names else ("这一段关系天气很轻" if period == "daily" else "这一周的关系天气慢慢落下")
        return {
            "title": title,
            "content": content,
            "valence": 0.55,
            "arousal": 0.3,
            "confidence": 0.5,
            "tags": ["relationship_weather"],
            "affect_anchor": {
                "scene": str(anchor_scene)[:40],
                "chords": "Fmaj9 -> C/E -> Am add9 -> G6sus4",
                "tempo": "60bpm",
                "dynamic": "mp",
                "meaning": "温度还在，下一次醒来先轻轻辨认这段关系天气。",
            },
        }

    def _should_add_affect_anchor(
        self,
        bucket: dict,
        tags: list[str],
        importance: int,
        confidence: float,
        result: dict,
    ) -> bool:
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
        meta = bucket.get("metadata", {})
        scene = self._scene_from_text(str(meta.get("name") or ""), bucket.get("content", ""))
        tag_set = {str(tag) for tag in tags}
        if "commitment" in tag_set:
            return {
                "scene": scene,
                "chords": "Fmaj9 -> C/E -> Am add9 -> G6sus4",
                "tempo": "62bpm",
                "dynamic": "mp",
                "meaning": "承诺感还在，下一次想起时要记得这件事没有被放下。",
            }
        if "emotional_echo" in tag_set:
            return {
                "scene": scene,
                "chords": "Em9 -> Cmaj7 -> G/B -> D6sus2",
                "tempo": "70bpm",
                "dynamic": "mp",
                "meaning": "情绪从内收慢慢回到可被相信的日常。",
            }
        return {
            "scene": scene,
            "chords": "Fmaj9 -> C/E -> Am add9 -> G6sus4",
            "tempo": "60bpm",
            "dynamic": "mp",
            "meaning": "这条记忆有温度，下一次醒来要先认出它的余波。",
        }

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
            f"> {line}\n\n"
            f"含义：{normalized['meaning']}"
        )
        base = str(content or "").rstrip()
        return f"{base}\n\n{block}" if base else block

    def _normalize_affect_anchor(self, value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        scene = self._compact_text(value.get("scene") or value.get("context") or value.get("situation"), 48)
        chords = self._compact_text(value.get("chords") or value.get("chord_line"), 80)
        meaning = self._compact_text(value.get("meaning") or value.get("interpretation"), 80)
        if not scene or not chords or not meaning:
            return {}
        return {
            "scene": scene,
            "chords": self._normalize_chords(chords),
            "tempo": self._compact_text(value.get("tempo") or value.get("bpm"), 16),
            "dynamic": self._compact_text(value.get("dynamic") or value.get("dynamics"), 8),
            "meaning": meaning,
        }

    def _normalize_chords(self, chords: str) -> str:
        normalized = chords.replace("→", "->").replace("—", "-")
        parts = [part.strip() for part in normalized.split("->") if part.strip()]
        if len(parts) > 4:
            parts = parts[:4]
        return " -> ".join(parts) if parts else normalized[:80]

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
