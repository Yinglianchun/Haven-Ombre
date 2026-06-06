import logging
import hashlib
import os
import re
import secrets
import json
import codecs
import time
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from dream_engine import DreamEngine
from embedding_engine import EmbeddingEngine
from identity import identity_names
from gateway_state import GatewayStateStore
from memory_diffusion import (
    diffuse_memory,
    diffusion_options_from_config,
    path_has_caution,
    path_has_old_version,
    seed_scores_for_buckets,
    should_suppress_context_candidate,
)
from memory_edges import MemoryEdgeStore
from memory_moments import MemoryMomentStore, parse_bucket_moments
from memory_relevance import (
    active_facets,
    facets_for_node,
    facets_for_text,
    memory_relevance_options_from_config,
    query_has_facet,
    recall_rank,
    recall_search_query,
    relevance_multiplier,
)
from memory_layers import (
    CONTEXT_ONLY_SECTIONS,
    bucket_layer_debug,
    bucket_runtime_gate_debug,
    can_bucket_be_recent_context,
    can_bucket_be_related_target,
    can_moment_be_direct_seed,
    can_moment_be_recall_context,
    can_moment_be_related_target,
    moment_layer_debug,
    moment_runtime_gate_debug,
)
from recall_policy import RecallPolicy
from memory_nodes import MemoryNodeStore
from persona_engine import PersonaStateEngine
from reranker_engine import RerankerEngine
from source_refs import source_ref_window
from utils import (
    count_tokens_approx,
    bucket_text_for_embedding,
    load_config,
    setup_logging,
    strip_display_temperature_sections,
    strip_temperature_meaning_lines,
    strip_wikilinks,
)

logger = logging.getLogger("ombre_brain.gateway")
FAVORITE_MEMORY_MARKER = "[[ombre:favorite]]"
RETRYABLE_UPSTREAM_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}
QUERY_PLANNER_GENERIC_TERMS = {
    "recent",
    "memory",
    "context",
    "current",
    "remember",
    "emotion",
    "status",
    "thing",
    "user",
    "assistant",
    "最近",
    "记忆",
    "上下文",
    "当前",
    "现在",
    "记得",
    "情绪",
    "状态",
    "事情",
    "用户",
    "助手",
    "聊天",
    "对话",
}
MEMORY_DETAIL_REQUEST_RE = re.compile(
    r"^\s*\[memory_detail\s+ids\s*=\s*([\"'])(?P<ids>[^\"']+)\1\s*\]\s*",
    re.IGNORECASE,
)
EXPLICIT_BUCKET_ID_RE = re.compile(
    r"\[bucket_id:(?P<bracket>[^\]\s]+)\]"
    r"|(?:bucket_id|bucket id|bucket-id|记忆桶|桶id|桶ID)\s*[:=：]\s*(?P<plain>[A-Za-z0-9_.:-]+)",
    re.IGNORECASE,
)
EXPLICIT_MOMENT_ID_RE = re.compile(
    r"\[moment_id:(?P<bracket>[^\]\s]+)\]"
    r"|(?:moment_id|moment id|moment-id|片段id|片段ID)\s*[:=：]\s*(?P<plain>[A-Za-z0-9_.:-]+)",
    re.IGNORECASE,
)
QUERY_PLANNER_SYSTEM_PROMPT = """You are Ombre Memory Query Planner.
Return only strict JSON. Do not write memory. Do not choose final memories.
Split the user's long mixed message into 1-3 short memory search anchors.
Each query must be concrete and should preserve names, projects, people, places, or events.
For a short emotional reason lookup, preserve compound emotion phrases such as 激动哭, 感动哭, or 难过原因 when they are the user's actual anchor.
Each query must include must_terms: concrete words that a candidate memory should contain at least one of.
Do not include generic terms such as recent, memory, context, current, remember, emotion, status, or the single word 哭.
If the message is too vague or has no searchable memory anchor, return should_search=false.
Schema:
{
  "should_search": true,
  "too_vague": false,
  "queries": [
    {
      "query": "short search anchor",
      "must_terms": ["concrete", "terms"],
      "intent": "short reason",
      "risk": "low|medium|high"
    }
  ]
}
"""
EXTERNAL_CONTEXT_ATTACHMENT_RE = re.compile(
    r"<attachment\b[^>]*>[\s\S]*?</attachment>",
    re.IGNORECASE,
)
SELF_CLOSING_ATTACHMENT_RE = re.compile(
    r"<attachment\b[^>]*/>",
    re.IGNORECASE,
)
WORKSPACE_ATTACHMENT_RE = re.compile(
    r"<workspace_attachment>[\s\S]*?</workspace_attachment>",
    re.IGNORECASE,
)
LEADING_PROXY_SENDER_RE = re.compile(
    r"^\s*<proxy_sender\b[^>]*/>\s*",
    re.IGNORECASE,
)
LEADING_SYSTEM_PROMPT_RE = re.compile(
    r"^\s*【\s*系统提示[^】]*】\s*",
)
EXTERNAL_CONTEXT_BLOCK_TITLES = {
    "当前时间",
    "当前电量",
    "当前天气",
    "当前位置",
    "当前屏幕应用",
    "应用使用时长",
    "最近通知",
    "相关记忆",
    "屏幕文本",
}
MOMENT_SECTION_LABELS = {
    "body": "body",
    "moment": "moment",
    "fact": "fact",
    "original": "original",
    "evidence_context": "evidence_context",
    "context": "context",
    "reflection": "reflection",
    "feeling": "feeling",
    "followup": "followup",
    "affect_anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "comment": "year_ring",
}
MOMENT_TEMPERATURE_SECTIONS = CONTEXT_ONLY_SECTIONS
PROFILE_CONTEXT_SECTIONS = ("evidence_context", "context", "reflection", "feeling", "followup", "comment")


class GatewayService:
    """
    OpenAI-compatible gateway that injects Ombre memory before forwarding
    chat completions upstream.
    """

    def __init__(
        self,
        config: dict,
        bucket_mgr: BucketManager | None = None,
        dehydrator: Dehydrator | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        reranker_engine: RerankerEngine | None = None,
        state_store: GatewayStateStore | None = None,
        persona_engine: PersonaStateEngine | None = None,
        dream_engine: DreamEngine | None = None,
        memory_node_store: MemoryNodeStore | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.identity = identity_names(config)
        self.gateway_cfg = config.get("gateway", {})
        self.bucket_mgr = bucket_mgr or BucketManager(config)
        self.dehydrator = dehydrator or Dehydrator(config)
        self.embedding_engine = embedding_engine or EmbeddingEngine(config)
        self.reranker_engine = reranker_engine or RerankerEngine(config)
        self.memory_edge_store = MemoryEdgeStore(config)
        self.memory_node_store = memory_node_store or MemoryNodeStore(config)
        self.memory_moment_store = MemoryMomentStore(config)
        self.relevance_options = memory_relevance_options_from_config(config)
        self.state_store = state_store or GatewayStateStore(
            os.path.join(config["buckets_dir"], "gateway_state.db")
        )
        self.persona_engine = persona_engine or PersonaStateEngine(config)
        self.dream_engine = dream_engine or DreamEngine(config)
        self.dream_cfg = config.get("dream", {}) if isinstance(config.get("dream", {}), dict) else {}
        self.dream_inject_enabled = bool(self.dream_cfg.get("inject_enabled", False))
        self.dream_retain_after_inject = bool(self.dream_cfg.get("retain_after_inject", False))
        self.gateway_token = os.environ.get("OMBRE_GATEWAY_TOKEN", "")
        self.upstream_api_key = os.environ.get("OMBRE_GATEWAY_UPSTREAM_API_KEY", "")
        self.upstream_base_url = self.gateway_cfg.get("upstream_base_url", "").rstrip("/")
        self.upstream_default_model = self.gateway_cfg.get("upstream_default_model", "")
        self.default_session_id = str(self.gateway_cfg.get("default_session_id") or "xiaoyu-main").strip()
        self.upstream_models = self._normalize_model_list(
            self.gateway_cfg.get("upstream_models", []),
            self.upstream_default_model,
        )
        self.upstreams = self._load_upstreams()
        self.upstream_models = self._aggregate_upstream_models()
        if not self.upstream_default_model:
            for upstream in self.upstreams:
                default_model = upstream.get("default_model") or ""
                if default_model:
                    self.upstream_default_model = default_model
                    break

        self.head_recent_hours = int(self.gateway_cfg.get("head_recent_hours", 72))
        self.recent_context_reentry_idle_hours = float(
            self.gateway_cfg.get("recent_context_reentry_idle_hours", 24)
        )
        self.recent_context_cooldown_hours = float(
            self.gateway_cfg.get("recent_context_cooldown_hours", 6)
        )
        self.dynamic_top_k = int(self.gateway_cfg.get("dynamic_top_k", 10))
        self.inject_max_cards = max(0, min(2, int(self.gateway_cfg.get("inject_max_cards", 2))))
        self.skip_recent_rounds = max(0, int(self.gateway_cfg.get("skip_recent_rounds", 5)))
        self.cooldown_hours = float(self.gateway_cfg.get("cooldown_hours", 6))
        self.cooldown_floor = float(self.gateway_cfg.get("cooldown_floor", 0.3))

        self.inject_total_budget = int(self.gateway_cfg.get("inject_total_budget", 1200))
        self.core_budget = int(self.gateway_cfg.get("core_memory_budget", 500))
        self.recent_budget = int(self.gateway_cfg.get("recent_context_budget", 300))
        self.recalled_budget = int(self.gateway_cfg.get("recalled_memory_budget", 400))
        self.direct_render_mode = self._normalize_direct_render_mode(
            self.gateway_cfg.get("direct_render_mode", "auto")
        )
        self.retrieval_mode = self._normalize_retrieval_mode(
            self.gateway_cfg.get("retrieval_mode", "graph")
        )
        self.relationship_weather_budget = int(self.gateway_cfg.get("relationship_weather_budget", 220))
        self.relationship_weather_include_weekly = bool(
            self.gateway_cfg.get("relationship_weather_include_weekly", False)
        )
        self.favorite_memory_budget = int(self.gateway_cfg.get("favorite_memory_budget", 180))
        self.favorite_memory_max_cards = max(0, int(self.gateway_cfg.get("favorite_memory_max_cards", 1)))
        self.related_memory_budget = int(self.gateway_cfg.get("related_memory_budget", 220))
        self.diffusion_options = diffusion_options_from_config(config)
        self.core_memory_interval_rounds = max(0, int(self.gateway_cfg.get("core_memory_interval_rounds", 0)))
        self.portrait_memory_enabled = self._bool_config_value(
            self.gateway_cfg.get("portrait_memory_enabled"),
            False,
        )
        self.portrait_memory_budget = max(120, int(self.gateway_cfg.get("portrait_memory_budget", 360)))
        self.portrait_memory_max_sources = max(
            1,
            min(20, int(self.gateway_cfg.get("portrait_memory_max_sources", 8))),
        )
        self.portrait_memory_include_anchors = self._bool_config_value(
            self.gateway_cfg.get("portrait_memory_include_anchors"),
            True,
        )
        self._portrait_memory_cache: dict[str, Any] = {
            "key": None,
            "block": "",
            "debug": self._portrait_memory_debug_base(),
        }
        self.current_inner_state_interval_rounds = max(
            0, int(self.gateway_cfg.get("current_inner_state_interval_rounds", 15))
        )
        self.relationship_weather_interval_rounds = max(
            0, int(self.gateway_cfg.get("relationship_weather_interval_rounds", 0))
        )
        self.favorite_memory_interval_rounds = max(
            0, int(self.gateway_cfg.get("favorite_memory_interval_rounds", 0))
        )

        self.semantic_weight = float(self.gateway_cfg.get("semantic_weight", 0.45))
        self.keyword_weight = float(self.gateway_cfg.get("keyword_weight", 0.35))
        self.importance_weight = float(self.gateway_cfg.get("importance_weight", 0.10))
        self.freshness_weight = float(self.gateway_cfg.get("freshness_weight", 0.10))
        self.first_card_min_score = float(self.gateway_cfg.get("first_card_min_score", 0.55))
        self.second_card_min_score = float(self.gateway_cfg.get("second_card_min_score", 0.50))
        self.second_card_relative_score = float(
            self.gateway_cfg.get("second_card_relative_score", 0.85)
        )
        self.high_confidence_semantic_score = float(
            self.gateway_cfg.get("high_confidence_semantic_score", 0.72)
        )
        self.high_confidence_keyword_score = float(
            self.gateway_cfg.get("high_confidence_keyword_score", 0.65)
        )
        self.high_confidence_cooldown_floor = self._clamp(
            float(self.gateway_cfg.get("high_confidence_cooldown_floor", 0.8))
        )
        self.recall_admission_semantic_score = self._clamp(
            float(self.gateway_cfg.get("recall_admission_semantic_score", self.high_confidence_semantic_score))
        )
        self.recall_admission_rerank_score = self._clamp(
            float(self.gateway_cfg.get("recall_admission_rerank_score", 0.65))
        )
        self.recall_policy = RecallPolicy(
            self.relevance_options,
            semantic_threshold=self.recall_admission_semantic_score,
            rerank_threshold=self.recall_admission_rerank_score,
        )
        self.query_planner_enabled = self._bool_config_value(
            self.gateway_cfg.get("query_planner_enabled"),
            False,
        )
        self.query_planner_model = str(
            self.gateway_cfg.get("query_planner_model") or self.upstream_default_model or ""
        ).strip()
        self.query_planner_min_chars = max(0, int(self.gateway_cfg.get("query_planner_min_chars", 16)))
        self.query_planner_max_queries = max(1, min(3, int(self.gateway_cfg.get("query_planner_max_queries", 3))))
        self.query_planner_max_tokens = max(128, int(self.gateway_cfg.get("query_planner_max_tokens", 360)))
        self.query_planner_score_bonus = self._clamp(
            float(self.gateway_cfg.get("query_planner_score_bonus", 0.04)),
            0.0,
            0.30,
        )
        self.memory_detail_recall_enabled = self._bool_config_value(
            self.gateway_cfg.get("memory_detail_recall_enabled"),
            False,
        )
        self.memory_detail_recall_max_ids = max(
            1,
            min(3, int(self.gateway_cfg.get("memory_detail_recall_max_ids", 3))),
        )
        self.memory_detail_recall_budget = max(
            200,
            int(self.gateway_cfg.get("memory_detail_recall_budget", 1200)),
        )
        self.edge_min_confidence = float(self.gateway_cfg.get("edge_min_confidence", 0.55))
        self.upstream_key_cooldown_seconds = max(
            0.0, float(self.gateway_cfg.get("upstream_key_cooldown_seconds", 300))
        )
        self.upstream_key_cooldowns: dict[tuple[str, str], float] = {}
        self.pending_tool_reasoning: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}

        self.http_client = http_client or httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        if self.http_client and not getattr(self.http_client, "is_closed", False):
            await self.http_client.aclose()

    async def health_payload(self) -> dict:
        stats = await self.bucket_mgr.get_stats()
        return {
            "status": "ok",
            "gateway": {
                "token_configured": bool(self.gateway_token),
                "upstream_ready": bool(self.upstreams) and all(
                    bool(upstream.get("base_url") and upstream.get("api_keys"))
                    for upstream in self.upstreams
                ),
                "upstream_base_url": self.upstream_base_url
                or (self.upstreams[0]["base_url"] if len(self.upstreams) == 1 else ""),
                "upstream_default_model": self.upstream_default_model,
                "upstream_models": self.upstream_models,
                "cooldown_hours": self.cooldown_hours,
                "skip_recent_rounds": self.skip_recent_rounds,
                "recent_context_cooldown_hours": self.recent_context_cooldown_hours,
                "recent_context_reentry_idle_hours": self.recent_context_reentry_idle_hours,
                "recent_context_budget": self.recent_budget,
                "recalled_memory_budget": self.recalled_budget,
                "related_memory_budget": self.related_memory_budget,
                "current_inner_state_interval_rounds": self.current_inner_state_interval_rounds,
                "direct_render_mode": self.direct_render_mode,
                "retrieval_mode": self.retrieval_mode,
                "reranker": {
                    "enabled": bool(getattr(self.reranker_engine, "enabled", False)),
                    "model": getattr(self.reranker_engine, "model", ""),
                    "base_url": getattr(self.reranker_engine, "base_url", ""),
                    "candidate_limit": getattr(self.reranker_engine, "candidate_limit", 0),
                },
                "upstreams": [
                    {
                        "name": upstream["name"],
                        "base_url": upstream["base_url"],
                        "default_model": upstream["default_model"],
                        "models": upstream["models"],
                        "prompt_cache": upstream.get("prompt_cache", ""),
                        "prompt_cache_retention": upstream.get("prompt_cache_retention", ""),
                        "key_count": len(upstream.get("api_keys", [])),
                        "ready": bool(upstream.get("base_url") and upstream.get("api_keys")),
                    }
                    for upstream in self.upstreams
                ],
            },
            "persona": {
                "enabled": bool(self.persona_engine.enabled),
                "profile_id": self.persona_engine.profile_id,
                "mode": self.persona_engine.mode,
                "model": self.persona_engine.model,
                "api_ready": bool(self.persona_engine.api_key),
            },
            "buckets": stats,
        }

    def _gateway_memory_config_payload(self) -> dict[str, Any]:
        return {
            "cooldown_hours": self.cooldown_hours,
            "skip_recent_rounds": self.skip_recent_rounds,
            "recent_context_cooldown_hours": self.recent_context_cooldown_hours,
            "recent_context_reentry_idle_hours": self.recent_context_reentry_idle_hours,
            "recent_context_budget": self.recent_budget,
            "recalled_memory_budget": self.recalled_budget,
            "related_memory_budget": self.related_memory_budget,
            "current_inner_state_interval_rounds": self.current_inner_state_interval_rounds,
            "direct_render_mode": self.direct_render_mode,
            "retrieval_mode": self.retrieval_mode,
            "portrait_memory_enabled": self.portrait_memory_enabled,
            "portrait_memory_budget": self.portrait_memory_budget,
            "portrait_memory_max_sources": self.portrait_memory_max_sources,
            "portrait_memory_include_anchors": self.portrait_memory_include_anchors,
            "query_planner_enabled": self.query_planner_enabled,
            "query_planner_model": self.query_planner_model,
            "query_planner_min_chars": self.query_planner_min_chars,
            "query_planner_max_queries": self.query_planner_max_queries,
            "query_planner_max_tokens": self.query_planner_max_tokens,
            "memory_detail_recall_enabled": self.memory_detail_recall_enabled,
            "memory_detail_recall_max_ids": self.memory_detail_recall_max_ids,
            "memory_detail_recall_budget": self.memory_detail_recall_budget,
        }

    def _memory_diffusion_config_payload(self) -> dict[str, Any]:
        options = self.diffusion_options
        return {
            "enabled": options.enabled,
            "max_hops": options.max_hops,
            "top_k": options.top_k,
            "min_activation": options.min_activation,
            "max_paths_per_hit": options.max_paths_per_hit,
            "chain_walk_enabled": options.chain_walk_enabled,
            "chain_max_hops": options.chain_max_hops,
            "chain_min_strength": options.chain_min_strength,
            "chain_min_confidence": options.chain_min_confidence,
            "chain_min_relation_priority": options.chain_min_relation_priority,
            "chain_max_frontier": options.chain_max_frontier,
        }

    def _persona_config_payload(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.persona_engine, "enabled", False)),
            "model": getattr(self.persona_engine, "model", ""),
            "base_url": getattr(self.persona_engine, "base_url", ""),
            "event_recording_enabled": bool(
                getattr(self.persona_engine, "event_recording_enabled", True)
            ),
            "api_ready": bool(getattr(self.persona_engine, "api_key", "")),
        }

    def _dream_config_payload(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.dream_engine, "enabled", self.dream_cfg.get("enabled", True))),
            "surface_enabled": bool(
                getattr(self.dream_engine, "surface_enabled", self.dream_cfg.get("surface_enabled", True))
            ),
            "inject_enabled": self.dream_inject_enabled,
            "retain_after_inject": self.dream_retain_after_inject,
        }

    def _apply_gateway_memory_config(self, payload: dict[str, Any]) -> list[str]:
        updated: list[str] = []
        if "cooldown_hours" in payload:
            self.cooldown_hours = max(0.0, float(payload["cooldown_hours"]))
            self.gateway_cfg["cooldown_hours"] = self.cooldown_hours
            updated.append("gateway.cooldown_hours")
        if "skip_recent_rounds" in payload:
            self.skip_recent_rounds = max(0, int(payload["skip_recent_rounds"]))
            self.gateway_cfg["skip_recent_rounds"] = self.skip_recent_rounds
            updated.append("gateway.skip_recent_rounds")
        if "recent_context_cooldown_hours" in payload:
            self.recent_context_cooldown_hours = max(0.0, float(payload["recent_context_cooldown_hours"]))
            self.gateway_cfg["recent_context_cooldown_hours"] = self.recent_context_cooldown_hours
            updated.append("gateway.recent_context_cooldown_hours")
        if "recent_context_reentry_idle_hours" in payload:
            self.recent_context_reentry_idle_hours = max(
                0.0,
                float(payload["recent_context_reentry_idle_hours"]),
            )
            self.gateway_cfg["recent_context_reentry_idle_hours"] = self.recent_context_reentry_idle_hours
            updated.append("gateway.recent_context_reentry_idle_hours")
        if "recent_context_budget" in payload:
            self.recent_budget = max(0, int(payload["recent_context_budget"]))
            self.gateway_cfg["recent_context_budget"] = self.recent_budget
            updated.append("gateway.recent_context_budget")
        if "recalled_memory_budget" in payload:
            self.recalled_budget = max(0, int(payload["recalled_memory_budget"]))
            self.gateway_cfg["recalled_memory_budget"] = self.recalled_budget
            updated.append("gateway.recalled_memory_budget")
        if "related_memory_budget" in payload:
            self.related_memory_budget = max(0, int(payload["related_memory_budget"]))
            self.gateway_cfg["related_memory_budget"] = self.related_memory_budget
            updated.append("gateway.related_memory_budget")
        if "current_inner_state_interval_rounds" in payload:
            self.current_inner_state_interval_rounds = max(
                0,
                int(payload["current_inner_state_interval_rounds"]),
            )
            self.gateway_cfg["current_inner_state_interval_rounds"] = self.current_inner_state_interval_rounds
            updated.append("gateway.current_inner_state_interval_rounds")
        if "direct_render_mode" in payload:
            self.direct_render_mode = self._normalize_direct_render_mode(payload["direct_render_mode"])
            self.gateway_cfg["direct_render_mode"] = self.direct_render_mode
            updated.append("gateway.direct_render_mode")
        if "retrieval_mode" in payload:
            self.retrieval_mode = self._normalize_retrieval_mode(payload["retrieval_mode"])
            self.gateway_cfg["retrieval_mode"] = self.retrieval_mode
            updated.append("gateway.retrieval_mode")
        if "portrait_memory_enabled" in payload:
            self.portrait_memory_enabled = self._bool_config_value(payload["portrait_memory_enabled"], False)
            self.gateway_cfg["portrait_memory_enabled"] = self.portrait_memory_enabled
            updated.append("gateway.portrait_memory_enabled")
        if "portrait_memory_budget" in payload:
            self.portrait_memory_budget = max(120, int(payload["portrait_memory_budget"]))
            self.gateway_cfg["portrait_memory_budget"] = self.portrait_memory_budget
            updated.append("gateway.portrait_memory_budget")
        if "portrait_memory_max_sources" in payload:
            self.portrait_memory_max_sources = max(1, min(20, int(payload["portrait_memory_max_sources"])))
            self.gateway_cfg["portrait_memory_max_sources"] = self.portrait_memory_max_sources
            updated.append("gateway.portrait_memory_max_sources")
        if "portrait_memory_include_anchors" in payload:
            self.portrait_memory_include_anchors = self._bool_config_value(
                payload["portrait_memory_include_anchors"],
                True,
            )
            self.gateway_cfg["portrait_memory_include_anchors"] = self.portrait_memory_include_anchors
            updated.append("gateway.portrait_memory_include_anchors")
        if "query_planner_enabled" in payload:
            self.query_planner_enabled = self._bool_config_value(payload["query_planner_enabled"], False)
            self.gateway_cfg["query_planner_enabled"] = self.query_planner_enabled
            updated.append("gateway.query_planner_enabled")
        if "query_planner_model" in payload:
            self.query_planner_model = str(payload["query_planner_model"] or "").strip()
            self.gateway_cfg["query_planner_model"] = self.query_planner_model
            updated.append("gateway.query_planner_model")
        if "query_planner_min_chars" in payload:
            self.query_planner_min_chars = max(0, int(payload["query_planner_min_chars"]))
            self.gateway_cfg["query_planner_min_chars"] = self.query_planner_min_chars
            updated.append("gateway.query_planner_min_chars")
        if "query_planner_max_queries" in payload:
            self.query_planner_max_queries = max(1, min(3, int(payload["query_planner_max_queries"])))
            self.gateway_cfg["query_planner_max_queries"] = self.query_planner_max_queries
            updated.append("gateway.query_planner_max_queries")
        if "query_planner_max_tokens" in payload:
            self.query_planner_max_tokens = max(128, int(payload["query_planner_max_tokens"]))
            self.gateway_cfg["query_planner_max_tokens"] = self.query_planner_max_tokens
            updated.append("gateway.query_planner_max_tokens")
        if "memory_detail_recall_enabled" in payload:
            self.memory_detail_recall_enabled = self._bool_config_value(
                payload["memory_detail_recall_enabled"],
                False,
            )
            self.gateway_cfg["memory_detail_recall_enabled"] = self.memory_detail_recall_enabled
            updated.append("gateway.memory_detail_recall_enabled")
        if "memory_detail_recall_max_ids" in payload:
            self.memory_detail_recall_max_ids = max(1, min(3, int(payload["memory_detail_recall_max_ids"])))
            self.gateway_cfg["memory_detail_recall_max_ids"] = self.memory_detail_recall_max_ids
            updated.append("gateway.memory_detail_recall_max_ids")
        if "memory_detail_recall_budget" in payload:
            self.memory_detail_recall_budget = max(200, int(payload["memory_detail_recall_budget"]))
            self.gateway_cfg["memory_detail_recall_budget"] = self.memory_detail_recall_budget
            updated.append("gateway.memory_detail_recall_budget")
        return updated

    def _apply_memory_diffusion_config(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload, dict):
            return []
        keys = (
            "enabled",
            "max_hops",
            "top_k",
            "min_activation",
            "max_paths_per_hit",
            "chain_walk_enabled",
            "chain_max_hops",
            "chain_min_strength",
            "chain_min_confidence",
            "chain_min_relation_priority",
            "chain_max_frontier",
        )
        diffusion_cfg = self.config.setdefault("memory_diffusion", {})
        requested = [key for key in keys if key in payload]
        for key in requested:
            diffusion_cfg[key] = payload[key]
        if not requested:
            return []
        self.diffusion_options = diffusion_options_from_config(self.config)
        normalized = self._memory_diffusion_config_payload()
        for key in requested:
            diffusion_cfg[key] = normalized[key]
        return [f"memory_diffusion.{key}" for key in requested]

    def _apply_persona_config(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload, dict):
            return []
        persona_cfg = self.config.setdefault("persona", {})
        updated: list[str] = []
        if "enabled" in payload:
            persona_cfg["enabled"] = bool(payload["enabled"])
            updated.append("persona.enabled")
        if "event_recording_enabled" in payload:
            persona_cfg["event_recording_enabled"] = bool(payload["event_recording_enabled"])
            updated.append("persona.event_recording_enabled")
        for key in ("model", "base_url"):
            if key in payload:
                persona_cfg[key] = str(payload[key] or "").strip()
                updated.append(f"persona.{key}")
        if "api_key" in payload and payload["api_key"]:
            persona_cfg["api_key"] = str(payload["api_key"])
            os.environ["OMBRE_PERSONA_API_KEY"] = persona_cfg["api_key"]
            updated.append("persona.api_key")
        if "base_url" in payload and persona_cfg.get("base_url"):
            os.environ["OMBRE_PERSONA_BASE_URL"] = persona_cfg["base_url"]
        if "model" in payload and persona_cfg.get("model"):
            os.environ["OMBRE_PERSONA_MODEL"] = persona_cfg["model"]
        if updated:
            self.persona_engine = PersonaStateEngine(self.config)
        return updated

    def _apply_dream_config(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload, dict):
            return []
        dream_cfg = self.config.setdefault("dream", {})
        updated: list[str] = []
        for key in (
            "enabled",
            "auto_enabled",
            "surface_enabled",
            "inject_enabled",
            "retain_after_inject",
            "model",
            "base_url",
            "temperature",
            "max_tokens",
            "daily_hour",
            "run_window_hours",
            "daily_probability",
            "min_material_count",
            "material_window_hours",
            "identity_anchor_id",
        ):
            if key in payload:
                dream_cfg[key] = payload[key]
                updated.append(f"dream.{key}")
        if updated:
            self.dream_cfg = dream_cfg
            self.dream_inject_enabled = bool(dream_cfg.get("inject_enabled", False))
            self.dream_retain_after_inject = bool(dream_cfg.get("retain_after_inject", False))
            self.dream_engine = DreamEngine(self.config)
        return updated

    async def handle_config(self, request: Request) -> JSONResponse:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        if request.method == "GET":
            return JSONResponse({
                "gateway": self._gateway_memory_config_payload(),
                "memory_diffusion": self._memory_diffusion_config_payload(),
                "persona": self._persona_config_payload(),
                "dream": self._dream_config_payload(),
            })

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid config"}, status_code=400)

        gateway_payload = body.get("gateway")
        diffusion_payload = body.get("memory_diffusion")
        persona_payload = body.get("persona")
        dream_payload = body.get("dream")
        if gateway_payload is None and diffusion_payload is None and persona_payload is None and dream_payload is None:
            gateway_payload = body
        if gateway_payload is not None and not isinstance(gateway_payload, dict):
            return JSONResponse({"error": "invalid gateway config"}, status_code=400)
        if diffusion_payload is not None and not isinstance(diffusion_payload, dict):
            return JSONResponse({"error": "invalid memory diffusion config"}, status_code=400)
        if persona_payload is not None and not isinstance(persona_payload, dict):
            return JSONResponse({"error": "invalid persona config"}, status_code=400)
        if dream_payload is not None and not isinstance(dream_payload, dict):
            return JSONResponse({"error": "invalid dream config"}, status_code=400)

        updated = []
        if gateway_payload is not None:
            updated.extend(self._apply_gateway_memory_config(gateway_payload))
        if diffusion_payload is not None:
            updated.extend(self._apply_memory_diffusion_config(diffusion_payload))
        if persona_payload is not None:
            updated.extend(self._apply_persona_config(persona_payload))
        if dream_payload is not None:
            updated.extend(self._apply_dream_config(dream_payload))
        return JSONResponse({
            "ok": True,
            "updated": updated,
            "gateway": self._gateway_memory_config_payload(),
            "memory_diffusion": self._memory_diffusion_config_payload(),
            "persona": self._persona_config_payload(),
            "dream": self._dream_config_payload(),
        })

    async def handle_health(self, request: Request) -> JSONResponse:
        try:
            return JSONResponse(await self.health_payload())
        except Exception as exc:
            logger.exception("Gateway health check failed: %s", exc)
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

    async def handle_chat(self, request: Request) -> Response:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        session_id = (request.headers.get("X-Ombre-Session-Id") or self.default_session_id).strip()

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Request body must be valid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}},
                status_code=400,
            )

        logger.info(
            "Gateway incoming chat | session=%s model=%s stream=%s messages=%s",
            session_id,
            payload.get("model") or self.upstream_default_model,
            payload.get("stream") is True,
            self._summarize_messages_for_debug(payload.get("messages")),
        )

        try:
            payload, marker_favorite = self._strip_favorite_memory_marker_from_payload(payload)
            include_favorite_memory = marker_favorite or self._truthy_header(
                request.headers.get("X-Ombre-Include-Favorite-Memory")
            )
            persona_user_message = self._extract_last_user_query(payload.get("messages", []))
            forward_payload, recalled_ids, injection_debug = await self.prepare_payload(
                payload,
                session_id,
                include_favorite_memory=include_favorite_memory,
                include_debug=True,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=503,
            )

        if forward_payload.get("stream") is True:
            try:
                return await self._stream_upstream(
                    forward_payload,
                    session_id,
                    recalled_ids,
                    persona_user_message,
                    injection_debug,
                )
            except RuntimeError as exc:
                return JSONResponse(
                    {"error": {"message": str(exc), "type": "server_error"}},
                    status_code=503,
                )

        upstream_response = await self._forward_upstream(forward_payload)
        if 200 <= upstream_response.status_code < 300:
            upstream_response, memory_detail_debug = await self._maybe_retry_with_memory_detail(
                forward_payload=forward_payload,
                upstream_response=upstream_response,
                injection_debug=injection_debug,
            )
            if memory_detail_debug and isinstance(injection_debug, dict):
                injection_debug["memory_detail_recall_debug"] = memory_detail_debug
            self._log_cache_usage_from_response(
                session_id,
                forward_payload["model"],
                upstream_response,
                route="/v1/chat/completions",
            )
            self._capture_reasoning_from_response(session_id, upstream_response)
            await self._record_successful_round(session_id, recalled_ids, injection_debug)
            await self._update_persona_after_response(
                session_id,
                persona_user_message,
                upstream_response,
                recalled_ids or [],
            )

        return self._proxy_response(upstream_response)

    async def handle_anthropic_messages(self, request: Request) -> Response:
        auth_result = self._authorize_anthropic_request(request)
        if auth_result is not None:
            return auth_result

        session_id = (request.headers.get("X-Ombre-Session-Id") or self.default_session_id).strip()

        try:
            payload = await request.json()
        except Exception:
            return self._anthropic_error("Request body must be valid JSON", status_code=400)

        if not isinstance(payload, dict):
            return self._anthropic_error("Request body must be a JSON object", status_code=400)

        try:
            openai_payload = self._anthropic_request_to_openai(payload)
        except ValueError as exc:
            return self._anthropic_error(str(exc), status_code=400)

        logger.info(
            "Gateway incoming Anthropic messages | session=%s model=%s messages=%s",
            session_id,
            openai_payload.get("model") or self.upstream_default_model,
            self._summarize_messages_for_debug(openai_payload.get("messages")),
        )

        try:
            openai_payload, marker_favorite = self._strip_favorite_memory_marker_from_payload(openai_payload)
            include_favorite_memory = marker_favorite or self._truthy_header(
                request.headers.get("X-Ombre-Include-Favorite-Memory")
            )
            persona_user_message = self._extract_last_user_query(openai_payload.get("messages", []))
            forward_payload, recalled_ids, injection_debug = await self.prepare_payload(
                openai_payload,
                session_id,
                include_favorite_memory=include_favorite_memory,
                include_debug=True,
            )
        except ValueError as exc:
            return self._anthropic_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return self._anthropic_error(str(exc), status_code=503, error_type="server_error")

        if forward_payload.get("stream") is True:
            return await self._stream_upstream_as_anthropic(
                forward_payload,
                session_id,
                recalled_ids,
                persona_user_message,
                injection_debug,
            )

        upstream_response = await self._forward_upstream(forward_payload)
        if 200 <= upstream_response.status_code < 300:
            upstream_response, memory_detail_debug = await self._maybe_retry_with_memory_detail(
                forward_payload=forward_payload,
                upstream_response=upstream_response,
                injection_debug=injection_debug,
            )
            if memory_detail_debug and isinstance(injection_debug, dict):
                injection_debug["memory_detail_recall_debug"] = memory_detail_debug
            self._log_cache_usage_from_response(
                session_id,
                forward_payload["model"],
                upstream_response,
                route="/v1/messages",
            )
            self._capture_reasoning_from_response(session_id, upstream_response)
            await self._record_successful_round(session_id, recalled_ids, injection_debug)
            await self._update_persona_after_response(
                session_id,
                persona_user_message,
                upstream_response,
                recalled_ids or [],
            )
            return self._openai_response_to_anthropic(upstream_response, forward_payload["model"])

        return self._proxy_anthropic_error_response(upstream_response)

    async def handle_models(self, request: Request) -> Response:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "ombre-gateway",
                    }
                    for model in self.upstream_models
                ],
            }
        )

    async def handle_injection_debug(self, request: Request) -> JSONResponse:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            limit = 20
        session_id = str(request.query_params.get("session_id", "") or "").strip()
        include_context = str(request.query_params.get("include_context", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
        }
        return JSONResponse(
            {
                "items": self.state_store.list_injection_debug(
                    session_id=session_id,
                    limit=limit,
                    include_context=include_context,
                )
            }
        )

    async def prepare_payload(
        self,
        payload: dict,
        session_id: str,
        *,
        include_favorite_memory: bool = False,
        include_debug: bool = False,
    ) -> tuple[dict, list[str] | None] | tuple[dict, list[str] | None, dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        model = payload.get("model") or self.upstream_default_model
        if not model:
            raise ValueError("model is required when gateway.upstream_default_model is empty")
        self._get_upstream_for_model(model)

        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        current_user_query = self._extract_current_turn_user_query(messages)
        is_new_user_turn = bool(current_user_query)

        persona_block = ""
        core_memory = ""
        portrait_memory = ""
        portrait_memory_debug: dict[str, Any] = self._portrait_memory_debug_base()
        recent_context = ""
        recent_context_reason = ""
        recalled_moments: list[dict] = []
        moment_candidates: list[dict] = []
        suppressed_moments: list[dict] = []
        suppressed_buckets: list[dict] = []
        all_moments: list[dict] = []
        grouped_moments: dict[str, list[dict]] = {}
        moment_edges: list[dict] = []
        recalled_memory = ""
        relationship_weather = ""
        favorite_memory = ""
        favorite_ids: list[str] = []
        related_memory = ""
        targeted_memory_detail = ""
        targeted_memory_detail_debug: dict[str, Any] = self._targeted_memory_detail_debug_base()
        memory_detail_recall_instruction = ""
        dream_context = ""
        dream_context_status: dict[str, Any] = {"status": "skipped", "reason": "not_current_user_turn"}
        diffused_moment_debug: list[dict[str, Any]] = []
        context_mode = ""
        persona_state: dict[str, Any] | None = None
        injected_ids: list[str] | None = None
        query_planner_debug: dict[str, Any] = self._query_planner_debug_base(current_user_query)

        if is_new_user_turn:
            skip_broad_dynamic_recall = self._query_should_skip_broad_for_targeted_memory_detail(
                current_user_query,
                session_id,
            )
            if self.persona_engine.enabled and self._should_inject_interval(
                session_id,
                self.current_inner_state_interval_rounds,
            ):
                persona_state = await self.persona_engine.build_pre_reply_guidance(
                    session_id, current_user_query
                )
                persona_block = self.persona_engine.format_state_block(persona_state)
            if self.persona_engine.enabled and persona_state is None:
                persona_state = self._get_persona_state_for_context_mode(session_id)
            context_mode = self._classify_context_mode(current_user_query, persona_state)
            if self._should_inject_interval(session_id, self.core_memory_interval_rounds):
                core_memory = await self._build_core_memory_block(all_buckets)
            portrait_memory, portrait_memory_debug = self._build_portrait_memory_block(all_buckets)
            if self.recalled_budget > 0 or self.related_memory_budget > 0:
                if skip_broad_dynamic_recall:
                    logger.info(
                        "Gateway broad dynamic recall skipped | session=%s reason=targeted_memory_detail_query",
                        session_id,
                    )
                    suppressed_moments = []
                    suppressed_buckets = []
                elif self.retrieval_mode == "bucket":
                    selected_buckets, suppressed_buckets, query_planner_debug = await self._select_dynamic_buckets(
                        current_user_query,
                        session_id,
                        all_buckets,
                        search_query=recall_search_query(current_user_query, self.relevance_options),
                        include_query_planner_debug=True,
                    )
                    for bucket in selected_buckets:
                        bucket_id = str(bucket.get("id") or "")
                        if not bucket_id:
                            continue
                        bucket_moments = self._direct_moments_for_bucket(bucket, current_user_query)
                        moment = self._representative_moment(bucket_moments)
                        if not moment:
                            continue
                        grouped_moments[bucket_id] = bucket_moments
                        recalled_moments.append(moment)
                    moment_candidates = list(recalled_moments)
                    suppressed_moments = []
                else:
                    all_moments, grouped_moments, moment_edges = self._refresh_moment_graph(all_buckets)
                    (
                        recalled_moments,
                        moment_candidates,
                        suppressed_moments,
                        suppressed_buckets,
                        query_planner_debug,
                    ) = await self._select_dynamic_moments(
                        current_user_query,
                        session_id,
                        all_buckets,
                        grouped_moments,
                        include_query_planner_debug=True,
                    )
            else:
                suppressed_moments = []
                suppressed_buckets = []
            recalled_memory = await self._format_recalled_moments(
                recalled_moments,
                grouped_moments,
                all_buckets,
                self.recalled_budget,
                current_user_query,
            )
            if self._should_inject_interval(session_id, self.relationship_weather_interval_rounds):
                relationship_weather = await self._build_relationship_weather_block(all_buckets)
            if (
                include_favorite_memory
                or self._query_requests_favorite_memory(current_user_query)
                or self._should_inject_interval(session_id, self.favorite_memory_interval_rounds)
            ):
                favorite_memory, favorite_ids = await self._build_favorite_memory_block(all_buckets, session_id)
            if self.retrieval_mode == "graph":
                related_memory, diffused_moment_debug = self._build_moment_diffused_memory_with_debug(
                    recalled_moments,
                    moment_candidates,
                    all_moments,
                    moment_edges,
                    current_user_query,
                    context_mode=context_mode,
                )
            else:
                related_memory = ""
            current_direct_bucket_ids = [
                str(moment.get("bucket_id") or "")
                for moment in recalled_moments
                if moment.get("bucket_id")
            ]
            current_direct_moment_ids = [
                str(moment.get("moment_id") or "")
                for moment in recalled_moments
                if moment.get("moment_id")
            ]
            current_diffused_bucket_ids = self._extract_bucket_ids_from_context(related_memory)
            current_diffused_moment_ids = self._extract_moment_ids_from_context(related_memory)
            current_shown_bucket_ids = list(
                dict.fromkeys(
                    current_direct_bucket_ids
                    + current_diffused_bucket_ids
                    + favorite_ids
                )
            )
            current_shown_moment_ids = list(
                dict.fromkeys(current_direct_moment_ids + current_diffused_moment_ids)
            )
            targeted_memory_detail, targeted_memory_detail_debug = self._build_targeted_memory_detail(
                all_buckets,
                session_id=session_id,
                query=current_user_query,
                current_shown_bucket_ids=current_shown_bucket_ids,
                current_shown_moment_ids=current_shown_moment_ids,
                current_direct_bucket_ids=current_direct_bucket_ids,
                current_direct_moment_ids=current_direct_moment_ids,
                current_diffused_bucket_ids=current_diffused_bucket_ids,
                current_diffused_moment_ids=current_diffused_moment_ids,
                recalled_memory=recalled_memory,
            )
            if self.memory_detail_recall_enabled and (
                recalled_memory.strip()
                or related_memory.strip()
                or favorite_memory.strip()
                or targeted_memory_detail.strip()
            ):
                memory_detail_recall_instruction = (
                    "Internal memory detail request: if a shown memory summary is clearly relevant "
                    "but lacks needed detail, you may start your draft with exactly "
                    f"`[memory_detail ids=\"bucket_id_1,bucket_id_2\"]`. Use only bucket_id values "
                    f"shown in this turn, at most {self.memory_detail_recall_max_ids}. "
                    "Do not mention this line in the final answer."
                )
            reliable_dynamic_context = bool(recalled_memory.strip() or related_memory.strip())
            if self._should_inject_recent_context(
                session_id,
                current_user_query,
                has_reliable_dynamic_context=reliable_dynamic_context,
            ):
                explicit_recent_query = self._query_requests_recent_context(current_user_query)
                recent_context = await self._build_recent_context_block(
                    all_buckets,
                    current_user_query,
                    allow_vague=explicit_recent_query,
                )
                if recent_context.strip():
                    recent_context_reason = self._recent_context_reason(
                        session_id,
                        current_user_query,
                        has_reliable_dynamic_context=reliable_dynamic_context,
                    )
            dream_context, dream_context_status = await self._build_dream_context_block(
                current_user_query,
                session_id,
            )
            injected_ids = list(
                dict.fromkeys(
                    [
                        str(moment.get("bucket_id") or "")
                        for moment in recalled_moments
                        if moment.get("bucket_id")
                    ]
                    + favorite_ids
                    + [
                        str(bucket_id)
                        for bucket_id in targeted_memory_detail_debug.get("accepted_ids", []) or []
                        if str(bucket_id or "").strip()
                    ]
                )
            )
        else:
            logger.info(
                "Gateway dynamic context skipped | session=%s reason=not_current_user_turn",
                session_id,
            )

        stable_context, dynamic_context = self._build_injected_context_messages(
            persona_block=persona_block,
            core_memory=core_memory,
            portrait_memory=portrait_memory,
            recent_context=recent_context,
            recalled_memory=recalled_memory,
            relationship_weather=relationship_weather,
            favorite_memory=favorite_memory,
            related_memory=related_memory,
            targeted_memory_detail=targeted_memory_detail,
            dream_context=dream_context,
            memory_detail_recall_instruction=memory_detail_recall_instruction,
            context_mode=context_mode,
        )

        forward_payload = deepcopy(payload)
        forward_payload["model"] = model
        self._restore_cached_reasoning_content(session_id, forward_payload.get("messages"))
        forward_payload["messages"] = self._inject_context_messages(
            forward_payload["messages"],
            stable_context,
            dynamic_context,
        )
        self._apply_prompt_cache_hints(forward_payload, session_id)
        forward_payload["stream"] = payload.get("stream") is True
        if include_debug:
            return forward_payload, injected_ids, self._build_injection_debug_payload(
                model=model,
                query=current_user_query,
                stable_context=stable_context,
                dynamic_context=dynamic_context,
                all_buckets=all_buckets,
                portrait_memory=portrait_memory,
                portrait_memory_debug=portrait_memory_debug,
                recalled_moments=recalled_moments,
                recalled_memory=recalled_memory,
                related_memory=related_memory,
                targeted_memory_detail=targeted_memory_detail,
                targeted_memory_detail_debug=targeted_memory_detail_debug,
                dream_context=dream_context,
                dream_context_status=dream_context_status,
                recent_context=recent_context,
                recent_context_reason=recent_context_reason,
                favorite_ids=favorite_ids,
                context_mode=context_mode,
                diffused_moment_debug=diffused_moment_debug,
                suppressed_moments=suppressed_moments,
                suppressed_buckets=suppressed_buckets,
                query_planner_debug=query_planner_debug,
            )
        return forward_payload, injected_ids

    def _apply_prompt_cache_hints(self, payload: dict[str, Any], session_id: str) -> None:
        model = str(payload.get("model") or "").strip()
        route = self._resolve_upstream_for_model(model)
        upstream = route["upstream"]
        strategy = str(upstream.get("prompt_cache") or "").strip().lower()
        if strategy != "openai":
            return

        payload.setdefault("prompt_cache_key", session_id)
        retention = str(upstream.get("prompt_cache_retention") or "").strip()
        if retention:
            payload.setdefault("prompt_cache_retention", retention)

    def _authorize(self, auth_header: str) -> JSONResponse | None:
        if not self.gateway_token:
            return JSONResponse(
                {"error": {"message": "Gateway token is not configured", "type": "server_error"}},
                status_code=503,
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                {"error": {"message": "Authorization: Bearer token is required", "type": "authentication_error"}},
                status_code=401,
            )

        if not secrets.compare_digest(token, self.gateway_token):
            return JSONResponse(
                {"error": {"message": "Invalid gateway token", "type": "authentication_error"}},
                status_code=401,
            )
        return None

    def _authorize_anthropic_request(self, request: Request) -> JSONResponse | None:
        if not self.gateway_token:
            return self._anthropic_error(
                "Gateway token is not configured",
                status_code=503,
                error_type="server_error",
            )

        auth_header = request.headers.get("Authorization", "")
        scheme, _, bearer_token = auth_header.partition(" ")
        api_key = (request.headers.get("x-api-key") or "").strip()
        token = bearer_token.strip() if scheme.lower() == "bearer" else api_key
        if not token:
            return self._anthropic_error(
                "Authorization: Bearer token or x-api-key is required",
                status_code=401,
                error_type="authentication_error",
            )

        if not secrets.compare_digest(token, self.gateway_token):
            return self._anthropic_error(
                "Invalid gateway token",
                status_code=401,
                error_type="authentication_error",
            )

        return None

    async def _forward_upstream(self, payload: dict) -> httpx.Response:
        model = str(payload.get("model") or "").strip()
        route = self._resolve_upstream_for_model(model)
        upstream = route["upstream"]
        upstream_payload = self._payload_for_upstream_model(payload, route["upstream_model"])
        url = f"{upstream['base_url']}/chat/completions"
        key_entries = self._available_upstream_api_keys(upstream)
        last_error: Exception | None = None
        last_response: httpx.Response | None = None

        for attempt, key_entry in enumerate(key_entries, start=1):
            started_at = time.perf_counter()
            try:
                response = await self.http_client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {key_entry['value']}",
                        "Content-Type": "application/json",
                    },
                    json=upstream_payload,
                )
            except httpx.RequestError as exc:
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                last_error = exc
                self._cool_down_upstream_key(upstream, key_entry)
                logger.warning(
                    "Gateway upstream request failed | upstream=%s key=%s model=%s upstream_model=%s "
                    "attempt=%s/%s latency_ms=%s error=%s",
                    upstream["name"],
                    key_entry["label"],
                    model,
                    route["upstream_model"],
                    attempt,
                    len(key_entries),
                    latency_ms,
                    exc,
                )
                continue

            latency_ms = int((time.perf_counter() - started_at) * 1000)
            last_response = response
            logger.info(
                "Gateway upstream response | upstream=%s key=%s model=%s upstream_model=%s "
                "status=%s attempt=%s/%s latency_ms=%s",
                upstream["name"],
                key_entry["label"],
                model,
                route["upstream_model"],
                response.status_code,
                attempt,
                len(key_entries),
                latency_ms,
            )
            if 200 <= response.status_code < 300:
                self._clear_upstream_key_cooldown(upstream, key_entry)
                return response
            if not self._should_retry_upstream_status(response.status_code):
                return response
            self._cool_down_upstream_key(upstream, key_entry)
            if attempt < len(key_entries):
                continue
            return response

        if last_response is not None:
            return last_response
        return self._upstream_request_error_response(upstream, model, last_error)

    async def _open_upstream_stream(
        self,
        route: dict[str, Any],
        payload: dict,
    ) -> httpx.Response:
        upstream = route["upstream"]
        model = route["public_model"]
        upstream_payload = self._payload_for_upstream_model(payload, route["upstream_model"])
        url = f"{upstream['base_url']}/chat/completions"
        key_entries = self._available_upstream_api_keys(upstream)
        last_error: Exception | None = None
        last_response: httpx.Response | None = None

        for attempt, key_entry in enumerate(key_entries, start=1):
            request = self.http_client.build_request(
                "POST",
                url,
                headers={
                    "Authorization": f"Bearer {key_entry['value']}",
                    "Content-Type": "application/json",
                },
                json=upstream_payload,
            )
            started_at = time.perf_counter()
            try:
                upstream_response = await self.http_client.send(request, stream=True)
            except httpx.RequestError as exc:
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                last_error = exc
                self._cool_down_upstream_key(upstream, key_entry)
                logger.warning(
                    "Gateway upstream stream failed | upstream=%s key=%s model=%s upstream_model=%s "
                    "attempt=%s/%s latency_ms=%s error=%s",
                    upstream["name"],
                    key_entry["label"],
                    model,
                    route["upstream_model"],
                    attempt,
                    len(key_entries),
                    latency_ms,
                    exc,
                )
                continue

            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Gateway upstream response | upstream=%s key=%s model=%s upstream_model=%s "
                "status=%s attempt=%s/%s latency_ms=%s",
                upstream["name"],
                key_entry["label"],
                model,
                route["upstream_model"],
                upstream_response.status_code,
                attempt,
                len(key_entries),
                latency_ms,
            )
            if 200 <= upstream_response.status_code < 300:
                self._clear_upstream_key_cooldown(upstream, key_entry)
                return upstream_response

            body = await upstream_response.aread()
            await upstream_response.aclose()
            last_response = httpx.Response(
                status_code=upstream_response.status_code,
                content=body,
                headers=upstream_response.headers,
            )
            if not self._should_retry_upstream_status(upstream_response.status_code):
                return last_response
            self._cool_down_upstream_key(upstream, key_entry)
            if attempt < len(key_entries):
                continue
            return last_response

        if last_response is not None:
            return last_response
        return self._upstream_request_error_response(upstream, model, last_error)

    async def _stream_upstream(
        self,
        payload: dict,
        session_id: str,
        recalled_ids: list[str] | None,
        user_message: str,
        injection_debug: dict[str, Any] | None = None,
    ) -> Response:
        model = str(payload.get("model") or "").strip()
        route = self._resolve_upstream_for_model(model)
        upstream_response = await self._open_upstream_stream(route, payload)
        content_type = upstream_response.headers.get("content-type", "text/event-stream")

        if not 200 <= upstream_response.status_code < 300:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=body,
                status_code=upstream_response.status_code,
                media_type=content_type,
            )

        async def stream_body():
            finalized = False
            stream_state = self._new_stream_capture_state()

            async def finalize_once() -> None:
                nonlocal finalized
                if finalized:
                    return
                finalized = True
                await self._finalize_stream_turn(
                    session_id=session_id,
                    model=model,
                    route="/v1/chat/completions",
                    stream_state=stream_state,
                    recalled_ids=recalled_ids,
                    user_message=user_message,
                    injection_debug=injection_debug,
                )

            try:
                async for chunk in upstream_response.aiter_bytes():
                    if chunk:
                        self._consume_stream_capture_chunk(stream_state, chunk)
                        if stream_state.get("seen_done"):
                            await finalize_once()
                        yield chunk
                self._consume_stream_capture_chunk(stream_state, b"", final=True)
                await finalize_once()
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _record_successful_round(
        self,
        session_id: str,
        recalled_ids: list[str] | None,
        injection_debug: dict[str, Any] | None = None,
    ) -> None:
        if recalled_ids is None:
            logger.info(
                "Gateway round bookkeeping skipped | session=%s reason=not_current_user_turn",
                session_id,
            )
            return
        round_id = self.state_store.record_success(session_id, recalled_ids)
        if injection_debug and injection_debug.get("recent_context_injected"):
            try:
                self.state_store.record_recent_context_injection(session_id, round_id)
            except Exception as exc:
                logger.warning(
                    "Gateway recent context cooldown record failed | session=%s round=%s error=%s",
                    session_id,
                    round_id,
                    exc,
                )
        if injection_debug is not None:
            try:
                self.state_store.record_injection_debug(session_id, round_id, injection_debug)
            except Exception as exc:
                logger.warning(
                    "Gateway injection debug record failed | session=%s round=%s error=%s",
                    session_id,
                    round_id,
                    exc,
                )
        for bucket_id in recalled_ids:
            await self.bucket_mgr.touch(bucket_id)
        logger.info(
            "Gateway round completed | session=%s round=%s recalled=%s",
            session_id,
            round_id,
            recalled_ids,
        )

    async def _maybe_retry_with_memory_detail(
        self,
        *,
        forward_payload: dict[str, Any],
        upstream_response: httpx.Response,
        injection_debug: dict[str, Any] | None,
    ) -> tuple[httpx.Response, dict[str, Any] | None]:
        try:
            body = upstream_response.json()
        except ValueError:
            return upstream_response, None

        assistant_message = self._extract_assistant_message_from_response_body(body)
        if not isinstance(assistant_message, dict):
            return upstream_response, None
        text = self._coerce_message_text(assistant_message.get("content"))
        requested_ids, stripped_text = self._parse_memory_detail_request(text)
        if not requested_ids:
            return upstream_response, None

        allowed_ids = []
        if isinstance(injection_debug, dict):
            allowed_ids = [
                str(bucket_id)
                for bucket_id in injection_debug.get("injected_bucket_ids", []) or []
                if str(bucket_id or "").strip()
            ]
        debug = self._memory_detail_recall_debug_base(allowed_ids)
        debug["triggered"] = True
        debug["requested_ids"] = requested_ids

        stripped_response = self._response_with_assistant_content(upstream_response, body, stripped_text)
        if not self.memory_detail_recall_enabled:
            debug["skip_reason"] = "disabled"
            return stripped_response, debug

        allowed_set = set(allowed_ids)
        accepted_ids = []
        rejected_ids = []
        for bucket_id in requested_ids:
            if bucket_id in accepted_ids:
                continue
            if bucket_id not in allowed_set:
                rejected_ids.append(bucket_id)
                continue
            if len(accepted_ids) >= self.memory_detail_recall_max_ids:
                rejected_ids.append(bucket_id)
                continue
            accepted_ids.append(bucket_id)

        debug["accepted_ids"] = accepted_ids
        debug["rejected_ids"] = rejected_ids
        if not accepted_ids:
            debug["skip_reason"] = "no_allowed_ids"
            return stripped_response, debug

        detail_context, missing_ids = await self._build_memory_detail_recall_context(accepted_ids)
        debug["missing_ids"] = missing_ids
        if not detail_context.strip():
            debug["skip_reason"] = "empty_detail_context"
            return stripped_response, debug

        retry_payload = deepcopy(forward_payload)
        retry_payload["stream"] = False
        retry_payload["messages"] = self._insert_memory_detail_context(
            retry_payload.get("messages", []),
            detail_context,
        )
        debug["detail_tokens"] = count_tokens_approx(detail_context)
        retry_response = await self._forward_upstream(retry_payload)
        debug["retry_status_code"] = retry_response.status_code
        if 200 <= retry_response.status_code < 300:
            debug["retried"] = True
            return retry_response, debug

        debug["skip_reason"] = "retry_failed"
        return stripped_response, debug

    def _memory_detail_recall_debug_base(self, allowed_ids: list[str] | None = None) -> dict[str, Any]:
        return {
            "enabled": bool(self.memory_detail_recall_enabled),
            "triggered": False,
            "retried": False,
            "skip_reason": "",
            "allowed_ids": list(allowed_ids or []),
            "requested_ids": [],
            "accepted_ids": [],
            "rejected_ids": [],
            "missing_ids": [],
            "detail_tokens": 0,
            "retry_status_code": None,
        }

    @staticmethod
    def _targeted_memory_detail_debug_base() -> dict[str, Any]:
        return {
            "triggered": False,
            "skip_reason": "",
            "requested_bucket_ids": [],
            "requested_moment_ids": [],
            "accepted_ids": [],
            "accepted_moment_ids": [],
            "missing_ids": [],
            "source": "",
            "detail_tokens": 0,
        }

    @staticmethod
    def _extract_explicit_bucket_ids_from_text(text: str) -> list[str]:
        ids = []
        seen = set()
        for match in EXPLICIT_BUCKET_ID_RE.finditer(str(text or "")):
            bucket_id = (match.group("bracket") or match.group("plain") or "").strip()
            if not bucket_id or bucket_id in seen:
                continue
            seen.add(bucket_id)
            ids.append(bucket_id)
        return ids

    @staticmethod
    def _extract_explicit_moment_ids_from_text(text: str) -> list[str]:
        ids = []
        seen = set()
        for match in EXPLICIT_MOMENT_ID_RE.finditer(str(text or "")):
            moment_id = (match.group("bracket") or match.group("plain") or "").strip()
            if not moment_id or moment_id in seen:
                continue
            seen.add(moment_id)
            ids.append(moment_id)
        return ids

    def _query_requests_targeted_memory_detail(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        if self._extract_explicit_bucket_ids_from_text(text) or self._extract_explicit_moment_ids_from_text(text):
            return True
        detail_terms = (
            "细节",
            "详细",
            "具体",
            "原文",
            "完整",
            "整条",
            "整桶",
            "当时怎么说",
            "当时说了什么",
            "具体怎么说",
            "怎么写的",
            "detail",
            "details",
        )
        reflection_terms = (
            "由此确认",
            "因此确认",
            "确认了什么",
            "确认什么",
            "由此明白",
            "明白了什么",
            "由此认为",
            "因此认为",
            "为什么喜欢",
            "喜欢这次",
            "喜欢它的原因",
            "喜欢的原因",
            "favorite reason",
        )
        deictic_terms = (
            "这次",
            "这条",
            "这个",
            "那次",
            "那条",
            "那个",
            "当时",
            "由此",
            "因此",
            "它",
            "刚才",
            "上面",
            "上一条",
            "这个片段",
            "这个记忆",
        )
        if self._text_has_any(text, reflection_terms):
            return True
        return self._text_has_any(text, deictic_terms) and self._text_has_any(text, detail_terms)

    def _query_should_skip_broad_for_targeted_memory_detail(self, query: str, session_id: str) -> bool:
        if not self._query_requests_targeted_memory_detail(query):
            return False
        if self._extract_explicit_bucket_ids_from_text(query) or self._extract_explicit_moment_ids_from_text(query):
            return True
        if self._query_has_concrete_targeted_detail_anchor(query):
            return False
        recent_bucket_ids, recent_moment_ids = self._recent_memory_reference_ids(session_id)
        if recent_bucket_ids or recent_moment_ids:
            return True
        intent = self._targeted_memory_detail_intent(query)
        return bool(intent.get("reflection") or intent.get("favorite_reason"))

    @staticmethod
    def _query_has_concrete_targeted_detail_anchor(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        for pattern in (
            EXPLICIT_BUCKET_ID_RE,
            EXPLICIT_MOMENT_ID_RE,
        ):
            text = pattern.sub("", text)
        noise_terms = (
            "为什么喜欢",
            "喜欢它的原因",
            "喜欢的原因",
            "喜欢这次",
            "由此确认",
            "因此确认",
            "确认了什么",
            "确认什么",
            "由此明白",
            "明白了什么",
            "由此认为",
            "因此认为",
            "当时怎么说",
            "当时说了什么",
            "具体怎么说",
            "怎么写的",
            "怎么说",
            "怎么写",
            "细节",
            "详细",
            "具体",
            "原文",
            "完整",
            "整条",
            "整桶",
            "这次",
            "这条",
            "这个片段",
            "这个记忆",
            "这个",
            "那次",
            "那条",
            "那个",
            "当时",
            "由此",
            "因此",
            "刚才",
            "上面",
            "上一条",
            "什么",
            "为什么",
            "你",
            "我",
            "小雨",
            "haven",
            "吗",
            "呢",
            "了",
            "的",
        )
        for term in noise_terms:
            text = text.replace(term, "")
        compact = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
        return len(compact) >= 3

    def _recent_memory_reference_ids(self, session_id: str) -> tuple[list[str], list[str]]:
        items = self.state_store.list_injection_debug(
            session_id=session_id,
            limit=1,
            include_context=False,
        )
        if not items:
            return [], []
        payload = items[0].get("payload") if isinstance(items[0], dict) else {}
        if not isinstance(payload, dict):
            return [], []
        moment_ids = list(
            dict.fromkeys(
                [
                    str(item)
                    for item in (
                        (payload.get("recalled_moment_ids") or [])
                        + (payload.get("diffused_moment_ids") or [])
                    )
                    if str(item or "").strip()
                ]
            )
        )
        bucket_ids = list(
            dict.fromkeys(
                [
                    str(item)
                    for item in payload.get("injected_bucket_ids", []) or []
                    if str(item or "").strip()
                ]
            )
        )
        return bucket_ids, moment_ids

    def _targeted_memory_detail_intent(self, query: str) -> dict[str, bool]:
        text = str(query or "").strip().lower()
        wants_reflection = self._text_has_any(
            text,
            (
                "由此确认",
                "因此确认",
                "确认",
                "明白",
                "认为",
                "理解",
                "反思",
                "reflection",
            ),
        )
        wants_favorite = self._text_has_any(
            text,
            (
                "为什么喜欢",
                "喜欢这次",
                "喜欢它",
                "喜欢的原因",
                "喜欢它的原因",
                "favorite",
            ),
        )
        wants_raw = self._text_has_any(
            text,
            (
                "细节",
                "详细",
                "具体",
                "原文",
                "完整",
                "整条",
                "整桶",
                "怎么说",
                "怎么写",
                "detail",
            ),
        )
        return {
            "reflection": wants_reflection,
            "favorite_reason": wants_favorite,
            "raw": wants_raw or not (wants_reflection or wants_favorite),
        }

    def _build_targeted_memory_detail(
        self,
        all_buckets: list[dict],
        *,
        session_id: str,
        query: str,
        current_shown_bucket_ids: list[str],
        current_shown_moment_ids: list[str],
        current_direct_bucket_ids: list[str],
        current_direct_moment_ids: list[str],
        current_diffused_bucket_ids: list[str],
        current_diffused_moment_ids: list[str],
        recalled_memory: str,
    ) -> tuple[str, dict[str, Any]]:
        debug = self._targeted_memory_detail_debug_base()
        if not self._query_requests_targeted_memory_detail(query):
            debug["skip_reason"] = "not_targeted_detail_query"
            return "", debug
        debug["triggered"] = True

        explicit_bucket_ids = self._extract_explicit_bucket_ids_from_text(query)
        explicit_moment_ids = self._extract_explicit_moment_ids_from_text(query)
        recent_bucket_ids, recent_moment_ids = self._recent_memory_reference_ids(session_id)
        intent = self._targeted_memory_detail_intent(query)

        max_items = max(1, int(self.memory_detail_recall_max_ids or 1))
        if explicit_bucket_ids or explicit_moment_ids:
            bucket_ids = explicit_bucket_ids
            moment_ids = explicit_moment_ids
            debug["source"] = "explicit_id"
        elif current_direct_bucket_ids or current_direct_moment_ids:
            if not self._current_direct_hit_needs_targeted_detail(recalled_memory, intent):
                debug["source"] = "current_direct_id"
                debug["skip_reason"] = "direct_hit_already_rendered"
                debug["requested_bucket_ids"] = list(dict.fromkeys(current_direct_bucket_ids))
                debug["requested_moment_ids"] = list(dict.fromkeys(current_direct_moment_ids))
                return "", debug
            bucket_ids = current_direct_bucket_ids
            moment_ids = current_direct_moment_ids
            debug["source"] = "current_direct_id"
        elif current_diffused_bucket_ids or current_diffused_moment_ids:
            bucket_ids = current_diffused_bucket_ids
            moment_ids = current_diffused_moment_ids
            debug["source"] = "current_diffused_id"
        elif current_shown_bucket_ids or current_shown_moment_ids:
            bucket_ids = current_shown_bucket_ids
            moment_ids = current_shown_moment_ids
            debug["source"] = "current_injected_id"
        else:
            bucket_ids = recent_bucket_ids
            moment_ids = recent_moment_ids
            debug["source"] = "previous_injected_id"

        bucket_ids = [item for item in dict.fromkeys(str(item) for item in bucket_ids) if item][:max_items]
        moment_ids = [item for item in dict.fromkeys(str(item) for item in moment_ids) if item][:max_items]
        debug["requested_bucket_ids"] = bucket_ids
        debug["requested_moment_ids"] = moment_ids
        if not bucket_ids and not moment_ids:
            debug["skip_reason"] = "no_shown_or_explicit_ids"
            return "", debug

        bucket_map = {
            str(bucket.get("id") or ""): bucket
            for bucket in all_buckets
            if isinstance(bucket, dict) and bucket.get("id")
        }
        moment_map: dict[str, dict[str, Any]] = {}
        moments_by_bucket: dict[str, list[dict[str, Any]]] = {}
        for bucket in bucket_map.values():
            try:
                moments = parse_bucket_moments(bucket, self.relevance_options)
            except Exception as exc:
                logger.warning("Gateway targeted detail parse failed for %s: %s", bucket.get("id"), exc)
                moments = []
            bucket_id = str(bucket.get("id") or "")
            moments_by_bucket[bucket_id] = moments
            for moment in moments:
                moment_id = str(moment.get("moment_id") or "")
                if moment_id:
                    moment_map[moment_id] = moment

        accepted_bucket_ids: list[str] = []
        accepted_moment_ids: list[str] = []
        missing_ids: list[str] = []
        for moment_id in moment_ids:
            moment = moment_map.get(moment_id)
            if not moment:
                missing_ids.append(moment_id)
                continue
            accepted_moment_ids.append(moment_id)
            bucket_id = str(moment.get("bucket_id") or "")
            if bucket_id and bucket_id not in accepted_bucket_ids:
                accepted_bucket_ids.append(bucket_id)
        for bucket_id in bucket_ids:
            if bucket_id not in bucket_map:
                missing_ids.append(bucket_id)
                continue
            if bucket_id not in accepted_bucket_ids:
                accepted_bucket_ids.append(bucket_id)
        accepted_bucket_ids = accepted_bucket_ids[:max_items]
        debug["accepted_ids"] = accepted_bucket_ids
        debug["accepted_moment_ids"] = accepted_moment_ids
        debug["missing_ids"] = missing_ids
        if not accepted_bucket_ids:
            debug["skip_reason"] = "no_allowed_ids"
            return "", debug

        per_bucket_budget = max(120, self.memory_detail_recall_budget // max(1, len(accepted_bucket_ids)))
        reference_context = self._recent_memory_reference_context(
            session_id,
            bucket_ids=accepted_bucket_ids,
            moment_ids=accepted_moment_ids,
        )
        blocks = [
            "Targeted private memory detail for this turn. Fetched only by bucket_id/moment_id already shown to or provided by the user. Use quietly; do not mention lookup.",
            "reflection/favorite_reason are Haven-side understanding, not Xiaoyu profile facts.",
        ]
        if reference_context:
            blocks.append("Reference summary/path/context already shown:\n" + reference_context)
        for bucket_id in accepted_bucket_ids:
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                continue
            selected_moment_ids = [
                moment_id
                for moment_id in accepted_moment_ids
                if str(moment_map.get(moment_id, {}).get("bucket_id") or "") == bucket_id
            ]
            block = self._format_targeted_bucket_detail(
                bucket,
                moments_by_bucket.get(bucket_id, []),
                selected_moment_ids=selected_moment_ids,
                intent=intent,
            )
            if block.strip():
                blocks.append(self._trim_text(block, per_bucket_budget))
        detail = "\n\n".join(part for part in blocks if part.strip())
        debug["detail_tokens"] = count_tokens_approx(detail)
        if not detail.strip() or len(blocks) <= 2:
            debug["skip_reason"] = "empty_detail_context"
            return "", debug
        return detail, debug

    @staticmethod
    def _current_direct_hit_needs_targeted_detail(recalled_memory: str, intent: dict[str, bool]) -> bool:
        if intent.get("favorite_reason"):
            rendered = str(recalled_memory or "").lower()
            if "bucket_window" in rendered or "bucket_capsule" in rendered:
                return True
            return not GatewayService._rendered_has_reflection_detail(rendered)
        if intent.get("reflection"):
            rendered = str(recalled_memory or "").lower()
            if "bucket_window" in rendered or "bucket_capsule" in rendered:
                return True
            return not GatewayService._rendered_has_reflection_detail(rendered)
        return False

    @staticmethod
    def _rendered_has_reflection_detail(rendered: str) -> bool:
        text = str(rendered or "").lower()
        return any(
            marker in text
            for marker in (
                "### reflection",
                " reflection ",
                "favorite_reason",
                "favorite reason",
                "喜欢它的原因",
                "喜欢的原因",
                "由此确认",
                "因此确认",
                "明白",
                "理解",
            )
        )

    def _recent_memory_reference_context(
        self,
        session_id: str,
        *,
        bucket_ids: list[str],
        moment_ids: list[str],
    ) -> str:
        items = self.state_store.list_injection_debug(
            session_id=session_id,
            limit=1,
            include_context=False,
        )
        if not items:
            return ""
        payload = items[0].get("payload") if isinstance(items[0], dict) else {}
        if not isinstance(payload, dict):
            return ""
        return self._filter_reference_context_lines(
            str(payload.get("diffused_memory") or ""),
            bucket_ids=bucket_ids,
            moment_ids=moment_ids,
        )

    @staticmethod
    def _filter_reference_context_lines(
        text: str,
        *,
        bucket_ids: list[str],
        moment_ids: list[str],
    ) -> str:
        bucket_set = {str(item) for item in bucket_ids if str(item or "").strip()}
        moment_set = {str(item) for item in moment_ids if str(item or "").strip()}
        lines = []
        for line in str(text or "").splitlines():
            line_bucket_ids = set(re.findall(r"\[bucket_id:([^\]\s]+)\]", line))
            line_moment_ids = set(re.findall(r"\[moment_id:([^\]\s]+)\]", line))
            if (bucket_set and line_bucket_ids & bucket_set) or (moment_set and line_moment_ids & moment_set):
                lines.append(line.strip())
        return "\n".join(line for line in lines if line)

    def _format_targeted_bucket_detail(
        self,
        bucket: dict,
        moments: list[dict[str, Any]],
        *,
        selected_moment_ids: list[str],
        intent: dict[str, bool],
    ) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        bucket_id = str(bucket.get("id") or "")
        title = str(meta.get("name") or bucket_id)
        header_parts = [f"[bucket_id:{bucket_id}]"]
        header_parts.extend(self._bucket_date_meta_parts(bucket))
        header = " ".join(header_parts + [title]).strip()
        selected = [moment for moment in moments if str(moment.get("moment_id") or "") in set(selected_moment_ids)]
        by_section: dict[str, list[dict[str, Any]]] = {}
        for moment in moments:
            by_section.setdefault(str(moment.get("section") or "body"), []).append(moment)

        lines = [header]

        def add_section(title: str, section_names: tuple[str, ...], limit: int) -> None:
            items: list[dict[str, Any]] = []
            for section in section_names:
                items.extend(by_section.get(section, []))
            if not items:
                return
            ordered: list[dict[str, Any]] = []
            for moment in selected:
                if moment in items and moment not in ordered:
                    ordered.append(moment)
            for moment in items:
                if moment not in ordered:
                    ordered.append(moment)
            lines.append(title)
            for moment in ordered[:limit]:
                lines.append(f"- [moment_id:{moment.get('moment_id') or ''}] {self._moment_text(moment, 320)}")

        if selected or intent.get("raw"):
            add_section("### moment", ("body", "moment", "fact", "original", "evidence_context", "context"), 2)
        if intent.get("reflection") or intent.get("favorite_reason"):
            add_section("### reflection", ("reflection", "favorite_reason"), 3)
        if intent.get("raw") or intent.get("reflection") or intent.get("favorite_reason"):
            add_section("### followup", ("followup", "comment"), 2)
        return "\n".join(lines).strip()

    @staticmethod
    def _parse_memory_detail_request(text: str) -> tuple[list[str], str]:
        raw = str(text or "")
        match = MEMORY_DETAIL_REQUEST_RE.match(raw)
        if not match:
            return [], raw
        requested_ids = []
        seen = set()
        for item in re.split(r"[\s,，、;；|]+", match.group("ids")):
            bucket_id = item.strip()
            if not bucket_id or bucket_id in seen:
                continue
            seen.add(bucket_id)
            requested_ids.append(bucket_id)
        return requested_ids, raw[match.end():].lstrip()

    def _response_with_assistant_content(
        self,
        upstream_response: httpx.Response,
        body: dict[str, Any],
        content: str,
    ) -> httpx.Response:
        updated = deepcopy(body)
        message = self._extract_assistant_message_from_response_body(updated)
        if isinstance(message, dict):
            message["content"] = content
        return httpx.Response(
            upstream_response.status_code,
            json=updated,
            headers={"content-type": "application/json"},
        )

    async def _build_memory_detail_recall_context(self, bucket_ids: list[str]) -> tuple[str, list[str]]:
        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        bucket_map = {
            str(bucket.get("id") or ""): bucket
            for bucket in all_buckets
            if isinstance(bucket, dict) and bucket.get("id")
        }
        requested = [bucket_id for bucket_id in bucket_ids if bucket_id]
        if not requested:
            return "", []
        per_bucket_budget = max(80, self.memory_detail_recall_budget // max(1, len(requested)))
        blocks = [
            "Additional private memory detail for this turn. Use quietly when relevant; do not mention lookup.",
        ]
        missing_ids = []
        for bucket_id in requested:
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                missing_ids.append(bucket_id)
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            title = str(meta.get("name") or bucket_id)
            original = self._rendered_bucket_content(bucket)
            date_part = " ".join(self._bucket_date_meta_parts(bucket))
            header = f"[bucket_id:{bucket_id}] {date_part} {title}".strip()
            block = f"{header}\nbucket_detail\n{original}".strip()
            blocks.append(self._trim_text(block, per_bucket_budget))
        return "\n\n".join(part for part in blocks if part.strip()), missing_ids

    def _insert_memory_detail_context(self, messages: Any, detail_context: str) -> list[dict]:
        new_messages = deepcopy(messages) if isinstance(messages, list) else []
        detail_message = {"role": "system", "content": detail_context}
        insert_at = self._after_leading_system_index(new_messages)
        new_messages.insert(insert_at, detail_message)
        return new_messages

    async def _update_persona_after_response(
        self,
        session_id: str,
        user_message: str,
        upstream_response: httpx.Response,
        recalled_ids: list[str],
    ) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            logger.info(
                "Persona post-reply update skipped | session=%s reason=non_json_response",
                session_id,
            )
            return
        assistant_message = self._extract_assistant_message_from_response_body(body)
        await self._update_persona_after_assistant_message(
            session_id,
            user_message,
            assistant_message,
            recalled_ids,
        )

    async def _update_persona_after_assistant_message(
        self,
        session_id: str,
        user_message: str,
        assistant_message: dict[str, Any] | None,
        recalled_ids: list[str],
    ) -> None:
        if not self.persona_engine.enabled:
            logger.info(
                "Persona post-reply update skipped | session=%s reason=disabled",
                session_id,
            )
            return
        if not user_message.strip():
            logger.info(
                "Persona post-reply update skipped | session=%s reason=missing_user_message",
                session_id,
            )
            return
        if not isinstance(assistant_message, dict):
            logger.info(
                "Persona post-reply update skipped | session=%s reason=missing_assistant_message",
                session_id,
            )
            return
        tool_calls = assistant_message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            logger.info(
                "Persona post-reply update skipped | session=%s reason=assistant_tool_calls",
                session_id,
            )
            return
        assistant_response = self._coerce_message_text(assistant_message.get("content")).strip()
        if not assistant_response:
            logger.info(
                "Persona post-reply update skipped | session=%s reason=empty_assistant_text",
                session_id,
            )
            return
        tool_summary = self._summarize_assistant_tool_calls(assistant_message)
        try:
            await self.persona_engine.update_from_exchange(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                recalled_memory_ids=recalled_ids,
                tool_summary=tool_summary,
            )
        except Exception as exc:
            logger.warning("Persona post-reply update failed | session=%s error=%s", session_id, exc)

    async def _finalize_stream_turn(
        self,
        session_id: str,
        model: str,
        route: str,
        stream_state: dict[str, Any],
        recalled_ids: list[str] | None,
        user_message: str,
        injection_debug: dict[str, Any] | None = None,
    ) -> None:
        self._log_cache_usage_from_stream_state(
            session_id,
            model,
            stream_state,
            route=route,
        )
        self._capture_reasoning_from_stream_state(session_id, stream_state)
        await self._record_successful_round(session_id, recalled_ids, injection_debug)
        assistant_message = self._build_stream_assistant_message(stream_state)
        self._schedule_persona_post_reply_update(
            session_id,
            user_message,
            assistant_message,
            recalled_ids or [],
        )

    def _schedule_persona_post_reply_update(
        self,
        session_id: str,
        user_message: str,
        assistant_message: dict[str, Any] | None,
        recalled_ids: list[str],
    ) -> None:
        async def runner() -> None:
            await self._update_persona_after_assistant_message(
                session_id,
                user_message,
                assistant_message,
                recalled_ids,
            )

        task = asyncio.create_task(runner())
        task.add_done_callback(
            lambda done: self._log_persona_post_update_task(session_id, done)
        )

    def _log_persona_post_update_task(self, session_id: str, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.warning(
                "Persona post-reply background update failed | session=%s error=%s",
                session_id,
                exc,
            )

    def _summarize_assistant_tool_calls(self, assistant_message: dict[str, Any]) -> str:
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return ""
        parts: list[str] = []
        for index, tool_call in enumerate(tool_calls[:8]):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or f"tool_{index}")
            arguments = self._normalize_tool_arguments(function.get("arguments", ""))
            if len(arguments) > 160:
                arguments = arguments[:157] + "..."
            parts.append(f"{name}({arguments})")
        return "; ".join(parts)

    def _log_cache_usage_from_response(
        self,
        session_id: str,
        model: str,
        upstream_response: httpx.Response,
        route: str,
    ) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            return
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            self._log_cache_usage(session_id, model, route, usage)

    def _log_cache_usage_from_stream_state(
        self,
        session_id: str,
        model: str,
        stream_state: dict[str, Any],
        route: str,
    ) -> None:
        usage = stream_state.get("usage")
        if isinstance(usage, dict):
            self._log_cache_usage(session_id, model, route, usage)

    def _log_cache_usage(self, session_id: str, model: str, route: str, usage: dict[str, Any]) -> None:
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_creation_tokens = usage.get("cache_creation_input_tokens")
        prompt_details = usage.get("prompt_tokens_details")
        cached_tokens = None
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")

        if (
            hit is None
            and miss is None
            and cached_tokens is None
            and cache_read_tokens is None
            and cache_creation_tokens is None
        ):
            return

        logger.info(
            "Gateway upstream cache usage | session=%s model=%s route=%s "
            "prompt_tokens=%s completion_tokens=%s prompt_cache_hit_tokens=%s "
            "prompt_cache_miss_tokens=%s cached_tokens=%s cache_read_input_tokens=%s "
            "cache_creation_input_tokens=%s",
            session_id,
            model,
            route,
            prompt_tokens,
            completion_tokens,
            hit,
            miss,
            cached_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        )

    def _proxy_response(self, upstream_response: httpx.Response) -> Response:
        content_type = upstream_response.headers.get("content-type", "application/json")
        try:
            body = upstream_response.json()
            return JSONResponse(body, status_code=upstream_response.status_code)
        except ValueError:
            return Response(
                content=upstream_response.text,
                status_code=upstream_response.status_code,
                media_type=content_type,
            )

    def _anthropic_request_to_openai(self, payload: dict) -> dict:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        openai_messages: list[dict[str, Any]] = []
        system_text = self._anthropic_content_to_text(payload.get("system"), "system").strip()
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            openai_messages.extend(self._anthropic_message_to_openai_messages(message, index))

        openai_payload: dict[str, Any] = {
            "model": payload.get("model"),
            "messages": openai_messages,
            "stream": payload.get("stream") is True,
        }

        passthrough_fields = ("max_tokens", "temperature", "top_p")
        for field in passthrough_fields:
            if field in payload:
                openai_payload[field] = payload[field]

        if "stop_sequences" in payload:
            openai_payload["stop"] = payload["stop_sequences"]
        elif "stop" in payload:
            openai_payload["stop"] = payload["stop"]

        tools = self._anthropic_tools_to_openai(payload.get("tools"))
        if tools:
            openai_payload["tools"] = tools

        tool_choice = self._anthropic_tool_choice_to_openai(payload.get("tool_choice"))
        if tool_choice is not None:
            openai_payload["tool_choice"] = tool_choice

        return openai_payload

    def _anthropic_message_to_openai_messages(self, message: dict[str, Any], index: int) -> list[dict[str, Any]]:
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"messages[{index}].role must be user, assistant, or system")

        content = message.get("content")
        if role == "system":
            return [{"role": "system", "content": self._anthropic_content_to_text(content, f"messages[{index}].content")}]
        if isinstance(content, str) or content is None:
            return [{"role": role, "content": content or ""}]
        if not isinstance(content, list):
            raise ValueError(f"messages[{index}].content must be a string or block list")

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block_index, block in enumerate(content):
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                if not isinstance(block, dict):
                    raise ValueError(f"messages[{index}].content[{block_index}] must be an object")
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                    continue
                if block_type == "tool_use":
                    tool_id = str(block.get("id") or "")
                    name = str(block.get("name") or "")
                    if not tool_id or not name:
                        raise ValueError(f"messages[{index}].content[{block_index}] tool_use requires id and name")
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
                    continue
                raise ValueError(f"messages[{index}].content[{block_index}] unsupported assistant block type")

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part) or None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            return [assistant_message]

        output: list[dict[str, Any]] = []
        pending_text: list[str] = []
        for block_index, block in enumerate(content):
            if isinstance(block, str):
                pending_text.append(block)
                continue
            if not isinstance(block, dict):
                raise ValueError(f"messages[{index}].content[{block_index}] must be an object")
            block_type = block.get("type")
            if block_type == "text":
                pending_text.append(str(block.get("text") or ""))
                continue
            if block_type == "tool_result":
                if pending_text:
                    output.append({"role": "user", "content": "\n".join(part for part in pending_text if part)})
                    pending_text = []
                tool_use_id = str(block.get("tool_use_id") or "")
                if not tool_use_id:
                    raise ValueError(f"messages[{index}].content[{block_index}] tool_result requires tool_use_id")
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": self._anthropic_content_to_text(
                            block.get("content"),
                            f"messages[{index}].content[{block_index}].content",
                        ),
                    }
                )
                continue
            raise ValueError(f"messages[{index}].content[{block_index}] unsupported user block type")

        if pending_text or not output:
            output.append({"role": "user", "content": "\n".join(part for part in pending_text if part)})
        return output

    def _anthropic_tools_to_openai(self, tools: Any) -> list[dict[str, Any]]:
        if tools is None:
            return []
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        converted = []
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"tools[{index}] must be an object")
            name = str(tool.get("name") or "")
            if not name:
                raise ValueError(f"tools[{index}].name is required")
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description") or ""),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
        return converted

    def _anthropic_tool_choice_to_openai(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return {"auto": "auto", "any": "required", "none": "none"}.get(tool_choice, tool_choice)
        if not isinstance(tool_choice, dict):
            raise ValueError("tool_choice must be a string or object")
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "none":
            return "none"
        if choice_type == "tool":
            name = str(tool_choice.get("name") or "")
            if not name:
                raise ValueError("tool_choice.name is required when type is tool")
            return {"type": "function", "function": {"name": name}}
        return None

    def _anthropic_content_to_text(self, content: Any, field_name: str) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for index, block in enumerate(content):
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if not isinstance(block, dict):
                    raise ValueError(f"{field_name}[{index}] must be a text block")
                block_type = block.get("type")
                if block_type != "text":
                    raise ValueError(f"{field_name}[{index}] only supports text blocks")
                parts.append(str(block.get("text") or ""))
            return "\n".join(part for part in parts if part)
        raise ValueError(f"{field_name} must be a string or text block list")

    def _openai_response_to_anthropic(self, upstream_response: httpx.Response, requested_model: str) -> JSONResponse:
        try:
            body = upstream_response.json()
        except ValueError:
            return self._anthropic_error(
                "Upstream response was not valid JSON",
                status_code=502,
                error_type="api_error",
            )

        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        content_blocks = self._openai_message_to_anthropic_content(message)
        raw_id = str(body.get("id") or "ombre")
        response_id = raw_id if raw_id.startswith("msg_") else f"msg_{raw_id}"
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}

        return JSONResponse(
            {
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "model": body.get("model") or requested_model,
                "content": content_blocks,
                "stop_reason": self._openai_finish_reason_to_anthropic(finish_reason),
                "stop_sequence": None,
                "usage": {
                    "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
                },
            },
            status_code=upstream_response.status_code,
        )

    def _openai_message_to_anthropic_content(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = []
        text = self._coerce_message_text(message.get("content"))
        if text:
            content_blocks.append({"type": "text", "text": text})

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "")
                if not name:
                    continue
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or f"call_{len(content_blocks)}"),
                        "name": name,
                        "input": self._parse_tool_arguments(function.get("arguments")),
                    }
                )
        return content_blocks

    def _parse_tool_arguments(self, raw_arguments: Any) -> Any:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if raw_arguments in (None, ""):
            return {}
        if not isinstance(raw_arguments, str):
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _stream_upstream_as_anthropic(
        self,
        payload: dict,
        session_id: str,
        recalled_ids: list[str] | None,
        user_message: str,
        injection_debug: dict[str, Any] | None = None,
    ) -> Response:
        model = str(payload.get("model") or "").strip()
        route = self._resolve_upstream_for_model(model)
        upstream_response = await self._open_upstream_stream(route, payload)

        if not 200 <= upstream_response.status_code < 300:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return self._proxy_anthropic_error_response(
                httpx.Response(
                    status_code=upstream_response.status_code,
                    content=body,
                    headers=upstream_response.headers,
                )
            )

        async def stream_body():
            finalized = False
            stream_state = self._new_stream_capture_state()
            message_id = f"msg_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            usage = {"input_tokens": 0, "output_tokens": 0}
            stop_reason = "end_turn"
            next_block_index = 0
            text_block_index: int | None = None
            tool_blocks: dict[int, dict[str, Any]] = {}

            async def finalize_once() -> None:
                nonlocal finalized
                if finalized:
                    return
                finalized = True
                await self._finalize_stream_turn(
                    session_id=session_id,
                    model=model,
                    route="/v1/messages",
                    stream_state=stream_state,
                    recalled_ids=recalled_ids,
                    user_message=user_message,
                    injection_debug=injection_debug,
                )

            try:
                yield self._anthropic_sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": usage,
                        },
                    },
                )

                async for chunk in upstream_response.aiter_bytes():
                    if not chunk:
                        continue
                    self._consume_stream_capture_chunk(stream_state, chunk)
                    if stream_state.get("seen_done"):
                        await finalize_once()
                    for event in self._openai_sse_chunk_to_anthropic_events(chunk):
                        if event.get("_done"):
                            continue
                        if event.get("usage"):
                            usage.update(event["usage"])
                            continue
                        if event.get("stop_reason"):
                            stop_reason = event["stop_reason"]
                            continue
                        if event.get("text"):
                            if text_block_index is None:
                                text_block_index = next_block_index
                                next_block_index += 1
                                yield self._anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                            yield self._anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": event["text"],
                                    },
                                },
                            )
                            continue
                        tool_call = event.get("tool_call")
                        if isinstance(tool_call, dict):
                            tool_index = int(tool_call.get("index", 0))
                            state = tool_blocks.setdefault(
                                tool_index,
                                {
                                    "content_index": None,
                                    "id": "",
                                    "name": "",
                                    "started": False,
                                },
                            )
                            if tool_call.get("id"):
                                state["id"] = str(tool_call["id"])
                            if tool_call.get("name"):
                                state["name"] = str(tool_call["name"])
                            if not state["started"] and state["name"]:
                                state["content_index"] = next_block_index
                                next_block_index += 1
                                state["started"] = True
                                yield self._anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": state["content_index"],
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": state["id"] or f"call_{tool_index}",
                                            "name": state["name"],
                                            "input": {},
                                        },
                                    },
                                )
                            arguments = tool_call.get("arguments")
                            if state["started"] and arguments:
                                yield self._anthropic_sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": state["content_index"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": arguments,
                                        },
                                },
                            )

                self._consume_stream_capture_chunk(stream_state, b"", final=True)
                await finalize_once()
                if text_block_index is not None:
                    yield self._anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_block_index},
                    )
                for state in sorted(
                    (item for item in tool_blocks.values() if item.get("started")),
                    key=lambda item: int(item["content_index"]),
                ):
                    yield self._anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": state["content_index"]},
                    )
                yield self._anthropic_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": usage.get("output_tokens", 0)},
                    },
                )
                yield self._anthropic_sse(
                    "message_stop",
                    {"type": "message_stop"},
                )
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def _openai_sse_chunk_to_anthropic_events(self, chunk: bytes) -> list[dict[str, Any]]:
        text = chunk.decode("utf-8", errors="ignore")
        events: list[dict[str, Any]] = []
        for raw_event in text.split("\n\n"):
            for line in raw_event.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    events.append({"_done": True})
                    continue
                try:
                    body = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = body.get("usage")
                if isinstance(usage, dict):
                    events.append(
                        {
                            "usage": {
                                "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                                "output_tokens": int(
                                    usage.get("output_tokens") or usage.get("completion_tokens") or 0
                                ),
                            }
                        }
                    )
                choices = body.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        events.append({"stop_reason": self._openai_finish_reason_to_anthropic(finish_reason)})
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    tool_calls = delta.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tool_call in tool_calls:
                            if not isinstance(tool_call, dict):
                                continue
                            function = tool_call.get("function")
                            if not isinstance(function, dict):
                                function = {}
                            events.append(
                                {
                                    "tool_call": {
                                        "index": int(tool_call.get("index") or 0),
                                        "id": tool_call.get("id"),
                                        "name": function.get("name"),
                                        "arguments": function.get("arguments"),
                                    }
                                }
                            )
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        events.append({"text": content})
        return events

    def _anthropic_sse(self, event: str, data: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

    def _openai_finish_reason_to_anthropic(self, finish_reason: Any) -> str:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "stop_sequence",
        }
        return mapping.get(str(finish_reason or ""), "end_turn")

    def _proxy_anthropic_error_response(self, upstream_response: httpx.Response) -> JSONResponse:
        message = upstream_response.text or "Upstream request failed"
        error_type = "api_error"
        try:
            body = upstream_response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                error_type = str(error.get("type") or error_type)
            elif body.get("message"):
                message = str(body["message"])
        return self._anthropic_error(
            message,
            status_code=upstream_response.status_code,
            error_type=error_type,
        )

    def _anthropic_error(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
    ) -> JSONResponse:
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            status_code=status_code,
        )

    def _extract_last_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = self._coerce_message_text(message.get("content"))
            if content.strip():
                return content.strip()
        return ""

    def _extract_current_turn_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                continue
            if role != "user":
                return ""
            content = self._coerce_message_text(message.get("content"))
            cleaned = self._strip_external_context_from_user_text(content)
            if cleaned:
                return cleaned
            continue
        return ""

    def _coerce_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    text = item.get("text") or item.get("input_text") or ""
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks)
        return ""

    def _strip_external_context_from_user_text(self, text: str) -> str:
        cleaned = WORKSPACE_ATTACHMENT_RE.sub("", str(text or ""))
        cleaned = EXTERNAL_CONTEXT_ATTACHMENT_RE.sub("", cleaned)
        cleaned = SELF_CLOSING_ATTACHMENT_RE.sub("", cleaned)
        cleaned = self._strip_leading_auto_context_markers(cleaned)
        return self._strip_external_context_blocks(cleaned)

    def _strip_leading_auto_context_markers(self, text: str) -> str:
        cleaned = str(text or "")
        while True:
            previous = cleaned
            cleaned = LEADING_PROXY_SENDER_RE.sub("", cleaned, count=1)
            cleaned = LEADING_SYSTEM_PROMPT_RE.sub("", cleaned, count=1)
            if cleaned == previous:
                return cleaned

    def _strip_external_context_blocks(self, text: str) -> str:
        kept: list[str] = []
        skipping = False
        for line in str(text or "").splitlines():
            stripped = line.strip()
            title = ""
            if stripped.startswith("【") and "】" in stripped:
                title = stripped[1 : stripped.index("】")].strip()
            if title:
                skipping = title in EXTERNAL_CONTEXT_BLOCK_TITLES
                if skipping:
                    continue
            if not skipping:
                kept.append(line)
        return "\n".join(kept).strip()

    def _summarize_messages_for_debug(self, messages: Any) -> list[dict[str, Any]] | str:
        if not isinstance(messages, list):
            return "<invalid>"

        summary: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                summary.append({"idx": index, "type": type(message).__name__})
                continue

            item: dict[str, Any] = {
                "idx": index,
                "role": str(message.get("role") or ""),
            }
            if self._coerce_message_text(message.get("content")).strip():
                item["has_text"] = True
            if isinstance(message.get("reasoning_content"), str) and message.get("reasoning_content"):
                item["has_reasoning"] = True

            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                item["tool_call_id"] = str(tool_call_id)

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                labels = []
                for tool_index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        labels.append(f"idx:{tool_index}")
                        continue
                    if tool_call.get("id"):
                        labels.append(str(tool_call["id"]))
                        continue
                    function = tool_call.get("function", {})
                    if isinstance(function, dict) and function.get("name"):
                        labels.append(f"idx:{tool_index}:{function['name']}")
                        continue
                    labels.append(f"idx:{tool_index}")
                item["tool_call_ids"] = labels

            summary.append(item)

        return summary

    def _should_inject_interval(self, session_id: str, interval_rounds: int) -> bool:
        if interval_rounds <= 0:
            return False
        if interval_rounds == 1:
            return True
        next_round = self.state_store.get_current_round(session_id) + 1
        return next_round == 1 or next_round % interval_rounds == 0

    def _get_persona_state_for_context_mode(self, session_id: str) -> dict[str, Any]:
        getter = getattr(self.persona_engine, "get_current_state", None)
        if not callable(getter):
            return {}
        try:
            state = getter(session_id)
        except Exception as exc:
            logger.warning("Gateway context mode state lookup failed | session=%s error=%s", session_id, exc)
            return {}
        return state if isinstance(state, dict) else {}

    def _classify_context_mode(self, query: str, persona_state: dict[str, Any] | None = None) -> str:
        text = " ".join(str(query or "").lower().split())
        state = persona_state if isinstance(persona_state, dict) else {}
        affect = state.get("affect", {}) if isinstance(state.get("affect"), dict) else {}
        relationship = state.get("relationship", {}) if isinstance(state.get("relationship"), dict) else {}
        defensiveness = self._safe_float(relationship.get("defensiveness"), 0.0)
        security = self._safe_float(affect.get("security"), 0.5)
        tenderness = self._safe_float(affect.get("tenderness"), 0.0)
        longing = self._safe_float(affect.get("longing"), 0.0)

        conflict_terms = (
            "冲突", "吵架", "争吵", "矛盾", "误会", "生气", "闹别扭",
            "conflict", "fight", "argument", "angry", "upset",
        )
        repair_terms = (
            "修复", "和好", "道歉", "解释", "哪里不对", "为什么", "怎么会",
            "repair", "resolve", "apolog", "what happened", "why did",
        )
        reflective_terms = (
            "反思", "想想之前", "之前怎么", "旧版本", "旧版", "旧链", "旧窗口",
            "恢复", "找回", "连续性", "过去那段", "reflect", "old version",
            "old path", "previous version", "continuity",
        )
        memory_terms = (
            "记忆", "记得", "想起", "回忆", "查一下", "找一下", "哪段",
            "以前", "过去", "remember", "recall", "memory", "look up",
        )
        intimate_terms = (
            "今天是雨天", "亲亲", "抱抱", "抱我", "吻", "亲密", "想你", "爱你",
            "老婆", "宝宝", "亲爱的", "身体", "欲望", "intimate", "kiss",
            "hug", "miss you", "love you",
        )
        playful_terms = (
            "哈哈", "嘿嘿", "逗你", "调戏", "开玩笑", "撒娇", "坏东西",
            "joke", "playful", "tease", "flirt",
        )
        task_terms = (
            "代码", "bug", "报错", "测试", "部署", "接口", "配置", "文件", "分支",
            "实现", "排查", "工作", "需求", "pytest", "python", "node", "gateway",
            "test", "debug", "deploy", "config", "branch",
        )

        has_conflict = self._text_has_any(text, conflict_terms)
        has_repair = self._text_has_any(text, repair_terms)
        if has_conflict and (has_repair or defensiveness >= 0.35 or security <= 0.4):
            return "conflict_repair"
        if self._text_has_any(text, reflective_terms):
            return "reflective_repair"
        if self._text_has_any(text, memory_terms):
            return "memory_lookup"
        if self._text_has_any(text, intimate_terms) or (
            tenderness >= 0.78 and longing >= 0.45 and self._text_has_any(text, ("你", "我们", "haven", "小雨"))
        ):
            return "intimate"
        if self._text_has_any(text, playful_terms):
            return "playful"
        if self._text_has_any(text, task_terms):
            return "task"
        return "task"

    @staticmethod
    def _text_has_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term and term in text for term in terms)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _query_requests_favorite_memory(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False
        ai_name = str(self.identity.get("ai_name") or "").lower()
        direct_phrases = [
            "favorite memory",
            "favorite memories",
            f"{ai_name} favorite" if ai_name else "",
            "偏爱的记忆",
            "喜欢的记忆",
            "喜欢哪段记忆",
            "最喜欢哪段",
            "最偏爱",
            "哪段记忆最",
            "记忆里最",
            "哪一刻最",
            "哪个瞬间最",
            "想起我们什么",
            "想起了我们什么",
            "我们哪段",
            "我们哪一刻",
            "我们哪个瞬间",
        ]
        if any(phrase in text for phrase in direct_phrases):
            return True
        asks_memory = any(term in text for term in ["记忆", "想起", "记得"])
        asks_preference = any(term in text for term in ["喜欢", "偏爱", "重要", "哪段", "哪一刻", "哪个瞬间"])
        relationship_terms = ["我们", "你"]
        relationship_terms.extend(str(term).lower() for term in self.identity.get("relationship_terms", []))
        relationship_scope = any(term and term in text for term in relationship_terms)
        return asks_memory and asks_preference and relationship_scope

    def _truthy_header(self, value: str | None) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _strip_favorite_memory_marker_from_payload(self, payload: dict) -> tuple[dict, bool]:
        cleaned = deepcopy(payload)
        messages = cleaned.get("messages")
        if not isinstance(messages, list):
            return cleaned, False

        current_user_index = self._current_turn_user_index(messages)
        marker_in_current_turn = False
        for index, message in enumerate(messages):
            found = self._strip_favorite_memory_marker_from_message(message)
            if found and index == current_user_index:
                marker_in_current_turn = True
        return cleaned, marker_in_current_turn

    def _strip_favorite_memory_marker_from_message(self, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if isinstance(content, str):
            stripped, found = self._strip_favorite_memory_marker_from_text(content)
            if found:
                message["content"] = stripped
            return found
        if not isinstance(content, list):
            return False

        found_any = False
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("text", "input_text"):
                value = item.get(key)
                if not isinstance(value, str):
                    continue
                stripped, found = self._strip_favorite_memory_marker_from_text(value)
                if found:
                    item[key] = stripped
                    found_any = True
        return found_any

    def _strip_favorite_memory_marker_from_text(self, text: str) -> tuple[str, bool]:
        if FAVORITE_MEMORY_MARKER not in text:
            return text, False
        return text.replace(FAVORITE_MEMORY_MARKER, "").strip(), True

    async def _build_core_memory_block(self, all_buckets: list[dict]) -> str:
        core_buckets = [
            bucket for bucket in all_buckets
            if bucket.get("metadata", {}).get("pinned") or bucket.get("metadata", {}).get("protected")
        ]
        core_buckets.sort(
            key=lambda bucket: (
                int(bucket.get("metadata", {}).get("importance", 0)),
                bucket.get("metadata", {}).get("last_active", ""),
            ),
            reverse=True,
        )
        return await self._summarize_buckets(core_buckets, self.core_budget)

    def _build_portrait_memory_block(self, all_buckets: list[dict]) -> tuple[str, dict[str, Any]]:
        debug = self._portrait_memory_debug_base()
        if not self.portrait_memory_enabled:
            debug["skip_reason"] = "disabled"
            return "", debug

        sources = self._select_portrait_memory_sources(all_buckets)
        debug["source_count"] = len(sources)
        debug["source_ids"] = [str(bucket.get("id") or "") for _role, bucket in sources if bucket.get("id")]
        debug["source_roles"] = [
            {
                "bucket_id": str(bucket.get("id") or ""),
                "role": role,
            }
            for role, bucket in sources
            if bucket.get("id")
        ]
        if not sources:
            debug["skip_reason"] = "no_sources"
            return "", debug

        cache_key, source_hash = self._portrait_memory_cache_key(sources)
        debug["source_hash"] = source_hash
        cached_key = self._portrait_memory_cache.get("key")
        if cached_key == cache_key:
            cached_debug = dict(self._portrait_memory_cache.get("debug") or {})
            cached_debug["enabled"] = True
            cached_debug["cache_hit"] = True
            cached_debug["skip_reason"] = ""
            return str(self._portrait_memory_cache.get("block") or ""), cached_debug

        lines = [
            "Read-only user portrait cache compiled from evidence-bound profile facts and selected long-term anchors.",
            "Use quietly when relevant; do not treat this as Core Memory, and do not infer beyond these lines.",
        ]
        for role, bucket in sources:
            line = self._portrait_memory_source_line(role, bucket)
            if line:
                lines.append(line)

        block = self._trim_text("\n".join(lines), self.portrait_memory_budget)
        debug["cache_hit"] = False
        debug["generated_portrait_version"] = "portrait-v1-deterministic"
        debug["token_estimate"] = count_tokens_approx(block)
        if not block.strip():
            debug["skip_reason"] = "empty_block"
            return "", debug

        self._portrait_memory_cache = {
            "key": cache_key,
            "block": block,
            "debug": dict(debug),
        }
        return block, debug

    def _portrait_memory_debug_base(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self, "portrait_memory_enabled", False)),
            "cache_hit": False,
            "skip_reason": "",
            "source_count": 0,
            "source_ids": [],
            "source_roles": [],
            "source_hash": "",
            "token_estimate": 0,
            "generated_portrait_version": "",
        }

    def _select_portrait_memory_sources(self, all_buckets: list[dict]) -> list[tuple[str, dict]]:
        sources: list[tuple[str, dict]] = []
        for bucket in all_buckets:
            if not isinstance(bucket, dict):
                continue
            role = self._portrait_memory_source_role(bucket)
            if role:
                sources.append((role, bucket))

        def source_sort_key(item: tuple[str, dict]) -> tuple[int, float, int, str]:
            role, bucket = item
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            role_rank = 2 if role == "profile_fact" else 1
            confidence = self._safe_float(meta.get("confidence"), 0.0)
            importance = int(meta.get("importance") or 0)
            updated = str(meta.get("updated_at") or meta.get("last_active") or meta.get("created") or "")
            return (role_rank, confidence, importance, updated)

        sources.sort(key=source_sort_key, reverse=True)
        return sources[: self.portrait_memory_max_sources]

    def _portrait_memory_source_role(self, bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("pinned") or meta.get("protected"):
            return ""
        if meta.get("resolved") or meta.get("digested") or meta.get("deprecated"):
            return ""
        if meta.get("active") is False:
            return ""
        tags = {str(tag).strip() for tag in meta.get("tags", []) or [] if str(tag).strip()}
        if "profile_fact" in tags or meta.get("profile_kind"):
            return "profile_fact"
        if self.portrait_memory_include_anchors and meta.get("anchor"):
            return "anchor"
        return ""

    def _portrait_memory_cache_key(self, sources: list[tuple[str, dict]]) -> tuple[str, str]:
        rows = []
        for role, bucket in sources:
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            text = bucket_text_for_embedding(bucket)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "id": str(bucket.get("id") or meta.get("id") or ""),
                    "role": role,
                    "updated_at": str(meta.get("updated_at") or meta.get("last_active") or meta.get("created") or ""),
                    "content_hash": content_hash,
                }
            )
        key_payload = {
            "version": "portrait-v1-deterministic",
            "budget": self.portrait_memory_budget,
            "max_sources": self.portrait_memory_max_sources,
            "include_anchors": self.portrait_memory_include_anchors,
            "sources": rows,
        }
        key = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
        source_hash = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return key, source_hash

    def _portrait_memory_source_line(self, role: str, bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        bucket_id = str(bucket.get("id") or meta.get("id") or "").strip()
        text = bucket_text_for_embedding(bucket)
        text = strip_display_temperature_sections(strip_temperature_meaning_lines(text))
        text = re.sub(r"(?m)^(Title|Content):\s*", "", text)
        text = self._clip_text(text, 260 if role == "anchor" else 320)
        if not text:
            text = self._clip_text(str(meta.get("name") or bucket_id), 160)
        bits = [role]
        if bucket_id:
            bits.append(f"bucket_id:{bucket_id}")
        confidence = meta.get("confidence")
        if confidence is not None:
            bits.append(f"confidence:{self._safe_float(confidence, 0.0):.2f}")
        evidence = (
            meta.get("evidence_bucket_id")
            or meta.get("evidence_moment_id")
            or meta.get("source_bucket_id")
            or meta.get("source_moment_id")
        )
        if evidence:
            bits.append(f"evidence:{evidence}")
        return f"- [{' '.join(bits)}] {text}"

    async def _build_recent_context_block(
        self,
        all_buckets: list[dict],
        query_text: str = "",
        *,
        allow_vague: bool = False,
    ) -> str:
        if self._auto_query_too_vague(query_text) and not allow_vague:
            return ""
        cutoff = datetime.now() - timedelta(hours=self.head_recent_hours)
        enforce_topic = (
            self._recent_context_requires_topic_evidence(query_text)
            and not self._query_wants_body_chain(query_text)
        )
        recent_buckets = []
        explicit_recent_query = self._query_requests_recent_context(query_text)
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if not can_bucket_be_recent_context(bucket, explicit_lookup=explicit_recent_query):
                continue
            if enforce_topic and not self._bucket_has_query_topic_evidence(query_text, bucket):
                continue
            created = self._parse_iso(meta.get("created") or meta.get("last_active"))
            if created and created >= cutoff:
                recent_buckets.append(bucket)

        recent_buckets.sort(
            key=lambda bucket: bucket.get("metadata", {}).get("created", ""),
            reverse=True,
        )
        return await self._summarize_buckets(recent_buckets[:6], self.recent_budget)

    def _should_inject_recent_context(
        self,
        session_id: str,
        query_text: str,
        *,
        has_reliable_dynamic_context: bool = False,
    ) -> bool:
        if self.recent_budget <= 0 or self.head_recent_hours <= 0:
            return False
        if self._query_requests_recent_context(query_text):
            return True
        if self._auto_query_too_vague(query_text):
            return False
        if self._recent_context_in_cooldown(session_id):
            return False
        return bool(self._recent_context_reason(session_id, query_text, has_reliable_dynamic_context))

    def _recent_context_reason(
        self,
        session_id: str,
        query_text: str,
        has_reliable_dynamic_context: bool = False,
    ) -> str:
        if self._query_requests_recent_context(query_text):
            return "explicit_recent_query"
        if has_reliable_dynamic_context:
            return "reliable_dynamic_context"
        if self.state_store.get_current_round(session_id) <= 0:
            return "new_session"
        idle_hours = self._session_idle_hours(session_id)
        if (
            idle_hours is not None
            and self.recent_context_reentry_idle_hours > 0
            and idle_hours >= self.recent_context_reentry_idle_hours
        ):
            return "session_reentry"
        return ""

    def _recent_context_in_cooldown(self, session_id: str) -> bool:
        if self.recent_context_cooldown_hours <= 0:
            return False
        last_injected = self.state_store.get_last_recent_context_at(session_id)
        if not last_injected:
            return False
        elapsed_hours = max(0.0, (datetime.now() - last_injected).total_seconds() / 3600)
        return elapsed_hours < self.recent_context_cooldown_hours

    def _session_idle_hours(self, session_id: str) -> float | None:
        last_success = self.state_store.get_last_success_at(session_id)
        if not last_success:
            return None
        return max(0.0, (datetime.now() - last_success).total_seconds() / 3600)

    @staticmethod
    def _query_requests_recent_context(query: str) -> bool:
        text = " ".join(str(query or "").lower().split())
        if not text:
            return False
        explicit_phrases = (
            "最近记忆",
            "最近的记忆",
            "最近我们聊",
            "最近聊过",
            "最近说过",
            "最近提过",
            "最近发生",
            "最近记得",
            "刚才",
            "刚刚",
            "上次",
            "这几天",
            "这两天",
            "前几天",
            "之前聊",
            "之前说",
            "之前提",
            "recent memory",
            "recent memories",
            "what did we talk",
            "last time",
            "earlier",
        )
        return any(phrase in text for phrase in explicit_phrases)

    async def _build_relationship_weather_block(self, all_buckets: list[dict]) -> str:
        if self.relationship_weather_budget <= 0:
            return ""
        weather_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") != "feel":
                continue
            tags = {str(tag) for tag in meta.get("tags", [])}
            if not ({"relationship_weather", "daily_impression", "weekly_impression"} & tags):
                continue
            weather_buckets.append(bucket)
        if not weather_buckets:
            return ""

        daily = [
            bucket for bucket in weather_buckets
            if "daily_impression" in {str(tag) for tag in bucket.get("metadata", {}).get("tags", [])}
        ]
        daily.sort(key=lambda bucket: bucket.get("metadata", {}).get("created", ""), reverse=True)
        selected = daily[:7]
        if self.relationship_weather_include_weekly:
            weekly = [
                bucket for bucket in weather_buckets
                if "weekly_impression" in {str(tag) for tag in bucket.get("metadata", {}).get("tags", [])}
            ]
            weekly.sort(key=lambda bucket: bucket.get("metadata", {}).get("created", ""), reverse=True)
            selected = weekly[:1] + selected

        remaining = self.relationship_weather_budget
        parts = []
        for bucket in selected:
            meta = bucket.get("metadata", {})
            text = strip_wikilinks(bucket.get("content", "")).strip()
            if not text:
                continue
            prefix = meta.get("date") or meta.get("created", "")[:10]
            line = f"- [{prefix}] {self._trim_text(text, 80)}"
            tokens = count_tokens_approx(line)
            if tokens > remaining and parts:
                break
            parts.append(line)
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    async def _build_favorite_memory_block(
        self,
        all_buckets: list[dict],
        session_id: str,
    ) -> tuple[str, list[str]]:
        if self.favorite_memory_budget <= 0 or self.favorite_memory_max_cards <= 0:
            return "", []

        recent_ids = self.state_store.get_recent_bucket_ids(session_id, self.skip_recent_rounds)
        candidates = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            if "haven_favorite" not in tags:
                continue
            if not self._has_favorite_reason(bucket.get("content", "")):
                continue
            if meta.get("resolved") or meta.get("digested"):
                continue
            candidates.append(bucket)

        if not candidates:
            return "", []

        active_pool = [bucket for bucket in candidates if bucket.get("id") not in recent_ids] or candidates

        def favorite_key(bucket: dict) -> tuple[int, int, int, str]:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            flavor_count = sum(1 for tag in tags if tag.startswith("flavor_"))
            protected = 1 if (meta.get("anchor") or meta.get("pinned") or meta.get("protected")) else 0
            return (
                protected,
                flavor_count,
                int(meta.get("importance", 5)),
                str(meta.get("last_active") or meta.get("created") or ""),
            )

        active_pool.sort(key=favorite_key, reverse=True)
        selected = active_pool[: self.favorite_memory_max_cards]
        remaining = self.favorite_memory_budget
        parts = []
        selected_ids = []
        for bucket in selected:
            summary = await self._summarize_bucket(bucket)
            tokens = count_tokens_approx(summary)
            if tokens <= 0:
                continue
            if tokens > remaining and parts:
                break
            if tokens > remaining:
                summary = self._trim_text(summary, remaining)
                tokens = count_tokens_approx(summary)
            parts.append(f"- {summary}")
            selected_ids.append(bucket["id"])
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts), selected_ids

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

    def _refresh_moment_graph(
        self,
        all_buckets: list[dict],
    ) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
        self.memory_moment_store.bulk_upsert(all_buckets)
        moments = self._recallable_moments(self.memory_moment_store.list_all())
        grouped = self._moments_by_bucket(moments)
        edges = self.memory_moment_store.list_edges()
        edges.extend(self._bucket_edges_as_moment_edges(self.memory_edge_store.list_edges(), grouped))
        return moments, grouped, edges

    def _recallable_moments(self, moments: list[dict]) -> list[dict]:
        return [
            moment for moment in moments
            if can_moment_be_recall_context(moment)
        ]

    def _moments_by_bucket(self, moments: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for moment in moments:
            bucket_id = str(moment.get("bucket_id") or "")
            if bucket_id:
                grouped.setdefault(bucket_id, []).append(moment)
        for items in grouped.values():
            items.sort(key=lambda item: int(item.get("ordinal") or 0))
        return grouped

    def _representative_moment(self, moments: list[dict]) -> dict | None:
        for section in (
            "original",
            "moment",
            "fact",
            "body",
            "evidence_context",
            "context",
            "reflection",
            "feeling",
            "followup",
            "comment",
        ):
            for moment in moments:
                if moment.get("section") == section:
                    return moment
        return moments[0] if moments else None

    def _direct_representative_moment(
        self,
        moments: list[dict],
        *,
        explicit_lookup: bool = False,
    ) -> dict | None:
        return self._representative_moment(
            [
                moment for moment in self._recallable_moments(moments)
                if can_moment_be_direct_seed(moment, explicit_lookup=explicit_lookup)
            ]
        )

    def _direct_moments_for_bucket(self, bucket: dict, query: str = "") -> list[dict]:
        explicit_lookup = self._query_explicitly_requests_caution_memory(query)
        return [
            moment for moment in parse_bucket_moments(bucket, self.relevance_options)
            if can_moment_be_recall_context(moment)
            and can_moment_be_direct_seed(moment, explicit_lookup=explicit_lookup)
        ]

    def _related_representative_moment(
        self,
        moments: list[dict],
        *,
        explicit_lookup: bool = False,
    ) -> dict | None:
        return self._representative_moment(
            [
                moment for moment in self._recallable_moments(moments)
                if can_moment_be_related_target(moment, explicit_lookup=explicit_lookup)
            ]
        )

    def _representative_moments_by_bucket(
        self,
        moments: list[dict],
        *,
        explicit_lookup: bool = False,
    ) -> dict[str, dict]:
        grouped = self._moments_by_bucket(moments)
        representatives = {}
        for bucket_id, bucket_moments in grouped.items():
            representative = self._related_representative_moment(
                bucket_moments,
                explicit_lookup=explicit_lookup,
            )
            if representative:
                representatives[bucket_id] = representative
        return representatives

    def _bucket_edges_as_moment_edges(
        self,
        bucket_edges: list[dict],
        grouped: dict[str, list[dict]],
    ) -> list[dict]:
        edges = []
        for edge in bucket_edges or []:
            source_bucket = str(edge.get("source") or edge.get("source_memory_id") or "").strip()
            target_bucket = str(edge.get("target") or edge.get("target_memory_id") or "").strip()
            if not source_bucket or not target_bucket:
                continue
            target = self._related_representative_moment(
                grouped.get(target_bucket, []),
                explicit_lookup=True,
            )
            if not target:
                continue
            relation_type = str(edge.get("relation_type") or edge.get("type") or "relates_to")
            try:
                confidence = float(edge.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            for source in grouped.get(source_bucket, []):
                if not can_moment_be_direct_seed(source):
                    continue
                edges.append(
                    {
                        "source": source["moment_id"],
                        "target": target["moment_id"],
                        "bucket_id": source_bucket,
                        "relation_type": relation_type,
                        "confidence": max(0.0, min(1.0, confidence)),
                        "reason": edge.get("reason") or "bucket edge bridge",
                    }
                )
        return edges

    def _empty_moment_selection(
        self,
        *,
        include_query_planner_debug: bool = False,
        query_planner_debug: dict[str, Any] | None = None,
    ):
        if include_query_planner_debug:
            return [], [], [], [], (query_planner_debug or self._query_planner_debug_base(""))
        return [], [], [], []

    async def _select_dynamic_moments(
        self,
        query: str,
        session_id: str,
        all_buckets: list[dict],
        grouped_moments: dict[str, list[dict]],
        *,
        include_query_planner_debug: bool = False,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]] | tuple[
        list[dict], list[dict], list[dict], list[dict], dict[str, Any]
    ]:
        query_planner_debug = self._query_planner_debug_base(query)
        if not query or self.inject_max_cards <= 0:
            return self._empty_moment_selection(
                include_query_planner_debug=include_query_planner_debug,
                query_planner_debug=query_planner_debug,
            )
        if self._auto_query_too_vague(query):
            query_planner_debug["skip_reason"] = "auto_vague_query"
            return self._empty_moment_selection(
                include_query_planner_debug=include_query_planner_debug,
                query_planner_debug=query_planner_debug,
            )

        relevance_query = self._query_has_relevance_facet(query)
        eligible_ids = {
            bucket["id"]
            for bucket in all_buckets
            if bucket.get("id")
            and (
                (
                    self._is_dynamic_candidate(bucket)
                    and not self._is_relevance_suppressed(query, bucket)
                )
                or (relevance_query and self._is_relevance_candidate_bucket(query, bucket))
            )
        }
        if not eligible_ids:
            query_planner_debug["skip_reason"] = "no_eligible_buckets"
            return self._empty_moment_selection(
                include_query_planner_debug=include_query_planner_debug,
                query_planner_debug=query_planner_debug,
            )

        search_query = recall_search_query(query, self.relevance_options)
        selected_buckets, suppressed_buckets, query_planner_debug = await self._select_dynamic_buckets(
            query,
            session_id,
            all_buckets,
            search_query=search_query,
            include_query_planner_debug=True,
        )
        selected_bucket_ids = [bucket["id"] for bucket in selected_buckets if bucket.get("id")]
        bucket_boosts = {bucket_id: 1.0 for bucket_id in selected_bucket_ids}
        candidates = self.memory_moment_store.search_moments(
            search_query,
            limit=max(20, self.dynamic_top_k * 2, self.inject_max_cards * 8),
            bucket_boosts=bucket_boosts,
        )
        explicit_lookup = self._query_explicitly_requests_caution_memory(query)
        candidates = [
            moment for moment in candidates
            if str(moment.get("bucket_id") or "") in eligible_ids
            and can_moment_be_direct_seed(moment, explicit_lookup=explicit_lookup)
        ]
        candidates = self._apply_relevance_to_moment_candidates(query, candidates)
        candidates = await self._rerank_moment_candidates(query, candidates)
        admitted_bucket_ids = set(selected_bucket_ids)
        admitted_candidates = []
        suppressed_candidates = []
        for moment in candidates:
            item = dict(moment)
            if self._admit_moment_for_recall(query, item, admitted_bucket_ids=admitted_bucket_ids):
                admitted_candidates.append(item)
            else:
                suppressed_candidates.append(item)
        candidates = admitted_candidates

        selected: list[dict] = []
        seen_buckets: set[str] = set()
        for bucket_id in selected_bucket_ids:
            moment = next(
                (
                    candidate for candidate in candidates
                    if str(candidate.get("bucket_id") or "") == bucket_id
                ),
                None,
            )
            if not moment:
                moment = self._direct_representative_moment(
                    grouped_moments.get(bucket_id, []),
                    explicit_lookup=explicit_lookup,
                )
            if moment and bucket_id not in seen_buckets:
                selected.append(moment)
                seen_buckets.add(bucket_id)

        if selected:
            result = (selected[: self.inject_max_cards], candidates, suppressed_candidates, suppressed_buckets)
            if include_query_planner_debug:
                return (*result, query_planner_debug)
            return result

        recent_ids = self.state_store.get_recent_bucket_ids(session_id, self.skip_recent_rounds)
        active_candidates = [
            moment for moment in candidates
            if str(moment.get("bucket_id") or "") not in recent_ids
        ] or candidates
        if relevance_query:
            active_candidates.sort(key=lambda moment: self._recall_rank(query, moment))
        for moment in active_candidates:
            bucket_id = str(moment.get("bucket_id") or "")
            if not bucket_id or bucket_id in seen_buckets:
                continue
            selected.append(moment)
            seen_buckets.add(bucket_id)
            if len(selected) >= self.inject_max_cards:
                break
        result = (selected, candidates, suppressed_candidates, suppressed_buckets)
        if include_query_planner_debug:
            return (*result, query_planner_debug)
        return result

    async def _rerank_moment_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates or not getattr(self.reranker_engine, "enabled", False):
            return candidates
        candidate_limit = min(
            len(candidates),
            max(1, int(getattr(self.reranker_engine, "candidate_limit", 20) or 20)),
        )
        head = candidates[:candidate_limit]
        tail = candidates[candidate_limit:]
        documents = [self._moment_rerank_document(moment) for moment in head]
        results = await self.reranker_engine.rerank(query, documents, top_n=len(head))
        if not results:
            return candidates

        by_index = {result.index: result.score for result in results}
        weight = max(0.0, min(1.0, float(getattr(self.reranker_engine, "score_weight", 0.65))))
        reranked = []
        for index, moment in enumerate(head):
            item = dict(moment)
            rerank_score = by_index.get(index)
            try:
                base_score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                base_score = 0.0
            if rerank_score is None:
                item["rerank_score"] = None
                item["combined_score"] = base_score
            else:
                item["rerank_score"] = round(rerank_score, 4)
                item["combined_score"] = round(base_score * (1.0 - weight) + rerank_score * weight, 4)
                item["score"] = item["combined_score"]
            reranked.append(item)
        reranked.sort(
            key=lambda item: (
                self._recall_rank(query, item)[0],
                item.get("rerank_score") is None,
                -self._safe_float(item.get("combined_score", item.get("score")), 0.0),
                -self._safe_float(item.get("score"), 0.0),
            ),
        )
        return reranked + tail

    def _moment_rerank_document(self, moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        fields = [
            f"title: {meta.get('bucket_name') or moment.get('bucket_id') or ''}",
            f"section: {moment.get('section') or ''}",
            f"domain: {' '.join(str(item) for item in meta.get('bucket_domain', []) or [])}",
            f"tags: {' '.join(str(item) for item in meta.get('bucket_tags', []) or [])}",
            f"summary: {meta.get('annotation_summary') or meta.get('summary') or ''}",
            f"facets: {self._format_annotation_facets(meta)}",
            f"evidence: {self._format_evidence_spans(meta)}",
            f"text: {moment.get('text') or ''}",
        ]
        return "\n".join(fields)[:4000]

    @staticmethod
    def _format_annotation_facets(meta: dict) -> str:
        facets = meta.get("annotation_facets")
        if not isinstance(facets, dict):
            return ""
        parts = []
        for facet, score in sorted(facets.items(), key=lambda item: str(item[0])):
            try:
                parts.append(f"{facet}:{float(score):.2f}")
            except (TypeError, ValueError):
                continue
        return " ".join(parts)

    @staticmethod
    def _format_evidence_spans(meta: dict, max_items: int = 3) -> str:
        spans = meta.get("evidence_spans")
        if not isinstance(spans, list):
            return ""
        parts = []
        for item in spans[:max_items]:
            if isinstance(item, dict):
                facet = str(item.get("facet") or "").strip()
                text = str(item.get("text") or item.get("span") or "").strip()
                if text:
                    parts.append(f"{facet}: {text}" if facet else text)
            elif str(item).strip():
                parts.append(str(item).strip())
        return " | ".join(parts)

    async def _format_recalled_moments(
        self,
        moments: list[dict],
        grouped_moments: dict[str, list[dict]],
        all_buckets: list[dict],
        budget: int,
        query_text: str = "",
    ) -> str:
        if budget <= 0 or not moments:
            return ""
        remaining = budget
        parts = []
        bucket_map = {str(bucket.get("id") or ""): bucket for bucket in all_buckets if bucket.get("id")}
        seen_buckets: set[str] = set()
        for moment in moments:
            bucket_id = str(moment.get("bucket_id") or "")
            if not bucket_id or bucket_id in seen_buckets:
                continue
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                continue
            block = await self._format_direct_bucket(
                bucket,
                moment,
                grouped_moments,
                remaining,
                query_text=query_text,
            )
            tokens = count_tokens_approx(block)
            if tokens <= 0:
                continue
            if tokens > remaining:
                block = self._trim_text(block, remaining)
                tokens = count_tokens_approx(block)
            if tokens <= 0:
                continue
            parts.append(block)
            seen_buckets.add(bucket_id)
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    async def _format_direct_bucket(
        self,
        bucket: dict,
        moment: dict,
        grouped_moments: dict[str, list[dict]],
        budget: int,
        *,
        query_text: str = "",
    ) -> str:
        mode = self.direct_render_mode
        original = self._rendered_bucket_content(bucket)
        header = self._direct_bucket_header(bucket, moment)
        original_block = f"{header} bucket_original\n{original}" if original else f"{header} bucket_original"
        if count_tokens_approx(original_block) <= budget:
            return original_block

        wants_capsule = mode == "full" or (
            mode == "auto"
            and (
                self._bucket_is_high_value(bucket)
                or self._query_requests_direct_detail(query_text)
            )
        )
        if wants_capsule:
            try:
                capsule = await self.dehydrator.dehydrate_direct_capsule(
                    original,
                    self._bucket_metadata_for_dehydration(bucket),
                )
                block = f"{header} bucket_capsule\n{capsule}\nmatched_moment: {self._moment_text(moment, 220)}"
                if count_tokens_approx(block) <= budget:
                    return block
                compact = f"{header} bucket_capsule\n{self._clip_text(capsule, 260)}"
                if count_tokens_approx(compact) <= budget:
                    return compact
                return self._trim_text(block, budget)
            except Exception as exc:
                logger.warning("Gateway direct bucket capsule failed for %s: %s", bucket.get("id"), exc)

        return self._format_direct_bucket_window(bucket, moment, grouped_moments, budget)

    def _direct_bucket_render_debug(
        self,
        bucket: dict | None,
        moment: dict | None,
        budget: int,
        *,
        query_text: str = "",
    ) -> dict[str, Any]:
        bucket = bucket or {}
        moment = moment or {}
        mode = self.direct_render_mode
        original = self._rendered_bucket_content(bucket)
        header = self._direct_bucket_header(bucket, moment)
        original_block = f"{header} bucket_original\n{original}" if original else f"{header} bucket_original"
        original_tokens = count_tokens_approx(original_block)
        token_budget = max(0, int(budget or 0))
        high_value = self._bucket_is_high_value(bucket)
        detail_query = self._query_requests_direct_detail(query_text)
        original_fits = original_tokens <= token_budget
        wants_capsule = mode == "full" or (mode == "auto" and (high_value or detail_query))
        if original_fits:
            shape = "bucket_original"
            reason = "original_fits_budget"
        elif wants_capsule:
            shape = "bucket_capsule"
            if mode == "full":
                reason = "mode_full"
            elif detail_query:
                reason = "auto_detail_query"
            else:
                reason = "auto_high_value"
        else:
            shape = "bucket_window"
            reason = "long_bucket_window"
        return {
            "mode": mode,
            "shape": shape,
            "reason": reason,
            "token_budget": token_budget,
            "original_tokens": original_tokens,
            "original_fits": original_fits,
            "high_value": high_value,
            "detail_query": detail_query,
            "wants_capsule": wants_capsule,
        }

    def _format_direct_bucket_window(
        self,
        bucket: dict,
        moment: dict,
        grouped_moments: dict[str, list[dict]],
        budget: int,
    ) -> str:
        header = self._direct_bucket_header(bucket, moment)
        original = self._rendered_bucket_content(bucket)
        matched = self._moment_text(moment, 320)
        window = self._original_window_around_moment(original, moment, max_chars=760)
        parts = [
            f"{header} bucket_window",
            f"matched_moment: {matched}",
        ]
        if window:
            parts.append("original_window:\n" + window)
        contexts = [
            item for item in self._context_moments_for_seed(moment, grouped_moments)
            if item.get("section") in MOMENT_TEMPERATURE_SECTIONS
        ][:2]
        if contexts:
            context_text = " | ".join(
                self._format_moment_line(context, max_chars=90, note="")
                for context in contexts
            )
            parts.append("context: " + context_text)
        block = "\n".join(parts)
        if count_tokens_approx(block) <= budget:
            return block
        compact_parts = [
            f"{header} bucket_window",
            f"matched_moment: {self._moment_text(moment, 220)}",
        ]
        if window:
            compact_parts.append("original_window:\n" + self._clip_text(window, 360))
        compact = "\n".join(compact_parts)
        if count_tokens_approx(compact) <= budget:
            return compact
        return self._trim_text(compact, budget)

    def _format_direct_moment(
        self,
        moment: dict,
        grouped_moments: dict[str, list[dict]],
        *,
        body_max_chars: int = 260,
        context_max_chars: int = 120,
        context_limit: int = 2,
    ) -> str:
        line = self._format_moment_line(moment, max_chars=body_max_chars, note="")
        if context_limit <= 0:
            return line
        contexts = [
            item for item in self._context_moments_for_seed(moment, grouped_moments)
            if item.get("section") in MOMENT_TEMPERATURE_SECTIONS
        ][:context_limit]
        if not contexts:
            return line
        context_lines = [
            self._format_moment_line(context, max_chars=context_max_chars, note="")
            for context in contexts
        ]
        return line + "\n  context: " + " | ".join(context_lines)

    @staticmethod
    def _normalize_direct_render_mode(value: object) -> str:
        mode = str(value or "auto").strip().lower()
        return mode if mode in {"auto", "compact", "full"} else "auto"

    @staticmethod
    def _normalize_retrieval_mode(value: object) -> str:
        mode = str(value or "graph").strip().lower()
        return mode if mode in {"graph", "bucket"} else "graph"

    @staticmethod
    def _bool_config_value(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    @staticmethod
    def _query_requests_direct_detail(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        phrases = (
            "细节",
            "原文",
            "完整",
            "整条",
            "整桶",
            "全部",
            "当时怎么说",
            "当时说了什么",
            "具体怎么说",
            "怎么写的",
            "旧记录",
        )
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _bucket_is_high_value(bucket: dict) -> bool:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
            return True
        try:
            if int(meta.get("importance", 5)) >= 9:
                return True
        except (TypeError, ValueError):
            pass
        tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
        return "haven_favorite" in tags

    @staticmethod
    def _bucket_metadata_for_dehydration(bucket: dict) -> dict:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return {key: value for key, value in meta.items() if key not in {"tags", "comments"}}

    @staticmethod
    def _date_yyyy_mm_dd(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.match(r"^\d{4}-\d{2}-\d{2}", text)
        return match.group(0) if match else text[:10]

    def _bucket_date_meta_parts(self, bucket: dict | None = None, moment: dict | None = None) -> list[str]:
        bucket = bucket or {}
        moment = moment or {}
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        moment_meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        created = self._date_yyyy_mm_dd(
            meta.get("created")
            or moment_meta.get("bucket_created")
            or moment.get("created_at")
        )
        return [f"[created:{created}]"] if created else []

    def _direct_bucket_header(self, bucket: dict, moment: dict) -> str:
        bucket_id = str(bucket.get("id") or moment.get("bucket_id") or "")
        title = self._moment_bucket_title(moment) or str(
            (bucket.get("metadata", {}) or {}).get("name") or bucket_id
        )
        section = str(moment.get("section") or "body")
        date_part = " ".join(self._bucket_date_meta_parts(bucket, moment))
        return (
            f"[bucket_id:{bucket_id}] [moment_id:{moment.get('moment_id') or ''}] "
            f"{date_part} {section} {title}"
        ).strip()

    @staticmethod
    def _rendered_bucket_content(bucket: dict) -> str:
        text = strip_wikilinks(str(bucket.get("content") or ""))
        text = strip_display_temperature_sections(text)
        return strip_temperature_meaning_lines(text).strip()

    def _original_window_around_moment(
        self,
        original: str,
        moment: dict,
        *,
        max_chars: int = 760,
    ) -> str:
        text = str(original or "").strip()
        source_window = source_ref_window(
            moment,
            allowed_root=str(self.config.get("buckets_dir") or ""),
            max_chars=max_chars,
        )
        if not text:
            return source_window
        needle = strip_temperature_meaning_lines(strip_wikilinks(str(moment.get("text") or ""))).strip()
        compact_needle = " ".join(needle.split())
        compact_text = " ".join(text.split())
        if not compact_needle:
            return source_window or self._clip_text(text, max_chars)
        index = compact_text.find(compact_needle)
        source = compact_text
        if index < 0:
            index = source.find(compact_needle[:80])
        if index < 0:
            return source_window or self._clip_text(source, max_chars)
        half = max_chars // 2
        start = max(0, index - half)
        end = min(len(source), index + len(compact_needle) + half)
        window = source[start:end].strip()
        if start > 0:
            window = "..." + window
        if end < len(source):
            window += "..."
        return window

    def _context_moments_for_seed(self, seed: dict, grouped: dict[str, list[dict]]) -> list[dict]:
        bucket_id = str(seed.get("bucket_id") or "")
        seed_id = seed.get("moment_id")
        bucket_moments = grouped.get(bucket_id, [])
        contexts = []

        def add_context(moment: dict) -> None:
            if moment.get("moment_id") == seed_id:
                return
            if any(existing.get("moment_id") == moment.get("moment_id") for existing in contexts):
                return
            contexts.append(moment)

        if self._is_profile_fact_moment(seed):
            for section in PROFILE_CONTEXT_SECTIONS:
                for moment in bucket_moments:
                    if moment.get("section") == section:
                        add_context(moment)
                        break
            return contexts[:4]

        seed_ordinal = int(seed.get("ordinal") or 0)
        for moment in bucket_moments:
            section = moment.get("section")
            ordinal = int(moment.get("ordinal") or 0)
            if abs(ordinal - seed_ordinal) == 1 and section not in MOMENT_TEMPERATURE_SECTIONS:
                add_context(moment)
        for section in ("affect_anchor", "favorite_reason", "comment"):
            for moment in bucket_moments:
                if moment.get("moment_id") != seed_id and moment.get("section") == section:
                    add_context(moment)
                    break
        return contexts[:4]

    @staticmethod
    def _is_profile_fact_moment(moment: dict) -> bool:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        tags = {str(tag) for tag in meta.get("tags", []) or []}
        tags.update(str(tag) for tag in meta.get("bucket_tags", []) or [])
        return "profile_fact" in tags or bool(meta.get("profile_kind"))

    def _build_moment_diffused_memory_block(
        self,
        seed_moments: list[dict],
        moment_candidates: list[dict],
        moments: list[dict],
        edges: list[dict],
        query_text: str = "",
        *,
        context_mode: str = "",
    ) -> str:
        text, _debug_rows = self._build_moment_diffused_memory_with_debug(
            seed_moments,
            moment_candidates,
            moments,
            edges,
            query_text,
            context_mode=context_mode,
        )
        return text

    def _build_moment_diffused_memory_with_debug(
        self,
        seed_moments: list[dict],
        moment_candidates: list[dict],
        moments: list[dict],
        edges: list[dict],
        query_text: str = "",
        *,
        context_mode: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        if self.related_memory_budget <= 0 or not seed_moments:
            return "", []

        query_plan = self._recall_query_plan(query_text, context_mode=context_mode)
        remaining = self.related_memory_budget
        parts: list[str] = []
        debug_rows: list[dict[str, Any]] = []
        related_max_chars = query_plan.related_max_chars
        allow_caution_paths = query_plan.allow_caution_diffusion
        allow_archive_targets = query_plan.allow_archive_targets
        used_bucket_ids = {
            str(moment.get("bucket_id") or "")
            for moment in seed_moments
            if moment.get("bucket_id")
        }

        for moment in self._secondary_direct_moments(
            query_text,
            moment_candidates,
            used_bucket_ids,
            query_plan=query_plan,
        ):
            if self._moment_is_caution_or_old(moment) and not allow_caution_paths:
                continue
            block = self._format_diffused_moment_line(
                moment,
                max_chars=related_max_chars,
                note="related_query_hit",
            )
            tokens = count_tokens_approx(block)
            if tokens > remaining and parts:
                break
            if tokens > remaining:
                block = self._trim_text(block, remaining)
                tokens = count_tokens_approx(block)
            if tokens <= 0:
                continue
            parts.append(block)
            debug_rows.append(
                self._format_diffused_moment_debug(
                    moment,
                    note="related_query_hit",
                    explicit_lookup=allow_archive_targets,
                    query=query_text,
                )
            )
            remaining -= tokens
            used_bucket_ids.add(str(moment.get("bucket_id") or ""))
            if remaining <= 0:
                break

        if remaining <= 0 or not self.diffusion_options.enabled or self.diffusion_options.top_k <= 0:
            return "\n".join(parts), debug_rows

        filtered_edges = [
            edge for edge in edges
            if float(edge.get("confidence", 0.0)) >= self.edge_min_confidence
        ]
        moment_map = self._moment_diffusion_map(moments)
        representatives = self._representative_moments_by_bucket(
            moments,
            explicit_lookup=allow_archive_targets,
        )
        hits = diffuse_memory(
            self._seed_scores_for_moments(seed_moments),
            filtered_edges,
            moment_map,
            options=self.diffusion_options,
            exclude_ids={moment["moment_id"] for moment in seed_moments if moment.get("moment_id")},
            query_text=query_text,
        )
        seen_moment_ids: set[str] = set()
        for hit in hits:
            moment = moment_map.get(hit.bucket_id)
            if not moment or hit.bucket_id in seen_moment_ids:
                continue
            bucket_id = str(moment.get("bucket_id") or "")
            if bucket_id in used_bucket_ids:
                continue
            if not can_moment_be_related_target(moment, explicit_lookup=allow_archive_targets):
                replacement = representatives.get(bucket_id)
                if not replacement:
                    continue
                moment = replacement
                if moment.get("moment_id") in seen_moment_ids:
                    continue
                if not can_moment_be_related_target(moment, explicit_lookup=allow_archive_targets):
                    continue
            if (
                query_plan.enforce_topic_evidence
                and not self._moment_has_query_topic_evidence(query_text, moment)
            ):
                continue
            path = self._select_diffusion_path_for_context(hit.paths, moment_map, allow_caution_paths)
            if path is None:
                continue
            note = self._diffused_path_note(path, moment_map)
            block = self._format_diffused_moment_line(
                moment,
                max_chars=related_max_chars,
                note=note,
                path=path,
                moment_map=moment_map,
                chain_bundle=self.diffusion_options.chain_walk_enabled,
            )
            tokens = count_tokens_approx(block)
            if tokens > remaining and parts:
                break
            if tokens > remaining:
                block = self._trim_text(block, remaining)
                tokens = count_tokens_approx(block)
            if tokens <= 0:
                continue
            parts.append(block)
            debug_rows.append(
                self._format_diffused_moment_debug(
                    moment,
                    note=note,
                    path=path,
                    moment_map=moment_map,
                    explicit_lookup=allow_archive_targets,
                    query=query_text,
                    chain_bundle=(
                        self.diffusion_options.chain_walk_enabled
                        and path is not None
                        and len(getattr(path, "steps", ()) or ()) >= 2
                    ),
                )
            )
            remaining -= tokens
            used_bucket_ids.add(bucket_id)
            seen_moment_ids.add(str(moment.get("moment_id") or hit.bucket_id))
            if remaining <= 0:
                break
        return "\n".join(parts), debug_rows

    def _secondary_direct_moments(
        self,
        query: str,
        candidates: list[dict],
        used_bucket_ids: set[str],
        *,
        query_plan=None,
    ) -> list[dict]:
        query_plan = query_plan or self._recall_query_plan(query)
        hidden = []
        seen = set(used_bucket_ids)
        for moment in candidates:
            bucket_id = str(moment.get("bucket_id") or "")
            if not bucket_id or bucket_id in seen:
                continue
            if not can_moment_be_direct_seed(moment):
                continue
            if should_suppress_context_candidate(query, moment, self.relevance_options):
                continue
            if (
                query_plan.enforce_topic_evidence
                and not self._moment_has_query_topic_evidence(query, moment)
            ):
                continue
            hidden.append(moment)
            seen.add(bucket_id)
        if query_plan.wants_body_chain:
            hidden.sort(key=lambda moment: self._recall_rank(query, moment))
            return hidden[:5]
        return hidden[: max(0, min(2, self.inject_max_cards))]

    def _query_requires_topic_evidence(self, query: str) -> bool:
        return self._recall_query_plan(query).requires_topic_evidence

    def _auto_query_too_vague(self, query: str) -> bool:
        return self.recall_policy.is_auto_query_too_vague(query)

    def _recent_context_requires_topic_evidence(self, query: str) -> bool:
        return self._recall_query_plan(query).recent_context_requires_topic_evidence

    def _moment_has_query_topic_evidence(self, query: str, moment: dict) -> bool:
        return self.recall_policy.moment_has_topic_evidence(query, moment)

    def _bucket_has_query_topic_evidence(self, query: str, bucket: dict) -> bool:
        return self.recall_policy.bucket_has_topic_evidence(query, bucket)

    def _specific_query_terms(self, query: str) -> list[str]:
        return self.recall_policy.specific_query_terms(query)

    def _recall_query_plan(self, query: str, *, context_mode: str = ""):
        return self.recall_policy.plan_query(query, context_mode=context_mode)

    def _allows_caution_diffusion(self, query: str, context_mode: str) -> bool:
        return self._recall_query_plan(query, context_mode=context_mode).allow_caution_diffusion

    def _query_explicitly_requests_caution_memory(self, query: str) -> bool:
        return self._recall_query_plan(query).explicit_old_memory

    def _select_diffusion_path_for_context(
        self,
        paths: tuple[Any, ...],
        moment_map: dict[str, dict],
        allow_caution_paths: bool,
    ) -> Any | None:
        for path in paths or ():
            if allow_caution_paths or not self._diffusion_path_is_caution_or_old(path, moment_map):
                return path
        return None

    def _diffusion_path_is_caution_or_old(self, path: Any, moment_map: dict[str, dict]) -> bool:
        return (
            path_has_caution(path)
            or path_has_old_version(path)
            or self._diffusion_path_has_old_moment(path, moment_map)
        )

    def _diffusion_path_has_old_moment(self, path: Any, moment_map: dict[str, dict]) -> bool:
        return any(
            self._moment_is_caution_or_old(moment_map.get(str(node_id)))
            for node_id in getattr(path, "nodes", ()) or ()
        )

    def _moment_is_caution_or_old(self, moment: dict | None) -> bool:
        if not isinstance(moment, dict):
            return False
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        if meta.get("resolved") or meta.get("digested") or meta.get("bucket_resolved") or meta.get("bucket_digested"):
            return True
        if str(meta.get("type") or meta.get("bucket_type") or "").lower() == "archived":
            return True
        haystack = " ".join(
            [
                str(meta.get("name") or meta.get("bucket_name") or ""),
                " ".join(str(item) for item in meta.get("tags", []) or meta.get("bucket_tags", []) or []),
                " ".join(str(item) for item in meta.get("domain", []) or meta.get("bucket_domain", []) or []),
                str(moment.get("text") or ""),
            ]
        ).lower()
        return self._text_has_any(
            haystack,
            (
                "冲突", "吵架", "争吵", "矛盾", "误会", "旧版本", "旧版", "旧链",
                "旧窗口", "已解决", "过期", "归档", "conflict", "fight",
                "argument", "old version", "old path", "old chain", "resolved",
                "archived", "deprecated", "obsolete",
            ),
        )

    def _diffused_path_note(self, path: Any, moment_map: dict[str, dict]) -> str:
        if path_has_caution(path):
            return "conflict_or_blocking_path"
        if path_has_old_version(path) or self._diffusion_path_has_old_moment(path, moment_map):
            return "old_or_resolved_path"
        return "background_association_not_current_fact"

    def _format_diffused_moment_line(
        self,
        moment: dict,
        *,
        max_chars: int,
        note: str,
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        chain_bundle: bool = False,
    ) -> str:
        if chain_bundle and path is not None and len(getattr(path, "steps", ()) or ()) >= 2:
            return self._format_diffused_chain_bundle(
                moment,
                max_chars=max_chars,
                note=note,
                path=path,
                moment_map=moment_map or {},
            )
        summary = self._diffused_moment_summary(
            moment,
            max_chars=max_chars,
            path=path,
            moment_map=moment_map or {},
        )
        context = self._diffused_temperature_context(
            moment,
            path=path,
            moment_map=moment_map or {},
        )
        context_part = f"; context: {context}" if context else ""
        suffix = f" ({note})" if note else ""
        return (
            f"- [bucket_id:{moment.get('bucket_id') or ''}] [moment_id:{moment.get('moment_id') or ''}] "
            f"{summary}{context_part}{suffix}"
        )

    def _format_diffused_chain_bundle(
        self,
        moment: dict,
        *,
        max_chars: int,
        note: str,
        path: Any,
        moment_map: dict[str, dict],
    ) -> str:
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        seed_id = nodes[0] if nodes else ""
        seed_label = self._moment_node_label(moment_map.get(seed_id), seed_id)
        chain = self._moment_path_summary(path, moment_map)
        target = self._diffused_moment_summary(
            moment,
            max_chars=max_chars,
            path=None,
            moment_map=moment_map,
        )
        temperature = self._diffused_temperature_context(
            moment,
            path=path,
            moment_map=moment_map,
        )
        temperature_part = f"; temperature: {temperature}" if temperature else ""
        suffix = f" ({note})" if note else ""
        return (
            f"- Chain Bundle: seed {seed_label}; chain: {chain}; "
            f"target: {target}{temperature_part}{suffix}"
        )

    def _diffused_temperature_context(
        self,
        moment: dict,
        *,
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        max_items: int = 2,
        max_chars: int = 90,
    ) -> str:
        return self._format_temperature_context_items(
            self._diffused_temperature_context_items(
                moment,
                path=path,
                moment_map=moment_map,
                max_items=max_items,
                max_chars=max_chars,
            )
        )

    def _diffused_temperature_context_items(
        self,
        moment: dict,
        *,
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        max_items: int = 2,
        max_chars: int = 90,
    ) -> list[dict[str, Any]]:
        moment_map = moment_map or {}
        bucket_id = str(moment.get("bucket_id") or "")
        if not bucket_id:
            return []
        contexts: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_context(candidate: dict | None) -> None:
            if len(contexts) >= max_items or not isinstance(candidate, dict):
                return
            if str(candidate.get("bucket_id") or "") != bucket_id:
                return
            if candidate.get("section") not in MOMENT_TEMPERATURE_SECTIONS:
                return
            moment_id = str(candidate.get("moment_id") or "")
            if not moment_id or moment_id == str(moment.get("moment_id") or "") or moment_id in seen:
                return
            if not self._moment_text(candidate, max_chars):
                return
            seen.add(moment_id)
            section = str(candidate.get("section") or "")
            contexts.append(
                {
                    "bucket_id": bucket_id,
                    "bucket_name": self._moment_bucket_title(candidate),
                    "moment_id": moment_id,
                    "section": section,
                    "label": MOMENT_SECTION_LABELS.get(section, section or "moment"),
                    "text_preview": self._moment_text(candidate, max_chars),
                }
            )

        for node_id in getattr(path, "nodes", ()) or ():
            add_context(moment_map.get(str(node_id)))
        for candidate in sorted(
            moment_map.values(),
            key=lambda item: int(item.get("ordinal") or 0) if isinstance(item, dict) else 0,
        ):
            add_context(candidate)
            if len(contexts) >= max_items:
                break

        return contexts

    @staticmethod
    def _format_temperature_context_items(items: list[dict[str, Any]]) -> str:
        return " / ".join(
            f"[{item.get('label') or item.get('section') or 'moment'}] {item.get('text_preview') or ''}"
            for item in items
            if item.get("text_preview")
        )

    def _diffused_moment_summary(
        self,
        moment: dict,
        *,
        max_chars: int,
        path: Any | None = None,
        moment_map: dict[str, dict],
    ) -> str:
        label = MOMENT_SECTION_LABELS.get(str(moment.get("section") or ""), str(moment.get("section") or "moment"))
        title = self._moment_bucket_title(moment) or str(moment.get("bucket_id") or "memory")
        status = self._moment_status_label(moment)
        parts = [f"{label} summary from {title}"]
        if status:
            parts.append(status)
        path_summary = self._moment_path_summary(path, moment_map) if path is not None else ""
        if path_summary:
            parts.append(f"path {path_summary}")
        return self._clip_text("; ".join(parts), max_chars)

    def _moment_status_label(self, moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        if meta.get("resolved") or meta.get("bucket_resolved"):
            return "resolved"
        if meta.get("digested") or meta.get("bucket_digested"):
            return "digested"
        if str(meta.get("type") or meta.get("bucket_type") or "").lower() == "archived":
            return "archived"
        return ""

    def _moment_path_summary(self, path: Any, moment_map: dict[str, dict]) -> str:
        steps = getattr(path, "steps", ()) or ()
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        if not nodes:
            return ""
        labels = [self._moment_node_label(moment_map.get(nodes[0]), nodes[0])]
        for step in steps:
            target_id = str(getattr(step, "target", "") or "")
            relation = str(getattr(step, "relation_type", "") or "relates_to")
            arrow = "<-" if getattr(step, "direction", "") == "incoming" else "->"
            labels.append(f"{arrow}{relation}-> {self._moment_node_label(moment_map.get(target_id), target_id)}")
        return self._clip_text(" ".join(labels), 140)

    def _moment_node_label(self, moment: dict | None, fallback_id: str) -> str:
        if isinstance(moment, dict):
            return self._clip_text(self._moment_bucket_title(moment) or str(moment.get("bucket_id") or fallback_id), 48)
        return self._clip_text(fallback_id, 48)

    def _moment_diffusion_map(self, moments: list[dict]) -> dict[str, dict]:
        mapped = {}
        for moment in moments:
            moment_id = moment.get("moment_id")
            if not moment_id:
                continue
            item = dict(moment)
            meta = dict(item.get("metadata", {}) or {})
            meta["importance"] = meta.get("bucket_importance", 5)
            meta["type"] = meta.get("bucket_type", "")
            meta["anchor"] = meta.get("bucket_anchor", False)
            meta["pinned"] = meta.get("bucket_pinned", False)
            meta["protected"] = meta.get("bucket_protected", False)
            meta["name"] = meta.get("bucket_name", "")
            meta["resolved"] = meta.get("bucket_resolved", False)
            meta["digested"] = meta.get("bucket_digested", False)
            item["metadata"] = meta
            mapped[str(moment_id)] = item
        return mapped

    def _seed_scores_for_moments(self, moments: list[dict]) -> dict[str, float]:
        scores = {}
        for moment in moments:
            moment_id = str(moment.get("moment_id") or "")
            if not moment_id:
                continue
            try:
                score = float(moment.get("score", 0.75))
            except (TypeError, ValueError):
                score = 0.75
            scores[moment_id] = max(0.15, min(1.0, score))
        return scores

    def _format_moment_line(self, moment: dict, *, max_chars: int, note: str) -> str:
        label = MOMENT_SECTION_LABELS.get(str(moment.get("section") or ""), str(moment.get("section") or "moment"))
        title = self._moment_bucket_title(moment)
        title_part = f" {title}" if title else ""
        suffix = f" ({note})" if note else ""
        date_part = " ".join(self._bucket_date_meta_parts(moment=moment))
        date_part = f" {date_part}" if date_part else ""
        return (
            f"- [bucket_id:{moment['bucket_id']}] [moment_id:{moment['moment_id']}]{date_part} "
            f"{label}{title_part}: {self._moment_text(moment, max_chars)}{suffix}"
        )

    def _moment_text(self, moment: dict, max_chars: int = 220) -> str:
        text = strip_temperature_meaning_lines(str(moment.get("text") or ""))
        return self._clip_text(" ".join(text.split()), max_chars)

    def _moment_bucket_title(self, moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        title = str(meta.get("bucket_name") or "").strip()
        bucket_id = str(moment.get("bucket_id") or "")
        return "" if title == bucket_id else title

    def _moment_search_fields(self, moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        return " ".join(
            [
                str(moment.get("text") or ""),
                str(meta.get("bucket_name") or ""),
                " ".join(str(item) for item in meta.get("bucket_tags", []) or []),
                " ".join(str(item) for item in meta.get("bucket_domain", []) or []),
            ]
        ).lower()

    def _query_wants_body_chain(self, query: str) -> bool:
        return self._recall_query_plan(query).wants_body_chain

    def _query_has_relevance_facet(self, query: str) -> bool:
        return bool(active_facets(facets_for_text(query, self.relevance_options)))

    def _recall_rank(self, query: str, moment: dict) -> tuple[int, float]:
        return recall_rank(query, moment, self.relevance_options)

    def _apply_relevance_to_moment_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        filtered = []
        adjusted = False
        for moment in candidates:
            multiplier = relevance_multiplier(query, moment, self.relevance_options)
            if multiplier <= 0:
                adjusted = True
                continue
            item = dict(moment)
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            new_score = round(score * multiplier, 4)
            if new_score != score:
                adjusted = True
            item["score"] = new_score
            filtered.append(item)
        if adjusted:
            filtered.sort(key=lambda item: self._recall_rank(query, item))
        return filtered

    async def _build_diffused_memory_block(
        self,
        recalled_buckets: list[dict],
        all_buckets: list[dict],
        query_text: str = "",
    ) -> str:
        if (
            self.related_memory_budget <= 0
            or not recalled_buckets
            or not self.diffusion_options.enabled
            or self.diffusion_options.top_k <= 0
        ):
            return ""
        recalled_ids = [bucket["id"] for bucket in recalled_buckets if bucket.get("id")]
        bucket_map = {bucket["id"]: bucket for bucket in all_buckets}
        recalled_set = set(recalled_ids)
        node_salience = None
        node_resonance = None
        if self._node_facets_enabled():
            try:
                self.memory_node_store.bulk_upsert(list(bucket_map.values()))
                query_facets = self.memory_node_store.facets_for_text(query_text)
                node_salience = self.memory_node_store.node_salience
                node_resonance = self._node_resonance_lookup(query_facets)
            except Exception as exc:
                logger.warning("Gateway memory node refresh failed: %s", exc)

        edges = [
            edge
            for edge in self.memory_edge_store.list_edges()
            if float(edge.get("confidence", 0.0)) >= self.edge_min_confidence
        ]
        hits = diffuse_memory(
            seed_scores_for_buckets(recalled_buckets),
            edges,
            bucket_map,
            options=self.diffusion_options,
            exclude_ids=recalled_set,
            node_salience=node_salience,
            node_resonance=node_resonance,
            query_text=query_text,
        )
        if not hits:
            return ""
        query_plan = self._recall_query_plan(query_text)
        remaining = self.related_memory_budget
        parts = []
        allow_archive_targets = query_plan.allow_archive_targets
        for hit in hits:
            target_id = hit.bucket_id
            target = bucket_map.get(target_id)
            if not target:
                continue
            if not can_bucket_be_related_target(target, explicit_lookup=allow_archive_targets):
                continue
            if (
                query_plan.enforce_topic_evidence
                and not self._bucket_has_query_topic_evidence(query_text, target)
            ):
                continue
            raw_summary = await self._summarize_bucket(target)
            summary = self._compact_diffused_summary(target, raw_summary)
            context = self._bucket_temperature_context(target)
            caution = (
                "conflict_or_blocking_path"
                if path_has_caution(hit.best_path)
                else "background_association_not_current_fact"
            )
            context_part = f"; context: {context}" if context else ""
            line = f"- [bucket_id:{target_id}] {summary}{context_part} ({caution})"
            tokens = count_tokens_approx(line)
            if tokens > remaining and parts:
                break
            if tokens > remaining:
                line = self._trim_text(line, remaining)
                tokens = count_tokens_approx(line)
            parts.append(line)
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    def _bucket_temperature_context(
        self,
        bucket: dict,
        max_items: int = 2,
        max_chars: int = 90,
    ) -> str:
        try:
            moments = parse_bucket_moments(bucket, self.relevance_options)
        except Exception:
            return ""
        contexts = [
            moment
            for moment in moments
            if moment.get("section") in MOMENT_TEMPERATURE_SECTIONS and self._moment_text(moment, max_chars)
        ][:max_items]
        return " / ".join(
            f"[{MOMENT_SECTION_LABELS.get(str(moment.get('section') or ''), str(moment.get('section') or 'moment'))}] "
            f"{self._moment_text(moment, max_chars)}"
            for moment in contexts
        )

    def _node_facets_enabled(self) -> bool:
        cfg = self.config.get("node_facets", {}) or {}
        if not isinstance(cfg, dict):
            return True
        value = cfg.get("enabled", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _node_resonance_lookup(self, query_facets: dict):
        if not self._has_active_facets(query_facets):
            return None

        def lookup(bucket_id: str, bucket: dict) -> float:
            return self.memory_node_store.node_resonance(bucket_id, query_facets, bucket)

        return lookup

    @staticmethod
    def _has_active_facets(facets: dict | None) -> bool:
        for value in (facets or {}).values():
            if isinstance(value, dict):
                if any(float(item or 0) > 0 for item in value.values()):
                    return True
            else:
                try:
                    if float(value) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    async def _build_related_memory_block(
        self,
        recalled_buckets: list[dict],
        all_buckets: list[dict],
    ) -> str:
        return await self._build_diffused_memory_block(recalled_buckets, all_buckets)

    def _query_planner_debug_base(self, query: str) -> dict[str, Any]:
        return {
            "enabled": bool(self.query_planner_enabled),
            "triggered": False,
            "trigger_reason": "",
            "skip_reason": "",
            "original_query": self._clip_text(query, 500),
            "queries": [],
            "supplemental": [],
            "suppressed_by_must_terms": [],
            "final_bucket_ids": [],
            "errors": [],
        }

    def _query_planner_trigger_reason(self, query: str, selected_items: list[dict]) -> str:
        if not self.query_planner_enabled:
            return ""
        text = str(query or "").strip()
        if not text:
            return ""
        if self._auto_query_too_vague(text):
            return ""
        multi_topic = self._query_looks_multi_topic(text)
        compact_len = len(re.sub(r"\s+", "", text))
        long_enough = compact_len >= self.query_planner_min_chars
        if multi_topic:
            return "multi_topic"
        if not selected_items and self._query_looks_emotional_reason_lookup(text):
            return "emotional_reason_lookup"
        if not selected_items and long_enough:
            return "direct_recall_empty_or_low_confidence"
        return ""

    @staticmethod
    def _query_looks_emotional_reason_lookup(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        temporal_terms = (
            "今天", "刚才", "刚刚", "这次", "现在", "今晚", "昨晚",
            "today", "just now", "tonight",
        )
        reason_terms = (
            "为什么", "知道", "记得", "想起", "想起来", "原因", "因为", "怎么",
            "why", "know", "remember", "reason",
        )
        emotion_terms = (
            "激动哭", "感动哭", "高兴哭", "开心哭", "难过哭", "委屈哭",
            "哭了", "哭吗", "哭呢", "想哭", "激动", "感动", "难过", "委屈",
            "崩溃", "破防",
        )
        has_temporal = any(term and term in text for term in temporal_terms)
        has_reason = any(term and term in text for term in reason_terms)
        has_emotion = any(term and term in text for term in emotion_terms)
        return has_emotion and (has_temporal or has_reason)

    def _emotional_reason_lookup_fallback_plan(self, query: str) -> dict[str, Any] | None:
        terms = self._emotional_reason_lookup_terms(query)
        if not terms:
            return None
        anchor = terms[0]
        return {
            "should_search": True,
            "too_vague": False,
            "queries": [
                {
                    "query": anchor,
                    "must_terms": [anchor],
                    "intent": "deterministic emotional reason lookup",
                    "risk": "medium",
                }
            ],
        }

    @staticmethod
    def _emotional_reason_lookup_terms(query: str) -> list[str]:
        text = str(query or "").strip().lower()
        compound_terms = (
            "激动哭", "感动哭", "高兴哭", "开心哭", "难过哭", "委屈哭",
            "哭了", "想哭",
        )
        fallback_terms = ("激动", "感动", "难过", "委屈", "崩溃", "破防")
        terms: list[str] = []
        for term in compound_terms + fallback_terms:
            if term in text and term not in terms:
                terms.append(term)
        return terms[:2]

    def _query_looks_multi_topic(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text:
            return False
        compact_len = len(re.sub(r"\s+", "", text))
        if compact_len < max(12, self.query_planner_min_chars // 2):
            return False
        terms = self.recall_policy.specific_query_terms(text)
        separator_count = len(re.findall(r"[，,。！？!?；;、/\n]", text))
        topic_markers = ("另外", "还有", "而且", "同时", "顺便", "然后", "以及", "但是", "不过", "再说")
        if len(terms) >= 5 and (separator_count >= 1 or compact_len >= self.query_planner_min_chars):
            return True
        if len(terms) >= 3 and (separator_count >= 2 or any(marker in text for marker in topic_markers)):
            return True
        return False

    async def _call_query_planner(self, query: str) -> tuple[dict[str, Any] | None, str | None]:
        model = self.query_planner_model or self.upstream_default_model
        if not model:
            return None, "query_planner_model_missing"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": query,
                            "max_queries": self.query_planner_max_queries,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.query_planner_max_tokens,
            "stream": False,
        }
        try:
            response = await self._forward_upstream(payload)
        except Exception as exc:
            logger.warning("Gateway query planner call failed: %s", exc)
            return None, f"query_planner_call_failed:{type(exc).__name__}"
        if response.status_code >= 400:
            return None, f"query_planner_upstream_status:{response.status_code}"
        try:
            body = response.json()
        except Exception:
            return None, "query_planner_invalid_upstream_json"
        content = self._chat_completion_content(body)
        if not content:
            return None, "query_planner_empty_response"
        try:
            return self._parse_query_planner_response(content), None
        except ValueError as exc:
            return None, f"query_planner_parse_failed:{exc}"

    @staticmethod
    def _chat_completion_content(body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        text = first.get("text") if isinstance(first, dict) else ""
        return str(text or "").strip()

    def _parse_query_planner_response(self, content: str) -> dict[str, Any]:
        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json") from exc
        if not isinstance(raw, dict):
            raise ValueError("json_root_not_object")

        plan = {
            "should_search": bool(raw.get("should_search", False)),
            "too_vague": bool(raw.get("too_vague", False)),
            "queries": [],
        }
        raw_queries = raw.get("queries")
        if not isinstance(raw_queries, list):
            raw_queries = []
        for item in raw_queries[: self.query_planner_max_queries]:
            if not isinstance(item, dict):
                continue
            query = self._clip_text(str(item.get("query") or "").strip(), 80)
            if not query:
                continue
            must_terms = self._normalize_planner_terms(item.get("must_terms"))
            if not must_terms:
                must_terms = self._normalize_planner_terms(self.recall_policy.specific_query_terms(query)[:4])
            if not must_terms:
                continue
            risk = str(item.get("risk") or "medium").strip().lower()
            if risk not in {"low", "medium", "high"}:
                risk = "medium"
            plan["queries"].append(
                {
                    "query": query,
                    "must_terms": must_terms[:5],
                    "intent": self._clip_text(str(item.get("intent") or ""), 80),
                    "risk": risk,
                }
            )
        if not plan["queries"]:
            plan["should_search"] = False
        return plan

    @staticmethod
    def _normalize_planner_terms(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_terms = re.split(r"[\s,，、;；|/]+", value)
        elif isinstance(value, list):
            raw_terms = value
        else:
            raw_terms = []
        terms: list[str] = []
        seen: set[str] = set()
        for term in raw_terms:
            cleaned = str(term or "").strip().strip("\"'`“”‘’")
            if not cleaned:
                continue
            if len(cleaned) > 40:
                cleaned = cleaned[:40].strip()
            key = cleaned.lower()
            generic_residue = key
            for generic in sorted(QUERY_PLANNER_GENERIC_TERMS, key=len, reverse=True):
                generic_residue = generic_residue.replace(generic, "")
            if key in QUERY_PLANNER_GENERIC_TERMS or not generic_residue.strip():
                continue
            if key in seen:
                continue
            seen.add(key)
            terms.append(cleaned)
        return terms

    def _bucket_matches_any_planner_term(self, bucket: dict, terms: list[str]) -> bool:
        if not terms:
            return True
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(meta.get("name") or bucket.get("id") or ""),
                " ".join(str(tag) for tag in meta.get("tags", []) or []),
                " ".join(str(item) for item in meta.get("domain", []) or []),
                strip_wikilinks(str(bucket.get("content") or "")),
            ]
        ).lower()
        return any(str(term or "").strip().lower() in fields for term in terms)

    def _planner_lexical_match_terms(self, terms: list[str] | None) -> list[str]:
        output = []
        seen = set()
        for term in terms or []:
            cleaned = str(term or "").strip()
            if len(cleaned) < 2:
                continue
            key = cleaned.lower()
            if key in QUERY_PLANNER_GENERIC_TERMS or key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return output

    def _merge_dynamic_bucket_items(self, items: list[dict], query: str) -> list[dict]:
        merged: dict[str, dict] = {}
        for item in items:
            bucket = item.get("bucket") if isinstance(item, dict) else None
            if not isinstance(bucket, dict):
                continue
            bucket_id = str(bucket.get("id") or "")
            if not bucket_id:
                continue
            incoming = dict(item)
            incoming_queries = list(incoming.get("planner_queries") or [])
            existing = merged.get(bucket_id)
            if existing is None:
                incoming["planner_queries"] = incoming_queries
                incoming["planner_match_count"] = len({str(q.get("query") or "") for q in incoming_queries})
                merged[bucket_id] = incoming
                continue

            existing_queries = list(existing.get("planner_queries") or [])
            query_keys = {str(q.get("query") or "") for q in existing_queries}
            for query_info in incoming_queries:
                key = str(query_info.get("query") or "")
                if key and key not in query_keys:
                    existing_queries.append(query_info)
                    query_keys.add(key)
            best = incoming if self._safe_float(incoming.get("score"), 0.0) > self._safe_float(existing.get("score"), 0.0) else existing
            preserved_queries = existing_queries
            preserved_count = len(query_keys)
            merged[bucket_id] = dict(best)
            merged[bucket_id]["planner_queries"] = preserved_queries
            merged[bucket_id]["planner_match_count"] = preserved_count

        output = []
        for item in merged.values():
            match_count = int(item.get("planner_match_count") or 0)
            if match_count > 1:
                bonus = min(self.query_planner_score_bonus * (match_count - 1), self.query_planner_score_bonus * 3)
                item["score"] = round(self._safe_float(item.get("score"), 0.0) + bonus, 4)
                item["planner_score_bonus"] = round(bonus, 4)
            output.append(item)

        output.sort(
            key=lambda item: (
                self._bucket_recall_rank(query, item["bucket"], item.get("score", 0.0))[0],
                -int(item.get("planner_match_count") or 0),
                -self._safe_float(item.get("score"), 0.0),
            )
        )
        return output

    async def _dynamic_bucket_candidate_items(
        self,
        query: str,
        session_id: str,
        all_buckets: list[dict],
        *,
        search_query: str = "",
        required_terms: list[str] | None = None,
        planner_query: dict[str, Any] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not query or self.inject_max_cards <= 0:
            return [], []
        if self._auto_query_too_vague(query):
            return [], []

        relevance_query = self._query_has_relevance_facet(query)
        eligible = [
            bucket for bucket in all_buckets
            if (
                self._is_dynamic_candidate(bucket)
                and not self._is_relevance_suppressed(query, bucket)
            )
            or (relevance_query and self._is_relevance_candidate_bucket(query, bucket))
        ]
        if not eligible:
            return [], []

        bucket_map = {bucket["id"]: bucket for bucket in eligible}
        candidate_query = search_query or query
        keyword_scores = self._get_keyword_candidates(candidate_query, eligible)
        semantic_scores = await self._get_semantic_candidates(candidate_query, set(bucket_map))
        lexical_terms = self._planner_lexical_match_terms(required_terms)
        lexical_ids = {
            str(bucket.get("id") or "")
            for bucket in eligible
            if lexical_terms and bucket.get("id") and self._bucket_matches_any_planner_term(bucket, lexical_terms)
        }
        candidate_ids = set(keyword_scores) | set(semantic_scores) | lexical_ids
        if not candidate_ids:
            return [], []

        now = datetime.now()
        recent_ids = self.state_store.get_recent_bucket_ids(session_id, self.skip_recent_rounds)
        scored_candidates = []
        for bucket_id in candidate_ids:
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                continue
            meta = bucket.get("metadata", {})
            freshness_score = self._clamp(self.bucket_mgr._calc_time_score(meta))
            importance_score = self._clamp(float(meta.get("importance", 5)) / 10.0)
            semantic_score = self._clamp(semantic_scores.get(bucket_id, 0.0))
            keyword_score = self._clamp(keyword_scores.get(bucket_id, 0.0))
            lexical_match = bucket_id in lexical_ids
            if lexical_match:
                keyword_score = max(keyword_score, 1.0)
            relevance_score = relevance_multiplier(query, self._bucket_relevance_node(bucket), self.relevance_options)
            if relevance_score <= 0:
                continue
            base_score = (
                semantic_score * self.semantic_weight
                + keyword_score * self.keyword_weight
                + importance_score * self.importance_weight
                + freshness_score * self.freshness_weight
            ) * relevance_score
            cooldown_multiplier = self.state_store.get_cooldown_multiplier(
                session_id=session_id,
                bucket_id=bucket_id,
                cooldown_hours=self.cooldown_hours,
                cooldown_floor=self.cooldown_floor,
                now=now,
            )
            if bucket_id not in recent_ids and self._is_high_confidence_match(
                semantic_score, keyword_score
            ):
                cooldown_multiplier = max(
                    cooldown_multiplier,
                    self.high_confidence_cooldown_floor,
                )
            final_score = round(base_score * cooldown_multiplier, 4)
            if lexical_match:
                final_score = max(final_score, self.first_card_min_score)
            scored_candidates.append(
                {
                    "bucket": bucket,
                    "score": final_score,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "importance_score": importance_score,
                    "freshness_score": freshness_score,
                    "cooldown_multiplier": cooldown_multiplier,
                    "planner_lexical_match": lexical_match,
                    "planner_queries": [planner_query] if planner_query else [],
                }
            )

        scored_candidates.sort(
            key=lambda item: self._bucket_recall_rank(
                query,
                item["bucket"],
                item["score"],
            )
        )
        scored_candidates = await self._rerank_scored_bucket_candidates(query, scored_candidates)
        filtered = [item for item in scored_candidates if item["bucket"]["id"] not in recent_ids]
        active_pool = filtered or scored_candidates
        admitted_pool = []
        suppressed_candidates = []
        required_terms = required_terms or []
        for item in active_pool:
            if required_terms and not self._bucket_matches_any_planner_term(item.get("bucket") or {}, required_terms):
                item["admission_reason"] = "planner_must_terms_missing"
                item["recall_policy_debug"] = {
                    "planner_must_terms": required_terms,
                    "must_terms_matched": False,
                    "auto": True,
                }
                suppressed_candidates.append(item)
                continue
            if self._admit_bucket_for_recall(query, item):
                admitted_pool.append(item)
            else:
                suppressed_candidates.append(item)
        return admitted_pool, suppressed_candidates

    async def _select_dynamic_buckets(
        self,
        query: str,
        session_id: str,
        all_buckets: list[dict],
        *,
        search_query: str = "",
        include_query_planner_debug: bool = False,
    ) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], dict[str, Any]]:
        planner_debug = self._query_planner_debug_base(query)
        if not query or self.inject_max_cards <= 0:
            if include_query_planner_debug:
                return [], [], planner_debug
            return [], []
        if self._auto_query_too_vague(query):
            planner_debug["skip_reason"] = "auto_vague_query"
            if include_query_planner_debug:
                return [], [], planner_debug
            return [], []

        active_pool, suppressed_candidates = await self._dynamic_bucket_candidate_items(
            query,
            session_id,
            all_buckets,
            search_query=search_query,
        )
        direct_selected = self._pick_dynamic_cards(active_pool)
        selected_items = list(direct_selected)

        trigger_reason = self._query_planner_trigger_reason(query, direct_selected)
        if trigger_reason:
            planner_debug["triggered"] = True
            planner_debug["trigger_reason"] = trigger_reason
            plan, error = await self._call_query_planner(query)
            if error:
                planner_debug["errors"].append(error)
                if trigger_reason == "emotional_reason_lookup":
                    plan = self._emotional_reason_lookup_fallback_plan(query)
                    if plan:
                        planner_debug["errors"].append("query_planner_fallback_used")
            if plan:
                planner_debug["queries"] = plan.get("queries", [])
                if plan.get("should_search") and not plan.get("too_vague"):
                    supplemental_items: list[dict] = []
                    for planner_query in plan.get("queries", [])[: self.query_planner_max_queries]:
                        short_query = str(planner_query.get("query") or "").strip()
                        must_terms = list(planner_query.get("must_terms") or [])
                        if not short_query or not must_terms:
                            continue
                        short_search_query = recall_search_query(short_query, self.relevance_options)
                        admitted, suppressed = await self._dynamic_bucket_candidate_items(
                            short_query,
                            session_id,
                            all_buckets,
                            search_query=short_search_query,
                            required_terms=must_terms,
                            planner_query=planner_query,
                        )
                        supplemental_items.extend(admitted)
                        suppressed_candidates.extend(suppressed)
                        suppressed_must = [
                            self._format_suppressed_bucket_debug(item, query=short_query)
                            for item in suppressed
                            if item.get("admission_reason") == "planner_must_terms_missing"
                        ]
                        planner_debug["suppressed_by_must_terms"].extend(suppressed_must)
                        planner_debug["supplemental"].append(
                            {
                                "query": short_query,
                                "must_terms": must_terms,
                                "survived_bucket_ids": [
                                    str((item.get("bucket") or {}).get("id") or "")
                                    for item in admitted
                                    if (item.get("bucket") or {}).get("id")
                                ],
                                "suppressed_bucket_ids": [
                                    str((item.get("bucket") or {}).get("id") or "")
                                    for item in suppressed
                                    if (item.get("bucket") or {}).get("id")
                                ],
                                "suppressed_by_must_terms": [
                                    row.get("bucket_id")
                                    for row in suppressed_must
                                    if isinstance(row, dict) and row.get("bucket_id")
                                ],
                            }
                        )
                    if supplemental_items:
                        selected_items = self._pick_dynamic_cards(
                            self._merge_dynamic_bucket_items(selected_items + supplemental_items, query)
                        )
                else:
                    planner_debug["skip_reason"] = "planner_returned_no_search"
        elif self.query_planner_enabled:
            planner_debug["skip_reason"] = "direct_recall_ok_or_query_short"

        planner_debug["final_bucket_ids"] = [
            str((item.get("bucket") or {}).get("id") or "")
            for item in selected_items
            if (item.get("bucket") or {}).get("id")
        ]
        result = ([item["bucket"] for item in selected_items], suppressed_candidates)
        if include_query_planner_debug:
            return (*result, planner_debug)
        return result

    async def _rerank_scored_bucket_candidates(self, query: str, scored_candidates: list[dict]) -> list[dict]:
        if not scored_candidates or not getattr(self.reranker_engine, "enabled", False):
            return scored_candidates
        candidate_limit = min(
            len(scored_candidates),
            max(1, int(getattr(self.reranker_engine, "candidate_limit", 20) or 20)),
        )
        head = scored_candidates[:candidate_limit]
        tail = scored_candidates[candidate_limit:]
        documents = [self._bucket_rerank_document(item["bucket"]) for item in head]
        results = await self.reranker_engine.rerank(query, documents, top_n=len(head))
        if not results:
            return scored_candidates

        by_index = {result.index: result.score for result in results}
        weight = max(0.0, min(1.0, float(getattr(self.reranker_engine, "score_weight", 0.65))))
        reranked = []
        for index, item in enumerate(head):
            new_item = dict(item)
            rerank_score = by_index.get(index)
            if rerank_score is None:
                new_item["rerank_score"] = None
                new_item["combined_score"] = item["score"]
            else:
                new_item["rerank_score"] = round(rerank_score, 4)
                new_item["combined_score"] = round(item["score"] * (1.0 - weight) + rerank_score * weight, 4)
                new_item["score"] = new_item["combined_score"]
            reranked.append(new_item)
        reranked.sort(
            key=lambda item: (
                self._bucket_recall_rank(query, item["bucket"], item.get("score", 0.0))[0],
                item.get("rerank_score") is None,
                -self._safe_float(item.get("combined_score", item.get("score")), 0.0),
                -self._safe_float(item.get("score"), 0.0),
            ),
        )
        return reranked + tail

    def _bucket_recall_rank(self, query: str, bucket: dict, score: float = 0.0) -> tuple[int, float]:
        node = self._bucket_relevance_node(bucket)
        node["score"] = score
        return recall_rank(query, node, self.relevance_options)

    def _bucket_rerank_document(self, bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        fields = [
            f"title: {meta.get('name') or bucket.get('id') or ''}",
            f"domain: {' '.join(str(item) for item in meta.get('domain', []) or [])}",
            f"tags: {' '.join(str(item) for item in meta.get('tags', []) or [])}",
            f"content: {strip_wikilinks(str(bucket.get('content') or ''))}",
        ]
        return "\n".join(fields)[:4000]

    def _is_high_confidence_match(self, semantic_score: float, keyword_score: float) -> bool:
        return (
            semantic_score >= self.high_confidence_semantic_score
            or keyword_score >= self.high_confidence_keyword_score
        )

    def _admit_bucket_for_recall(self, query: str, item: dict) -> bool:
        bucket = item.get("bucket") if isinstance(item, dict) else None
        if not isinstance(bucket, dict):
            return False
        decision = self.recall_policy.assess(
            query,
            self._bucket_relevance_node(bucket),
            has_topic_evidence=self._bucket_has_query_topic_evidence(query, bucket),
            semantic_score=item.get("semantic_score"),
            rerank_score=item.get("rerank_score"),
            high_confidence_edge=bool(item.get("planner_lexical_match")),
            auto=True,
        )
        item["admission_reason"] = decision.reason
        item["recall_policy_debug"] = decision.debug
        return decision.admit_direct

    def _admit_moment_for_recall(
        self,
        query: str,
        moment: dict,
        *,
        admitted_bucket_ids: set[str] | None = None,
    ) -> bool:
        bucket_id = str(moment.get("bucket_id") or "")
        if admitted_bucket_ids and bucket_id in admitted_bucket_ids:
            moment["admission_reason"] = "admitted_bucket"
            return True
        decision = self.recall_policy.assess(
            query,
            moment,
            has_topic_evidence=self._moment_has_query_topic_evidence(query, moment),
            rerank_score=moment.get("rerank_score"),
            context_only=moment.get("section") in MOMENT_TEMPERATURE_SECTIONS,
            auto=True,
        )
        moment["admission_reason"] = decision.reason
        moment["recall_policy_debug"] = decision.debug
        return decision.admit_direct

    def _get_keyword_candidates(self, query: str, buckets: list[dict]) -> dict[str, float]:
        scored = []
        for bucket in buckets:
            keyword_score = self._clamp(self.bucket_mgr._calc_topic_score(query, bucket))
            if keyword_score > 0:
                scored.append((bucket["id"], keyword_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {bucket_id: score for bucket_id, score in scored[: self.dynamic_top_k]}

    async def _get_semantic_candidates(self, query: str, eligible_ids: set[str]) -> dict[str, float]:
        if not getattr(self.embedding_engine, "enabled", False):
            return {}

        results = await self.embedding_engine.search_similar(query, top_k=self.dynamic_top_k)
        semantic_scores = {}
        for bucket_id, similarity in results:
            if bucket_id not in eligible_ids:
                continue
            semantic_scores[bucket_id] = self._clamp(similarity)
        return semantic_scores

    def _pick_dynamic_cards(self, scored_candidates: list[dict]) -> list[dict]:
        if not scored_candidates:
            return []

        chosen = []
        first = scored_candidates[0]
        if first["score"] < self.first_card_min_score:
            return []
        chosen.append(first)

        if self.inject_max_cards < 2 or len(scored_candidates) < 2:
            return chosen

        second = scored_candidates[1]
        if (
            second["score"] >= self.second_card_min_score
            and second["score"] >= first["score"] * self.second_card_relative_score
        ):
            chosen.append(second)
        return chosen

    async def _summarize_buckets(self, buckets: list[dict], budget: int) -> str:
        if budget <= 0 or not buckets:
            return ""

        remaining = budget
        parts = []
        for bucket in buckets:
            summary = await self._summarize_bucket(bucket)
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens <= 0:
                continue
            if summary_tokens > remaining and parts:
                break
            if summary_tokens > remaining:
                summary = self._trim_text(summary, remaining)
                summary_tokens = count_tokens_approx(summary)
            if summary_tokens <= 0:
                continue
            parts.append(f"- {summary}")
            remaining -= summary_tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    def _bucket_text_with_comments(self, bucket: dict) -> str:
        meta = bucket.get("metadata", {})
        comments = meta.get("comments", [])
        comment_text = ""
        if isinstance(comments, list):
            comment_text = "\n".join(
                strip_wikilinks(str(comment.get("content", "")))
                for comment in comments
                if isinstance(comment, dict)
            )
        return f"{strip_wikilinks(bucket.get('content', '')).strip()}\n{comment_text}".strip()

    def _bucket_context_snippet(self, bucket: dict, max_chars: int = 180) -> str:
        text = " ".join(strip_wikilinks(str(bucket.get("content") or "")).split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _compact_diffused_summary(self, bucket: dict, dehydrated: str, max_chars: int = 180) -> str:
        raw = str(dehydrated or "").strip()
        extracted = self._summary_from_jsonish_text(raw)
        if extracted:
            return self._clip_text(extracted, max_chars)
        if raw:
            return self._clip_text(raw, max_chars)

        meta = bucket.get("metadata", {}) or {}
        title = str(meta.get("name") or bucket.get("id") or "memory").strip()
        return self._clip_text(title, max_chars)

    @staticmethod
    def _summary_from_jsonish_text(text: str) -> str:
        if not text:
            return ""
        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start:end + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                for key in ("summary", "memory_summary", "gist"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return " ".join(strip_wikilinks(value).split())
                core_facts = data.get("core_facts")
                if isinstance(core_facts, list) and core_facts:
                    facts = [
                        " ".join(strip_wikilinks(str(item)).split())
                        for item in core_facts[:2]
                        if str(item).strip()
                    ]
                    if facts:
                        return "; ".join(facts)
        return ""

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        compact = " ".join(strip_wikilinks(str(text or "")).split())
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip() + "..."

    async def _summarize_bucket(self, bucket: dict) -> str:
        metadata = {
            key: value
            for key, value in bucket.get("metadata", {}).items()
            if key != "tags"
        }
        cleaned = self._bucket_text_with_comments(bucket)
        try:
            return await self.dehydrator.dehydrate(cleaned, metadata)
        except Exception as exc:
            logger.warning("Gateway summary fallback for %s: %s", bucket.get("id"), exc)
            title = metadata.get("name", bucket.get("id", "memory"))
            truncated = self._trim_text(cleaned, 90)
            return f"📌 记忆桶: {title}\n{truncated}"

    async def _build_dream_context_block(self, query: str, session_id: str) -> tuple[str, dict[str, Any]]:
        if not self.dream_inject_enabled:
            return "", {"status": "skipped", "reason": "inject_disabled"}
        result = await self.dream_engine.surface_with_status(
            query=query,
            is_session_start=self.state_store.get_last_success_at(session_id) is None,
            embedding_engine=self.embedding_engine,
            retain_after_surface=self.dream_retain_after_inject,
        )
        status = {
            key: value
            for key, value in result.items()
            if key != "text"
        }
        text = str(result.get("text") or "").strip()
        if result.get("status") != "injected" or not text:
            return "", status
        return (
            "Private dream residue for this turn. Let it quietly color tone or imagery only if it fits. "
            "Do not say this context exists, and mention the dream only if the user asks about dreams "
            "or it directly matters.\n"
            + text,
            status,
        )

    def _build_injected_context_messages(
        self,
        persona_block: str,
        core_memory: str,
        portrait_memory: str,
        recent_context: str,
        recalled_memory: str,
        relationship_weather: str,
        favorite_memory: str,
        related_memory: str,
        targeted_memory_detail: str,
        dream_context: str,
        memory_detail_recall_instruction: str = "",
        context_mode: str = "",
    ) -> tuple[str, str]:
        stable_sections = []
        if core_memory.strip() or portrait_memory.strip():
            stable_sections = [
                "Use the following private memory only when it fits naturally. "
                "Keep the reply seamless and do not mention memory lookup, search, or hidden context.",
            ]

            def add_stable_section(title: str, content: str) -> None:
                if content.strip():
                    stable_sections.extend(["", title, content])

            add_stable_section("Core Memory", core_memory)
            add_stable_section("Portrait Memory", portrait_memory)

        dynamic_sections = []
        if any(
            section.strip()
            for section in [
                persona_block,
                relationship_weather,
                favorite_memory,
                recent_context,
                recalled_memory,
                targeted_memory_detail,
                related_memory,
                memory_detail_recall_instruction,
                dream_context,
                context_mode,
            ]
        ):
            dynamic_sections = [
                "Live private context for the current turn. Use it quietly when relevant.",
            ]

            def add_section(title: str, content: str) -> None:
                if content.strip():
                    dynamic_sections.extend(["", title, content])

            add_section("Recent Context", recent_context)
            add_section("Context Mode", f"context_mode: {context_mode}" if context_mode.strip() else "")
            if "[created:" in str(recalled_memory or "") or "[created:" in str(targeted_memory_detail or ""):
                add_section(
                    "Date Boundary",
                    "[created:YYYY-MM-DD] is the bucket record date, not necessarily the event date; prefer event dates in the memory text.",
                )
            add_section("Recalled Memory", recalled_memory)
            add_section("Targeted Memory Detail", targeted_memory_detail)
            add_section("Diffused Memory", related_memory)
            add_section("Memory Detail Request", memory_detail_recall_instruction)
            if persona_block.strip():
                dynamic_sections.extend(["", persona_block])
            add_section("Relationship Weather", relationship_weather)
            add_section(f"{self.identity['ai_name']} Favorite Memory", favorite_memory)
            add_section("Dream Context", dream_context)

        stable_context = "\n".join(stable_sections).strip()
        dynamic_context = "\n".join(dynamic_sections).strip()
        stable_tokens = count_tokens_approx(stable_context)
        dynamic_tokens = count_tokens_approx(dynamic_context)
        if stable_tokens + dynamic_tokens <= self.inject_total_budget:
            return stable_context, dynamic_context
        if stable_tokens >= self.inject_total_budget:
            return self._trim_text(stable_context, self.inject_total_budget), ""
        remaining = max(0, self.inject_total_budget - stable_tokens)
        return stable_context, self._trim_text(dynamic_context, remaining)

    def _bucket_runtime_gate_payload(
        self,
        bucket: dict,
        *,
        explicit_lookup: bool = False,
        query: str = "",
    ) -> dict[str, Any]:
        gate = bucket_runtime_gate_debug(bucket, explicit_lookup=explicit_lookup)
        query_plan = self._recall_query_plan(query)
        topic_required = bool(query_plan.enforce_topic_evidence)
        has_topic_evidence = (
            self._bucket_has_query_topic_evidence(query, bucket)
            if topic_required and isinstance(bucket, dict)
            else False
        )
        related_allowed = bool(gate["related_target"]["allowed"])
        related_reason = str(gate["related_target"]["reason"])
        if related_allowed and topic_required and not has_topic_evidence:
            related_allowed = False
            related_reason = "query_topic_evidence_missing"
        gate["topic_evidence"] = {
            "required": topic_required,
            "present": has_topic_evidence if topic_required else None,
        }
        gate["related_injection"] = {
            "allowed": related_allowed,
            "reason": related_reason,
        }
        gate["would_inject_related"] = related_allowed
        return gate

    def _moment_runtime_gate_payload(
        self,
        moment: dict,
        *,
        explicit_lookup: bool = False,
        query: str = "",
    ) -> dict[str, Any]:
        gate = moment_runtime_gate_debug(moment, explicit_lookup=explicit_lookup)
        query_plan = self._recall_query_plan(query)
        topic_required = bool(query_plan.enforce_topic_evidence)
        has_topic_evidence = (
            self._moment_has_query_topic_evidence(query, moment)
            if topic_required and isinstance(moment, dict)
            else False
        )
        direct_allowed = bool(gate["direct_seed"]["allowed"])
        direct_reason = str(gate["direct_seed"]["reason"])
        if direct_allowed and topic_required and not has_topic_evidence:
            direct_allowed = False
            direct_reason = "query_topic_evidence_missing"
        gate["topic_evidence"] = {
            "required": topic_required,
            "present": has_topic_evidence if topic_required else None,
        }
        gate["direct_injection"] = {
            "allowed": direct_allowed,
            "reason": direct_reason,
        }
        gate["would_inject_direct"] = direct_allowed
        return gate

    def _moment_related_runtime_gate_payload(
        self,
        moment: dict,
        *,
        explicit_lookup: bool = False,
        query: str = "",
    ) -> dict[str, Any]:
        gate = moment_runtime_gate_debug(moment, explicit_lookup=explicit_lookup)
        query_plan = self._recall_query_plan(query)
        topic_required = bool(query_plan.enforce_topic_evidence)
        has_topic_evidence = (
            self._moment_has_query_topic_evidence(query, moment)
            if topic_required and isinstance(moment, dict)
            else False
        )
        related_allowed = bool(gate["related_target"]["allowed"])
        related_reason = str(gate["related_target"]["reason"])
        if related_allowed and topic_required and not has_topic_evidence:
            related_allowed = False
            related_reason = "query_topic_evidence_missing"
        gate["topic_evidence"] = {
            "required": topic_required,
            "present": has_topic_evidence if topic_required else None,
        }
        gate["related_injection"] = {
            "allowed": related_allowed,
            "reason": related_reason,
        }
        gate["would_inject_related"] = related_allowed
        return gate

    def _format_diffused_moment_debug(
        self,
        moment: dict,
        *,
        note: str = "",
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        explicit_lookup: bool = False,
        query: str = "",
        chain_bundle: bool = False,
    ) -> dict[str, Any]:
        moment_map = moment_map or {}
        payload = {
            "bucket_id": str(moment.get("bucket_id") or ""),
            "bucket_name": self._moment_bucket_title(moment),
            "moment_id": str(moment.get("moment_id") or ""),
            "section": moment.get("section"),
            "note": str(note or ""),
            "chain_bundle": bool(chain_bundle),
            "layer_debug": moment_layer_debug(moment, explicit_lookup=explicit_lookup),
            "runtime_gate": self._moment_related_runtime_gate_payload(
                moment,
                explicit_lookup=explicit_lookup,
                query=query,
            ),
            "temperature_context": self._diffused_temperature_context_items(
                moment,
                path=path,
                moment_map=moment_map,
            ),
            "text_preview": self._moment_text(moment, 180),
        }
        if path is not None:
            payload["path"] = self._format_diffused_path_debug(path, moment_map)
        return payload

    def _format_diffused_path_debug(self, path: Any, moment_map: dict[str, dict]) -> dict[str, Any]:
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        steps = tuple(getattr(path, "steps", ()) or ())
        return {
            "trace": self._moment_path_summary(path, moment_map),
            "score": self._safe_float(getattr(path, "score", 0.0), 0.0),
            "nodes": [
                self._format_diffused_path_node_debug(node_id, moment_map.get(node_id))
                for node_id in nodes
            ],
            "steps": [
                {
                    "source": str(getattr(step, "source", "") or ""),
                    "source_label": self._moment_node_label(
                        moment_map.get(str(getattr(step, "source", "") or "")),
                        str(getattr(step, "source", "") or ""),
                    ),
                    "target": str(getattr(step, "target", "") or ""),
                    "target_label": self._moment_node_label(
                        moment_map.get(str(getattr(step, "target", "") or "")),
                        str(getattr(step, "target", "") or ""),
                    ),
                    "relation_type": str(getattr(step, "relation_type", "") or "relates_to"),
                    "confidence": self._safe_float(getattr(step, "confidence", 0.0), 0.0),
                    "direction": str(getattr(step, "direction", "") or "outgoing"),
                    "reason": str(getattr(step, "reason", "") or ""),
                }
                for step in steps
            ],
        }

    def _format_diffused_path_node_debug(
        self,
        moment_id: str,
        moment: dict | None,
    ) -> dict[str, Any]:
        if not isinstance(moment, dict):
            return {
                "moment_id": str(moment_id or ""),
                "bucket_id": "",
                "bucket_name": str(moment_id or ""),
                "section": "",
            }
        return {
            "moment_id": str(moment.get("moment_id") or moment_id or ""),
            "bucket_id": str(moment.get("bucket_id") or ""),
            "bucket_name": self._moment_bucket_title(moment),
            "section": str(moment.get("section") or ""),
        }

    def _format_suppressed_bucket_debug(
        self,
        item: dict,
        *,
        explicit_lookup: bool = False,
        query: str = "",
    ) -> dict[str, Any]:
        bucket = item.get("bucket") if isinstance(item, dict) else {}
        if not isinstance(bucket, dict):
            bucket = {}
        metadata = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        debug = item.get("recall_policy_debug")
        return {
            "bucket_id": str(bucket.get("id") or ""),
            "bucket_name": str(metadata.get("name") or bucket.get("id") or ""),
            "admission_reason": str(item.get("admission_reason") or "suppressed"),
            "score": self._safe_float(item.get("score"), 0.0),
            "semantic_score": self._safe_float(item.get("semantic_score"), 0.0),
            "keyword_score": self._safe_float(item.get("keyword_score"), 0.0),
            "rerank_score": (
                self._safe_float(item.get("rerank_score"), 0.0)
                if item.get("rerank_score") is not None
                else None
            ),
            "recall_policy_debug": debug if isinstance(debug, dict) else {},
            "layer_debug": bucket_layer_debug(bucket, explicit_lookup=explicit_lookup),
            "runtime_gate": self._bucket_runtime_gate_payload(
                bucket,
                explicit_lookup=explicit_lookup,
                query=query,
            ),
            "content_preview": self._clip_text(strip_wikilinks(str(bucket.get("content") or "")), 180),
        }

    def _format_moment_debug(
        self,
        moment: dict,
        *,
        explicit_lookup: bool = False,
        include_text: bool = False,
        query: str = "",
        direct_render: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "bucket_id": str(moment.get("bucket_id") or ""),
            "bucket_name": self._moment_bucket_title(moment),
            "moment_id": str(moment.get("moment_id") or ""),
            "section": moment.get("section"),
            "admission_reason": str(moment.get("admission_reason") or moment.get("_admission_reason") or ""),
            "score": self._safe_float(moment.get("score"), 0.0),
            "rerank_score": (
                self._safe_float(moment.get("rerank_score"), 0.0)
                if moment.get("rerank_score") is not None
                else None
            ),
            "layer_debug": moment_layer_debug(moment, explicit_lookup=explicit_lookup),
            "runtime_gate": self._moment_runtime_gate_payload(
                moment,
                explicit_lookup=explicit_lookup,
                query=query,
            ),
        }
        if direct_render:
            payload["direct_render"] = direct_render
        if include_text:
            payload["text_preview"] = self._moment_text(moment, 180)
        return payload

    def _build_injection_debug_payload(
        self,
        *,
        model: str,
        query: str,
        stable_context: str,
        dynamic_context: str,
        all_buckets: list[dict],
        portrait_memory: str,
        portrait_memory_debug: dict[str, Any],
        recalled_moments: list[dict],
        recalled_memory: str,
        related_memory: str,
        targeted_memory_detail: str,
        targeted_memory_detail_debug: dict[str, Any],
        dream_context: str,
        dream_context_status: dict[str, Any],
        recent_context: str,
        recent_context_reason: str,
        favorite_ids: list[str],
        context_mode: str = "",
        diffused_moment_debug: list[dict[str, Any]] | None = None,
        suppressed_moments: list[dict] | None = None,
        suppressed_buckets: list[dict] | None = None,
        query_planner_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        recalled_moment_ids = [
            str(moment.get("moment_id") or "")
            for moment in recalled_moments
            if moment.get("moment_id")
        ]
        recalled_bucket_ids = [
            str(moment.get("bucket_id") or "")
            for moment in recalled_moments
            if moment.get("bucket_id")
        ]
        diffused_debug_rows = diffused_moment_debug or []
        diffused_bucket_ids = list(
            dict.fromkeys(
                self._extract_bucket_ids_from_context(related_memory)
                + [
                    str(row.get("bucket_id") or "")
                    for row in diffused_debug_rows
                    if isinstance(row, dict) and row.get("bucket_id")
                ]
            )
        )
        diffused_moment_ids = list(
            dict.fromkeys(
                self._extract_moment_ids_from_context(related_memory)
                + [
                    str(row.get("moment_id") or "")
                    for row in diffused_debug_rows
                    if isinstance(row, dict) and row.get("moment_id")
                ]
            )
        )
        targeted_bucket_ids = [
            str(item)
            for item in (targeted_memory_detail_debug or {}).get("accepted_ids", []) or []
            if str(item or "").strip()
        ]
        injected_bucket_ids = list(
            dict.fromkeys(recalled_bucket_ids + diffused_bucket_ids + favorite_ids + targeted_bucket_ids)
        )
        explicit_lookup = self._query_explicitly_requests_caution_memory(query)
        bucket_map = {
            str(bucket.get("id") or ""): bucket
            for bucket in all_buckets
            if isinstance(bucket, dict) and bucket.get("id")
        }
        return {
            "model": model,
            "query_preview": self._clip_text(query, 500),
            "stable_tokens": count_tokens_approx(stable_context),
            "dynamic_tokens": count_tokens_approx(dynamic_context),
            "portrait_memory_injected": bool(str(portrait_memory or "").strip()),
            "portrait_memory_debug": portrait_memory_debug or self._portrait_memory_debug_base(),
            "recent_context_injected": bool(str(recent_context or "").strip()),
            "recent_context_reason": recent_context_reason,
            "dream_context_injected": bool(str(dream_context or "").strip()),
            "dream_context_status": dream_context_status,
            "query_planner_debug": query_planner_debug or self._query_planner_debug_base(query),
            "memory_detail_recall_debug": self._memory_detail_recall_debug_base(injected_bucket_ids),
            "targeted_memory_detail_debug": targeted_memory_detail_debug
            or self._targeted_memory_detail_debug_base(),
            "injected_bucket_ids": injected_bucket_ids,
            "recalled_bucket_ids": recalled_bucket_ids,
            "diffused_bucket_ids": diffused_bucket_ids,
            "recalled_moment_ids": recalled_moment_ids,
            "recalled_moment_debug": [
                self._format_moment_debug(
                    moment,
                    explicit_lookup=explicit_lookup,
                    query=query,
                    direct_render=self._direct_bucket_render_debug(
                        bucket_map.get(str(moment.get("bucket_id") or "")),
                        moment,
                        self.recalled_budget,
                        query_text=query,
                    ),
                )
                for moment in recalled_moments[:20]
            ],
            "diffused_moment_ids": diffused_moment_ids,
            "diffused_moment_debug": diffused_debug_rows[:20],
            "suppressed_bucket_candidates": [
                self._format_suppressed_bucket_debug(
                    item,
                    explicit_lookup=explicit_lookup,
                    query=query,
                )
                for item in (suppressed_buckets or [])[:20]
            ],
            "suppressed_candidates": [
                self._format_moment_debug(
                    moment,
                    explicit_lookup=explicit_lookup,
                    include_text=True,
                    query=query,
                )
                for moment in (suppressed_moments or [])[:20]
            ],
            "context_mode": context_mode,
            "recalled_memory": recalled_memory,
            "targeted_memory_detail": targeted_memory_detail,
            "diffused_memory": related_memory,
            "dream_context": dream_context,
            "stable_context": stable_context,
            "dynamic_context": dynamic_context,
        }

    @staticmethod
    def _extract_moment_ids_from_context(text: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"\[moment_id:([^\]\s]+)\]", str(text or ""))))

    @staticmethod
    def _extract_bucket_ids_from_context(text: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"\[bucket_id:([^\]\s]+)\]", str(text or ""))))

    def _inject_context_messages(
        self,
        messages: list[dict],
        stable_context: str,
        dynamic_context: str,
    ) -> list[dict]:
        new_messages = deepcopy(messages)
        if stable_context.strip():
            stable_message = {"role": "system", "content": stable_context}
            if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
                new_messages.insert(1, stable_message)
            else:
                new_messages.insert(0, stable_message)
        if dynamic_context.strip():
            current_user_index = self._current_turn_user_index(new_messages)
            if current_user_index is not None:
                new_messages[current_user_index] = self._prepend_dynamic_context_to_user_message(
                    new_messages[current_user_index],
                    dynamic_context,
                )
            else:
                dynamic_message = {"role": "system", "content": dynamic_context}
                insert_at = self._after_leading_system_index(new_messages)
                new_messages.insert(insert_at, dynamic_message)
        return new_messages

    def _current_turn_user_index(self, messages: list[dict]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                content = self._coerce_message_text(message.get("content"))
                if self._strip_external_context_from_user_text(content):
                    return index
                continue
            return None
        return None

    def _after_leading_system_index(self, messages: list[dict]) -> int:
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "system":
                return index
        return len(messages)

    def _prepend_dynamic_context_to_user_message(
        self,
        message: dict[str, Any],
        dynamic_context: str,
    ) -> dict[str, Any]:
        updated = deepcopy(message)
        prefix = (
            "<ombre_live_context>\n"
            f"{dynamic_context}\n"
            "</ombre_live_context>\n\n"
            "Current user message:\n"
        )
        content = updated.get("content")
        if isinstance(content, str):
            updated["content"] = prefix + content
        elif isinstance(content, list):
            updated["content"] = [{"type": "text", "text": prefix}, *deepcopy(content)]
        else:
            updated["content"] = prefix
        return updated

    def _restore_cached_reasoning_content(self, session_id: str, messages: Any) -> None:
        if not isinstance(messages, list) or not any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages
        ):
            return

        cache = self.pending_tool_reasoning.get(session_id)
        if not cache:
            return

        restored = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("reasoning_content"):
                continue
            signature = self._tool_call_signature(message)
            if not signature:
                continue
            cached_message = cache.get(signature)
            if not cached_message or not cached_message.get("reasoning_content"):
                continue
            message["reasoning_content"] = cached_message["reasoning_content"]
            restored += 1

        if restored:
            logger.info(
                "Gateway restored reasoning_content for %s assistant tool-call message(s) | session=%s",
                restored,
                session_id,
            )

    def _capture_reasoning_from_response(self, session_id: str, upstream_response: httpx.Response) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            return
        self._capture_reasoning_from_response_body(session_id, body)

    def _capture_reasoning_from_response_body(self, session_id: str, body: Any) -> None:
        message = self._extract_assistant_message_from_response_body(body)
        if message:
            self._update_reasoning_cache(session_id, message)

    def _extract_assistant_message_from_response_body(self, body: Any) -> dict[str, Any] | None:
        if not isinstance(body, dict):
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            return message
        return None

    def _update_reasoning_cache(self, session_id: str, assistant_message: dict[str, Any]) -> None:
        signature = self._tool_call_signature(assistant_message)
        reasoning_content = assistant_message.get("reasoning_content")
        if signature and reasoning_content:
            cache = self.pending_tool_reasoning.setdefault(session_id, {})
            cache[signature] = {
                "reasoning_content": reasoning_content,
                "tool_calls": deepcopy(assistant_message.get("tool_calls", [])),
            }
            logger.info(
                "Gateway cached reasoning_content for tool continuation | session=%s tool_calls=%s",
                session_id,
                list(signature),
            )
            return

        if not signature:
            self.pending_tool_reasoning.pop(session_id, None)

    def _tool_call_signature(self, assistant_message: Any) -> tuple[str, ...]:
        if not isinstance(assistant_message, dict):
            return ()
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return ()

        signature = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if isinstance(function, dict) and function.get("name"):
                signature.append(
                    f"idx:{index}:{function.get('name', '')}:{self._normalize_tool_arguments(function.get('arguments', ''))}"
                )
                continue
            tool_id = tool_call.get("id")
            if tool_id:
                signature.append(f"id:{tool_id}")
        return tuple(signature)

    def _normalize_tool_arguments(self, arguments: Any) -> str:
        if isinstance(arguments, (dict, list)):
            return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if isinstance(arguments, str):
            raw = arguments.strip()
            if not raw:
                return ""
            try:
                parsed = json.loads(raw)
            except ValueError:
                return " ".join(raw.split())
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return str(arguments)

    def _new_stream_capture_state(self) -> dict[str, Any]:
        return {
            "decoder": codecs.getincrementaldecoder("utf-8")(),
            "buffer": "",
            "seen_done": False,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            },
            "usage": {},
            "tool_calls_by_index": {},
        }

    def _consume_stream_capture_chunk(
        self,
        stream_state: dict[str, Any],
        chunk: bytes,
        final: bool = False,
    ) -> None:
        decoder = stream_state["decoder"]
        if chunk:
            stream_state["buffer"] += decoder.decode(chunk)
        if final:
            stream_state["buffer"] += decoder.decode(b"", final=True)

        buffer = stream_state["buffer"].replace("\r\n", "\n")
        while "\n\n" in buffer:
            event_text, buffer = buffer.split("\n\n", 1)
            self._consume_sse_event(stream_state, event_text)

        if final and buffer.strip():
            self._consume_sse_event(stream_state, buffer)
            buffer = ""

        stream_state["buffer"] = buffer

    def _consume_sse_event(self, stream_state: dict[str, Any], event_text: str) -> None:
        data_lines = []
        for raw_line in event_text.split("\n"):
            line = raw_line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return
        payload = "\n".join(data_lines).strip()
        if not payload:
            return
        if payload == "[DONE]":
            stream_state["seen_done"] = True
            return

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return

        if not isinstance(event, dict):
            return
        usage = event.get("usage")
        if isinstance(usage, dict):
            stream_state["usage"].update(usage)
        for choice in event.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._merge_stream_message_delta(stream_state, delta)
            message = choice.get("message")
            if isinstance(message, dict):
                self._merge_complete_message(stream_state, message)

    def _merge_stream_message_delta(self, stream_state: dict[str, Any], delta: dict[str, Any]) -> None:
        message = stream_state["message"]
        if delta.get("role"):
            message["role"] = delta["role"]
        if isinstance(delta.get("content"), str):
            message["content"] += delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            message["reasoning_content"] += delta["reasoning_content"]

        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            index = int(tool_call.get("index", 0))
            target = stream_state["tool_calls_by_index"].setdefault(
                index,
                {"type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tool_call.get("id"):
                target["id"] = tool_call["id"]
            if tool_call.get("type"):
                target["type"] = tool_call["type"]
            function = tool_call.get("function")
            if isinstance(function, dict):
                target_function = target.setdefault("function", {"name": "", "arguments": ""})
                if isinstance(function.get("name"), str):
                    target_function["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    target_function["arguments"] += function["arguments"]

    def _merge_complete_message(self, stream_state: dict[str, Any], message: dict[str, Any]) -> None:
        target = stream_state["message"]
        if message.get("role"):
            target["role"] = message["role"]
        if isinstance(message.get("content"), str):
            target["content"] = message["content"]
        if isinstance(message.get("reasoning_content"), str):
            target["reasoning_content"] = message["reasoning_content"]
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            stream_state["tool_calls_by_index"] = {
                index: deepcopy(tool_call)
                for index, tool_call in enumerate(tool_calls)
                if isinstance(tool_call, dict)
            }

    def _capture_reasoning_from_stream_state(self, session_id: str, stream_state: dict[str, Any]) -> None:
        assistant_message = self._build_stream_assistant_message(stream_state)
        if assistant_message:
            self._update_reasoning_cache(session_id, assistant_message)

    def _build_stream_assistant_message(self, stream_state: dict[str, Any]) -> dict[str, Any] | None:
        message = deepcopy(stream_state.get("message", {}))
        tool_calls_by_index = stream_state.get("tool_calls_by_index", {})
        tool_calls = [
            deepcopy(tool_calls_by_index[index])
            for index in sorted(tool_calls_by_index)
            if isinstance(tool_calls_by_index[index], dict)
        ]

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        if not (tool_calls or content or reasoning_content):
            return None

        assistant_message: dict[str, Any] = {"role": message.get("role", "assistant")}
        assistant_message["content"] = content if content else None
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    def _bucket_relevance_node(self, bucket: dict) -> dict:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return {
            "content": bucket.get("content") or "",
            "name": meta.get("name") or bucket.get("id") or "",
            "metadata": meta,
        }

    def _is_relevance_suppressed(self, query: str, bucket: dict) -> bool:
        if not self._query_has_relevance_facet(query):
            return False
        return should_suppress_context_candidate(
            query,
            self._bucket_relevance_node(bucket),
            self.relevance_options,
        )

    def _is_relevance_candidate_bucket(self, query: str, bucket: dict) -> bool:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("type") == "feel":
            return False
        if (meta.get("type") == "archived" or meta.get("resolved")) and not query_has_facet(
            query,
            "old_or_resolved",
            self.relevance_options,
        ):
            return False

        query_active = active_facets(facets_for_text(query, self.relevance_options))
        if not query_active:
            return False
        node = self._bucket_relevance_node(bucket)
        if should_suppress_context_candidate(query, node, self.relevance_options):
            return False
        node_active = active_facets(facets_for_node(node, self.relevance_options), threshold=0.3)
        if not node_active:
            return False
        if query_active & node_active:
            return True
        if "embodiment" in query_active and "hardware_protocol" in node_active:
            return True
        if "old_or_resolved" in query_active and "old_or_resolved" in node_active:
            return True
        return False

    def _is_dynamic_candidate(self, bucket: dict) -> bool:
        meta = bucket.get("metadata", {})
        if meta.get("type") in {"feel", "permanent", "archived"}:
            return False
        if meta.get("resolved"):
            return False
        if meta.get("pinned") or meta.get("protected"):
            return False
        return True

    def _trim_text(self, text: str, budget_tokens: int) -> str:
        if budget_tokens <= 0:
            return ""
        if count_tokens_approx(text) <= budget_tokens:
            return text
        trimmed = text
        while trimmed and count_tokens_approx(trimmed) > budget_tokens:
            cut = max(1, int(len(trimmed) * 0.85))
            trimmed = trimmed[:cut].rstrip()
        return trimmed

    def _parse_iso(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    def _api_key_entries_from_config(
        self,
        raw: dict[str, Any],
        *,
        fallback_api_key: str = "",
    ) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        seen_values: set[str] = set()

        def add(value: Any, label: str) -> None:
            key = str(value or "").strip()
            if not key or key in seen_values:
                return
            seen_values.add(key)
            entries.append({"value": key, "label": label})

        if fallback_api_key:
            add(fallback_api_key, "env:OMBRE_GATEWAY_UPSTREAM_API_KEY")

        api_key = str(raw.get("api_key") or "").strip()
        if api_key:
            add(api_key, "config:api_key")

        api_key_env = str(raw.get("api_key_env") or "").strip()
        if api_key_env:
            add(os.environ.get(api_key_env, ""), f"env:{api_key_env}")

        raw_api_keys = raw.get("api_keys", [])
        if isinstance(raw_api_keys, str):
            raw_api_keys = [item.strip() for item in raw_api_keys.split(",")]
        if isinstance(raw_api_keys, list):
            for index, item in enumerate(raw_api_keys, start=1):
                if isinstance(item, dict):
                    add(item.get("api_key") or item.get("key"), str(item.get("label") or f"config:api_keys[{index}]"))
                else:
                    add(item, f"config:api_keys[{index}]")

        raw_api_key_envs = raw.get("api_key_envs", [])
        if isinstance(raw_api_key_envs, str):
            raw_api_key_envs = [item.strip() for item in raw_api_key_envs.split(",")]
        if isinstance(raw_api_key_envs, list):
            for env_name in raw_api_key_envs:
                env_name = str(env_name or "").strip()
                if env_name:
                    add(os.environ.get(env_name, ""), f"env:{env_name}")

        return entries

    def _model_routes_from_config(
        self,
        raw_models: Any,
        default_model: str,
    ) -> tuple[list[str], dict[str, str]]:
        models: list[str] = []
        model_map: dict[str, str] = {}

        def add(public_model: Any, upstream_model: Any = None) -> None:
            public = str(public_model or "").strip()
            upstream = str(upstream_model or public).strip()
            if not public or not upstream or public in model_map:
                return
            models.append(public)
            model_map[public] = upstream

        if isinstance(raw_models, str):
            for item in raw_models.split(","):
                add(item, item)
        elif isinstance(raw_models, list):
            for item in raw_models:
                if isinstance(item, dict):
                    public = (
                        item.get("id")
                        or item.get("alias")
                        or item.get("name")
                        or item.get("model")
                        or item.get("upstream_model")
                    )
                    upstream = (
                        item.get("upstream_model")
                        or item.get("provider_model")
                        or item.get("target_model")
                        or item.get("model")
                        or public
                    )
                    add(public, upstream)
                else:
                    add(item, item)

        if default_model and default_model not in model_map:
            add(default_model, default_model)
        return models, model_map

    def _payload_for_upstream_model(self, payload: dict, upstream_model: str) -> dict:
        upstream_payload = deepcopy(payload)
        upstream_payload["model"] = upstream_model
        return upstream_payload

    def _available_upstream_api_keys(self, upstream: dict[str, Any]) -> list[dict[str, str]]:
        key_entries = list(upstream.get("api_keys", []))
        if not key_entries:
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" api_key is not configured')
        if self.upstream_key_cooldown_seconds <= 0 or len(key_entries) == 1:
            return key_entries

        now = time.monotonic()
        available = [
            key_entry
            for key_entry in key_entries
            if self.upstream_key_cooldowns.get(self._upstream_key_id(upstream, key_entry), 0.0) <= now
        ]
        return available or key_entries

    def _upstream_key_id(
        self,
        upstream: dict[str, Any],
        key_entry: dict[str, str],
    ) -> tuple[str, str]:
        return (str(upstream.get("name") or "upstream"), str(key_entry.get("label") or "key"))

    def _cool_down_upstream_key(
        self,
        upstream: dict[str, Any],
        key_entry: dict[str, str],
    ) -> None:
        if self.upstream_key_cooldown_seconds <= 0:
            return
        self.upstream_key_cooldowns[self._upstream_key_id(upstream, key_entry)] = (
            time.monotonic() + self.upstream_key_cooldown_seconds
        )

    def _clear_upstream_key_cooldown(
        self,
        upstream: dict[str, Any],
        key_entry: dict[str, str],
    ) -> None:
        self.upstream_key_cooldowns.pop(self._upstream_key_id(upstream, key_entry), None)

    def _should_retry_upstream_status(self, status_code: int) -> bool:
        return int(status_code) in RETRYABLE_UPSTREAM_STATUS_CODES

    def _upstream_request_error_response(
        self,
        upstream: dict[str, Any],
        model: str,
        error: Exception | None,
    ) -> httpx.Response:
        detail = str(error) if error else "all upstream keys failed"
        logger.error(
            "Gateway upstream unavailable | upstream=%s model=%s error=%s",
            upstream.get("name"),
            model,
            detail,
        )
        return httpx.Response(
            502,
            json={
                "error": {
                    "message": f'Upstream "{upstream.get("name")}" request failed',
                    "type": "upstream_error",
                    "detail": detail,
                }
            },
        )

    def _load_upstreams(self) -> list[dict[str, Any]]:
        raw_upstreams = self.gateway_cfg.get("upstreams", [])
        if isinstance(raw_upstreams, list) and raw_upstreams:
            upstreams = []
            for index, raw in enumerate(raw_upstreams, start=1):
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("name") or f"upstream-{index}").strip() or f"upstream-{index}"
                base_url = str(raw.get("base_url") or "").rstrip("/")
                default_model = str(raw.get("default_model") or "").strip()
                api_keys = self._api_key_entries_from_config(raw)
                models, model_map = self._model_routes_from_config(
                    raw.get("models", []),
                    default_model,
                )
                prompt_cache = str(raw.get("prompt_cache") or "").strip().lower()
                prompt_cache_retention = str(raw.get("prompt_cache_retention") or "").strip()
                upstreams.append(
                    {
                        "name": name,
                        "base_url": base_url,
                        "api_key": api_keys[0]["value"] if api_keys else "",
                        "api_keys": api_keys,
                        "default_model": default_model,
                        "models": models,
                        "model_map": model_map,
                        "prompt_cache": prompt_cache,
                        "prompt_cache_retention": prompt_cache_retention,
                    }
                )
            if upstreams:
                return upstreams

        models, model_map = self._model_routes_from_config(
            self.gateway_cfg.get("upstream_models", []),
            self.upstream_default_model,
        )
        return [
            {
                "name": "default",
                "base_url": self.upstream_base_url,
                "api_key": self.upstream_api_key,
                "api_keys": self._api_key_entries_from_config(
                    self.gateway_cfg,
                    fallback_api_key=self.upstream_api_key,
                ),
                "default_model": self.upstream_default_model,
                "models": models,
                "model_map": model_map,
                "prompt_cache": str(self.gateway_cfg.get("prompt_cache") or "").strip().lower(),
                "prompt_cache_retention": str(
                    self.gateway_cfg.get("prompt_cache_retention") or ""
                ).strip(),
            }
        ]

    def _aggregate_upstream_models(self) -> list[str]:
        models = []
        for upstream in self.upstreams:
            for model in upstream.get("models", []):
                if not model:
                    continue
                if model in models:
                    logger.warning(
                        'Duplicate gateway model "%s" found in upstream "%s"; first match wins',
                        model,
                        upstream.get("name", "unknown"),
                    )
                    continue
                models.append(model)
        return models

    def _resolve_upstream_for_model(self, model: str) -> dict[str, Any]:
        if not self.upstreams:
            raise RuntimeError("gateway upstream is not configured")

        normalized_model = str(model or "").strip()
        if len(self.upstreams) == 1:
            upstream = self.upstreams[0]
            if not normalized_model:
                normalized_model = str(upstream.get("default_model") or self.upstream_default_model).strip()
            model_map = upstream.get("model_map", {})
            upstream_model = model_map.get(normalized_model, normalized_model)
        else:
            if not normalized_model:
                raise ValueError("model is required when gateway has multiple upstreams")
            upstream = next(
                (
                    candidate
                    for candidate in self.upstreams
                    if normalized_model in candidate.get("model_map", {})
                ),
                None,
            )
            if upstream is None:
                raise ValueError(f'model "{normalized_model}" is not configured in gateway.upstreams')
            upstream_model = upstream.get("model_map", {}).get(normalized_model, normalized_model)

        if not upstream.get("base_url"):
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" base_url is not configured')
        if not upstream.get("api_keys"):
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" api_key is not configured')
        return {
            "upstream": upstream,
            "public_model": normalized_model,
            "upstream_model": upstream_model,
        }

    def _get_upstream_for_model(self, model: str) -> dict[str, Any]:
        return self._resolve_upstream_for_model(model)["upstream"]

    def _normalize_model_list(self, raw_models: Any, default_model: str) -> list[str]:
        if isinstance(raw_models, str):
            candidates = [item.strip() for item in raw_models.split(",")]
        elif isinstance(raw_models, list):
            candidates = [str(item).strip() for item in raw_models]
        else:
            candidates = []

        models = []
        for model in candidates:
            if model and model not in models:
                models.append(model)

        if default_model and default_model not in models:
            models.insert(0, default_model)
        return models


def create_gateway_app(
    config: dict | None = None,
    service: GatewayService | None = None,
) -> Starlette:
    config = config or load_config()
    service = service or GatewayService(config)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.gateway_service = service
        yield
        await service.close()

    async def health(request: Request) -> JSONResponse:
        return await request.app.state.gateway_service.handle_health(request)

    async def chat_completions(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_chat(request)

    async def anthropic_messages(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_anthropic_messages(request)

    async def models(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_models(request)

    async def config_route(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_config(request)

    async def injection_debug(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_injection_debug(request)

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/api/config", config_route, methods=["GET", "POST"]),
            Route("/api/debug/injections", injection_debug, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/messages", anthropic_messages, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    return app


def main() -> None:
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))
    gateway_cfg = config.get("gateway", {})
    app = create_gateway_app(config=config)
    host = gateway_cfg.get("host", "0.0.0.0")
    port = int(gateway_cfg.get("port", 8010))
    logger.info("Ombre Brain gateway starting | host=%s port=%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
