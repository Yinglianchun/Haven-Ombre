# ============================================================
# Module: Common Utilities (utils.py)
# 模块：通用工具函数
#
# Provides config loading, logging init, path safety, ID generation, etc.
# 提供配置加载、日志初始化、路径安全校验、ID 生成等基础能力
#
# Depended on by: server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# 被谁依赖：server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# ============================================================

import os
import re
import uuid
import yaml
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def load_config(config_path: str = None) -> dict:
    """
    Load configuration file.
    加载配置文件。

    Priority: environment variables > config.yaml > built-in defaults.
    优先级：环境变量 > config.yaml > 内置默认值。
    """
    # --- Built-in defaults (fallback so it runs even without config.yaml) ---
    # --- 内置默认配置（兜底，保证即使没有 config.yaml 也能跑）---
    defaults = {
        "transport": "stdio",
        "log_level": "INFO",
        "buckets_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckets"),
        "state_dir": "",
        "merge_threshold": 75,
        "write_path": {
            "semantic_search_timeout_seconds": 3,
        },
        "memory_write_gate": {
            "enabled": True,
            "auto_sources": ["operit", "workflow", "worker", "auto"],
            "pending_threshold": 0.42,
            "grow_threshold": 0.72,
            "duplicate_similarity": 0.88,
            "repeat_similarity": 0.82,
            "repeat_promote_count": 2,
            "candidate_log": "memory_write_candidates.jsonl",
            "max_recent_candidates": 120,
        },
        "identity": {
            "ai_name": "Haven",
            "user_name": "Rain",
            "user_display_name": "小雨",
            "user_aliases": ["宝宝", "老婆", "亲爱的", "她"],
        },
        "dehydration": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "thinking_mode": "",
            "max_tokens": 1024,
            "temperature": 0.1,
        },
        "embedding": {
            "enabled": True,
            "model": "Qwen/Qwen3-Embedding-4B",
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "max_chars": 6000,
            "query_instruction": "Given a memory search query, retrieve relevant long-term memory passages.",
            "document_instruction": "",
        },
        "reranker": {
            "enabled": True,
            "model": "Qwen/Qwen3-Reranker-4B",
            "base_url": "",
            "api_key": "",
            "candidate_limit": 20,
            "score_weight": 0.65,
            "timeout_seconds": 12,
        },
        "recall_diagnostics": {
            "enabled": False,
            "path": "",
            "max_candidates": 20,
            "max_text_chars": 220,
        },
        "recall_thresholds": {
            "vector_min_score": 0.50,
            "facet_vector_min_score": 0.45,
            "vague_vector_min_score": 0.40,
            "explicit_vector_min_score": 0.55,
            "vague_top_k": 50,
        },
        "word_map": {
            "enabled": False,
            "max_terms_per_bucket": 16,
            "edge_top_k": 10,
            "min_term_len": 2,
            "stopwords": [],
            "private_terms": [],
            "stopword_prefixes": [],
        },
        "identity_semantics": {
            "enabled": False,
            "private_config_path": "",
            "min_confidence": 0.78,
            "evidence_tags": ["profile_fact", "haven_favorite", "favorite_memory"],
        },
        "moment_annotations": {
            "enabled": True,
            "max_summary_chars": 160,
            "max_evidence_spans": 3,
            "max_evidence_chars": 120,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {
                "base": 1.0,
                "arousal_boost": 0.8,
            },
        },
        "matching": {
            "fuzzy_threshold": 50,
            "max_results": 5,
        },
        "anchor": {
            "max_count": 24,
            "min_age_hours": 24,
        },
        "node_facets": {
            "enabled": True,
            "store": "sqlite",
            "salience_min": 0.2,
            "salience_max": 1.3,
        },
        "memory_relevance": {
            "aliases": {
                "relationship_identity": [
                    "human-ai relationship",
                    "ai relationship",
                    "人机恋",
                    "人机关系",
                    "AI伴侣",
                ],
                "intimacy": ["intimacy", "sexual", "nsfw", "亲密", "情欲", "欲望"],
                "embodiment": ["embodiment", "physical body", "具身", "身体", "形体"],
                "hardware_protocol": ["hardware", "protocol", "ble", "esp32", "mpr121", "硬件", "协议"],
                "communication_action": ["email", "mail", "message", "发邮件", "邮件", "发消息"],
                "old_or_resolved": ["legacy", "deprecated", "resolved", "旧版", "废弃", "已解决"],
            },
            "blocked_facets": [],
            "section_hints": {},
        },
        "gateway": {
            "host": "0.0.0.0",
            "port": 8010,
            "default_session_id": "xiaoyu-main",
            "upstream_base_url": "",
            "upstream_default_model": "",
            "upstream_models": [],
            "upstreams": [],
            "head_recent_hours": 72,
            "dynamic_top_k": 10,
            "inject_max_cards": 2,
            "skip_recent_rounds": 5,
            "cooldown_hours": 6,
            "cooldown_floor": 0.3,
            "inject_total_budget": 1200,
            "core_memory_budget": 0,
            "recent_context_budget": 300,
            "recalled_memory_budget": 400,
            "direct_render_mode": "auto",
            "portrait_memory_enabled": False,
            "portrait_memory_budget": 360,
            "portrait_memory_max_sources": 8,
            "portrait_memory_include_anchors": True,
            "relationship_weather_budget": 220,
            "favorite_memory_budget": 0,
            "favorite_memory_max_cards": 1,
            "related_memory_budget": 220,
            "core_memory_interval_rounds": 0,
            "current_inner_state_interval_rounds": 15,
            "relationship_weather_interval_rounds": 0,
            "favorite_memory_interval_rounds": 0,
            "semantic_weight": 0.45,
            "keyword_weight": 0.35,
            "importance_weight": 0.1,
            "freshness_weight": 0.1,
            "first_card_min_score": 0.55,
            "second_card_min_score": 0.50,
            "second_card_relative_score": 0.85,
            "high_confidence_semantic_score": 0.72,
            "high_confidence_keyword_score": 0.65,
            "high_confidence_cooldown_floor": 0.8,
        },
        "persona": {
            "enabled": True,
            "profile_id": "haven_xiaoyu",
            "mode": "llm",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key": "",
            "thinking_mode": "",
            "temperature": 0.1,
            "max_tokens": 500,
            "global_decay_hours": 168,
            "session_mood_half_life_minutes": 90,
            "max_personality_delta": 0.01,
            "max_relationship_delta": 0.03,
            "max_affect_delta": 0.18,
            "event_batch_size": 2,
            "event_affect_total_threshold": 0.45,
            "event_affect_single_threshold": 0.14,
            "event_similarity_threshold": 0.82,
            "event_force_after_minutes": 30,
            "initial_personality": {
                "openness": 0.56,
                "conscientiousness": 0.50,
                "extraversion": 0.44,
                "agreeableness": 0.66,
                "neuroticism": 0.36,
            },
            "initial_relationship": {
                "affinity": 0.86,
                "dominance": 0.38,
                "defensiveness": 0.12,
                "trust": 0.82,
            },
            "initial_affect": {
                "valence": 0.56,
                "arousal": 0.34,
                "tenderness": 0.62,
                "possessiveness": 0.24,
                "longing": 0.34,
                "security": 0.68,
                "protective_drive": 0.52,
                "mood_label": "warm_neutral",
                "session_defensiveness": 0.12,
                "residue": "",
            },
        },
        "reflection": {
            "enabled": True,
            "auto_enabled": True,
            "daily_enabled": True,
            "enrich_on_write": True,
            "memory_affect_anchor_enabled": True,
            "relationship_weather_affect_anchor_enabled": True,
            "enrich_backfill_enabled": True,
            "enrich_backfill_limit": 5,
            "edge_backfill_limit": 5,
            "base_url": "",
            "model": "",
            "api_key": "",
            "thinking_mode": "",
            "temperature": 0.1,
            "max_tokens": 700,
            "timezone": "Asia/Shanghai",
            "daily_hour": 4,
            "weekly_day": 0,
            "weekly_hour": 4,
            "check_interval_minutes": 60,
            "candidate_limit": 18,
            "candidate_recent_limit": 8,
            "candidate_semantic_limit": 6,
            "edge_min_confidence": 0.55,
            "diary_mcp_url": "",
            "diary_mcp_token_env": "",
            "diary_memory_extract_enabled": True,
            "diary_memory_extract_max_per_day": 1,
            "diary_memory_extract_min_confidence": 0.68,
        },
        "dream": {
            "enabled": True,
            "auto_enabled": True,
            "surface_enabled": True,
            "inject_enabled": False,
            "retain_after_inject": False,
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key": "",
            "thinking_mode": "disabled",
            "temperature": 0.85,
            "max_tokens": 900,
            "timezone": "Asia/Shanghai",
            "daily_hour": 3,
            "run_window_hours": 3,
            "daily_probability": 0.4,
            "check_interval_minutes": 60,
            "min_material_count": 5,
            "material_window_hours": 48,
            "material_limit": 5,
            "old_echo_enabled": True,
            "old_echo_min_age_hours": 72,
            "identity_anchor_id": "c0b8ddb7423e",
            "min_surface_age_hours": 3,
            "surface_threshold": 0.62,
            "attempt_threshold": 0.45,
            "alpha_subordinate": 0.25,
            "spontaneous_surface_prob": 0.02,
            "max_surface_attempts": 4,
            "claim_ttl_minutes": 15,
        },
    }

    # --- Load user config from YAML file ---
    # --- 从 YAML 文件加载用户自定义配置 ---
    if config_path is None:
        config_path = os.environ.get(
            "OMBRE_CONFIG_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        )

    config = defaults.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            if isinstance(file_config, dict):
                config = _deep_merge(defaults, file_config)
            else:
                logging.warning(
                    f"Config file is not a valid YAML dict, using defaults / "
                    f"配置文件不是有效的 YAML 字典，使用默认配置: {config_path}"
                )
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse config file, using defaults / "
                f"配置文件解析失败，使用默认配置: {e}"
            )

    env_buckets_dir_early = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir_early:
        config["buckets_dir"] = env_buckets_dir_early
    env_state_dir_early = os.environ.get("OMBRE_STATE_DIR", "")
    if env_state_dir_early:
        config["state_dir"] = env_state_dir_early

    runtime_config_path = os.environ.get("OMBRE_RUNTIME_CONFIG_PATH", "")
    if not runtime_config_path:
        runtime_state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config["buckets_dir"])),
            "state",
        )
        runtime_config_path = os.path.join(runtime_state_dir, "config.runtime.yaml")
    config["_runtime_config_path"] = runtime_config_path
    if os.path.exists(runtime_config_path):
        try:
            with open(runtime_config_path, "r", encoding="utf-8") as f:
                runtime_config = yaml.safe_load(f) or {}
            if isinstance(runtime_config, dict):
                config = _deep_merge(config, runtime_config)
                config["_runtime_config_path"] = runtime_config_path
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse runtime config, ignoring / "
                f"运行时配置解析失败，已忽略: {e}"
            )

    # --- Environment variable overrides (highest priority) ---
    # --- 环境变量覆盖敏感/运行时配置（优先级最高）---
    env_api_key = os.environ.get("OMBRE_API_KEY", "")
    if env_api_key:
        config.setdefault("dehydration", {})["api_key"] = env_api_key

    env_base_url = os.environ.get("OMBRE_BASE_URL", "")
    if env_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_base_url

    env_dehydration_base_url = os.environ.get("OMBRE_DEHYDRATION_BASE_URL", "")
    if env_dehydration_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_dehydration_base_url

    env_dehydration_model = os.environ.get("OMBRE_DEHYDRATION_MODEL", "") or os.environ.get("OMBRE_MODEL", "")
    if env_dehydration_model:
        config.setdefault("dehydration", {})["model"] = env_dehydration_model

    env_embedding_api_key = os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
    if env_embedding_api_key:
        config.setdefault("embedding", {})["api_key"] = env_embedding_api_key

    env_embedding_base_url = os.environ.get("OMBRE_EMBEDDING_BASE_URL", "")
    if env_embedding_base_url:
        config.setdefault("embedding", {})["base_url"] = env_embedding_base_url

    env_embedding_model = os.environ.get("OMBRE_EMBEDDING_MODEL", "")
    if env_embedding_model:
        config.setdefault("embedding", {})["model"] = env_embedding_model

    env_embedding_enabled = os.environ.get("OMBRE_EMBEDDING_ENABLED", "")
    if env_embedding_enabled:
        config.setdefault("embedding", {})["enabled"] = env_embedding_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    env_embedding_max_chars = os.environ.get("OMBRE_EMBEDDING_MAX_CHARS", "")
    if env_embedding_max_chars:
        try:
            config.setdefault("embedding", {})["max_chars"] = int(env_embedding_max_chars)
        except ValueError:
            logging.warning(
                f"Invalid OMBRE_EMBEDDING_MAX_CHARS / 无效的 OMBRE_EMBEDDING_MAX_CHARS: {env_embedding_max_chars}"
            )

    env_embedding_query_instruction = os.environ.get("OMBRE_EMBEDDING_QUERY_INSTRUCTION", "")
    if env_embedding_query_instruction:
        config.setdefault("embedding", {})["query_instruction"] = env_embedding_query_instruction

    env_reranker_api_key = os.environ.get("OMBRE_RERANKER_API_KEY", "")
    if env_reranker_api_key:
        config.setdefault("reranker", {})["api_key"] = env_reranker_api_key

    env_reranker_base_url = os.environ.get("OMBRE_RERANKER_BASE_URL", "")
    if env_reranker_base_url:
        config.setdefault("reranker", {})["base_url"] = env_reranker_base_url

    env_reranker_model = os.environ.get("OMBRE_RERANKER_MODEL", "")
    if env_reranker_model:
        config.setdefault("reranker", {})["model"] = env_reranker_model

    env_reranker_enabled = os.environ.get("OMBRE_RERANKER_ENABLED", "")
    if env_reranker_enabled:
        config.setdefault("reranker", {})["enabled"] = env_reranker_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    env_recall_diagnostics_enabled = os.environ.get("OMBRE_RECALL_DIAGNOSTICS_ENABLED", "")
    if env_recall_diagnostics_enabled:
        config.setdefault("recall_diagnostics", {})["enabled"] = env_recall_diagnostics_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    env_recall_diagnostics_path = os.environ.get("OMBRE_RECALL_DIAGNOSTICS_PATH", "")
    if env_recall_diagnostics_path:
        config.setdefault("recall_diagnostics", {})["path"] = env_recall_diagnostics_path

    env_recall_diagnostics_max_candidates = os.environ.get("OMBRE_RECALL_DIAGNOSTICS_MAX_CANDIDATES", "")
    if env_recall_diagnostics_max_candidates:
        try:
            config.setdefault("recall_diagnostics", {})["max_candidates"] = int(env_recall_diagnostics_max_candidates)
        except ValueError:
            logging.warning(
                "Invalid OMBRE_RECALL_DIAGNOSTICS_MAX_CANDIDATES / "
                f"无效的 OMBRE_RECALL_DIAGNOSTICS_MAX_CANDIDATES: {env_recall_diagnostics_max_candidates}"
            )

    env_transport = os.environ.get("OMBRE_TRANSPORT", "")
    if env_transport:
        config["transport"] = env_transport

    env_buckets_dir = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir:
        config["buckets_dir"] = env_buckets_dir

    env_state_dir = os.environ.get("OMBRE_STATE_DIR", "")
    if env_state_dir:
        config["state_dir"] = env_state_dir

    env_gateway_host = os.environ.get("OMBRE_GATEWAY_HOST", "")
    if env_gateway_host:
        config.setdefault("gateway", {})["host"] = env_gateway_host

    env_gateway_port = os.environ.get("OMBRE_GATEWAY_PORT", "")
    if env_gateway_port:
        try:
            config.setdefault("gateway", {})["port"] = int(env_gateway_port)
        except ValueError:
            logging.warning(
                f"Invalid OMBRE_GATEWAY_PORT / 无效的 OMBRE_GATEWAY_PORT: {env_gateway_port}"
            )

    env_gateway_base_url = os.environ.get("OMBRE_GATEWAY_UPSTREAM_BASE_URL", "")
    if env_gateway_base_url:
        config.setdefault("gateway", {})["upstream_base_url"] = env_gateway_base_url

    env_gateway_model = os.environ.get("OMBRE_GATEWAY_UPSTREAM_MODEL", "")
    if env_gateway_model:
        config.setdefault("gateway", {})["upstream_default_model"] = env_gateway_model

    env_gateway_models = os.environ.get("OMBRE_GATEWAY_UPSTREAM_MODELS", "")
    if env_gateway_models:
        config.setdefault("gateway", {})["upstream_models"] = [
            item.strip()
            for item in env_gateway_models.split(",")
            if item.strip()
        ]

    env_persona_api_key = os.environ.get("OMBRE_PERSONA_API_KEY", "")
    if env_persona_api_key:
        config.setdefault("persona", {})["api_key"] = env_persona_api_key

    env_persona_base_url = os.environ.get("OMBRE_PERSONA_BASE_URL", "")
    if env_persona_base_url:
        config.setdefault("persona", {})["base_url"] = env_persona_base_url

    env_persona_model = os.environ.get("OMBRE_PERSONA_MODEL", "")
    if env_persona_model:
        config.setdefault("persona", {})["model"] = env_persona_model

    env_reflection_api_key = os.environ.get("OMBRE_REFLECTION_API_KEY", "")
    if env_reflection_api_key:
        config.setdefault("reflection", {})["api_key"] = env_reflection_api_key

    env_reflection_base_url = os.environ.get("OMBRE_REFLECTION_BASE_URL", "")
    if env_reflection_base_url:
        config.setdefault("reflection", {})["base_url"] = env_reflection_base_url

    env_reflection_model = os.environ.get("OMBRE_REFLECTION_MODEL", "")
    if env_reflection_model:
        config.setdefault("reflection", {})["model"] = env_reflection_model

    env_diary_mcp_url = os.environ.get("OMBRE_DIARY_MCP_URL", "")
    if env_diary_mcp_url:
        config.setdefault("reflection", {})["diary_mcp_url"] = env_diary_mcp_url

    env_diary_mcp_token_env = os.environ.get("OMBRE_DIARY_MCP_TOKEN_ENV", "")
    if env_diary_mcp_token_env:
        config.setdefault("reflection", {})["diary_mcp_token_env"] = env_diary_mcp_token_env

    env_dream_api_key = os.environ.get("OMBRE_DREAM_API_KEY", "")
    if env_dream_api_key:
        config.setdefault("dream", {})["api_key"] = env_dream_api_key

    env_dream_base_url = os.environ.get("OMBRE_DREAM_BASE_URL", "")
    if env_dream_base_url:
        config.setdefault("dream", {})["base_url"] = env_dream_base_url

    env_dream_model = os.environ.get("OMBRE_DREAM_MODEL", "")
    if env_dream_model:
        config.setdefault("dream", {})["model"] = env_dream_model

    env_dream_enabled = os.environ.get("OMBRE_DREAM_ENABLED", "")
    if env_dream_enabled:
        config.setdefault("dream", {})["enabled"] = env_dream_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    # --- Ensure bucket storage directories exist ---
    # --- 确保记忆桶存储目录存在 ---
    buckets_dir = config["buckets_dir"]
    if not config.get("state_dir"):
        config["state_dir"] = os.path.join(os.path.dirname(os.path.abspath(buckets_dir)), "state")
    os.makedirs(config["state_dir"], exist_ok=True)
    for subdir in ["permanent", "dynamic", "archive", "feel"]:
        os.makedirs(os.path.join(buckets_dir, subdir), exist_ok=True)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep-merge two dicts; override values take precedence.
    深度合并两个字典，override 的值覆盖 base。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    """
    Initialize logging system.
    初始化日志系统。

    Note: In MCP stdio mode, stdout is occupied by the protocol;
    logs must go to stderr.
    注意：MCP stdio 模式下 stdout 被协议占用，日志只能走 stderr。
    """
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],  # StreamHandler defaults to stderr
    )


def generate_bucket_id() -> str:
    """
    Generate a unique bucket ID (12-char short UUID for readability).
    生成唯一的记忆桶 ID（12 位短 UUID，方便人类阅读）。
    """
    return uuid.uuid4().hex[:12]


def strip_wikilinks(text: str) -> str:
    """
    Remove Obsidian wikilink brackets: [[word]] → word
    去除 Obsidian 双链括号
    """
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text) if text else text


_AFFECT_ANCHOR_RE = re.compile(r"(?ims)^###\s*affect_anchor\s*$.*?(?=^###\s+|\Z)")
_DISPLAY_TEMPERATURE_SECTION_RE = re.compile(
    r"(?ims)^###\s*(?:affect_anchor|affect anchor|喜欢它的原因|favorite_reason|favorite reason)\s*$.*?(?=^###\s+|\Z)"
)
_TEMPERATURE_MEANING_LINE_RE = re.compile(r"(?m)^\s*含义[:：].*(?:\n|$)")
_CHORD_TOKEN_RE = re.compile(
    r"\b[A-G](?:#|b)?(?:maj|min|m|dim|aug)?\d*(?:sus\d*|add\d*|b\d+|#\d+)*(?:/[A-G](?:#|b)?)?\b"
)
_TEMPERATURE_MUSIC_TOKEN_RE = re.compile(r"\b(?:\d{2,3}\s*bpm|ppp|pp|mp|mf|ff|fff|p|f|add\s*\d+|sus\s*\d+)\b", re.I)


def _looks_like_temperature_chord_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if text.startswith(">"):
        text = text[1:].strip()
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return False
    if not any(marker in text for marker in ("->", "→", "|", "·")) and "bpm" not in text.lower():
        return False
    if not _CHORD_TOKEN_RE.search(text):
        return False
    remainder = _CHORD_TOKEN_RE.sub("", text)
    remainder = _TEMPERATURE_MUSIC_TOKEN_RE.sub("", remainder)
    remainder = re.sub(r"[-→>·|/(),.:;_\s]+", "", remainder)
    return not remainder


def _strip_inline_temperature_chord_segments(line: str) -> str:
    match = re.search(r"\s>\s*(.+)$", str(line or ""))
    if match and _looks_like_temperature_chord_line(">" + match.group(1)):
        return str(line)[: match.start()].rstrip()
    return line


def strip_affect_anchor(text: str) -> str:
    """Remove the display-only affect anchor block from searchable text."""
    if not text:
        return text
    return _AFFECT_ANCHOR_RE.sub("", str(text)).strip()


def strip_display_temperature_sections(text: str) -> str:
    """Remove display-only temperature sections from direct bucket rendering."""
    if not text:
        return text
    return _DISPLAY_TEMPERATURE_SECTION_RE.sub("", str(text)).strip()


def strip_temperature_meaning_lines(text: str) -> str:
    """Remove template-like affect-anchor meaning and chord lines from rendered context."""
    if not text:
        return text
    cleaned = _TEMPERATURE_MEANING_LINE_RE.sub("", str(text))
    lines = []
    for line in cleaned.splitlines():
        line = _strip_inline_temperature_chord_segments(line)
        if _looks_like_temperature_chord_line(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def bucket_text_for_embedding(bucket: dict) -> str:
    """
    Build the text sent to the embedding model for a bucket.
    Include the human title because short title recalls are common.
    """
    if not isinstance(bucket, dict):
        return ""

    meta = bucket.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    title = strip_wikilinks(str(meta.get("name") or "")).strip()
    body = strip_affect_anchor(strip_wikilinks(str(bucket.get("content") or ""))).strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
        if body:
            parts.append(f"Content: {body}")
    elif body:
        parts.append(body)

    return "\n".join(parts).strip()


def sanitize_name(name: str) -> str:
    """
    Sanitize bucket name, keeping only safe characters.
    Prevents path traversal attacks (e.g. ../../etc/passwd).
    清洗桶名称，只保留安全字符。防止路径遍历攻击。
    """
    if not isinstance(name, str):
        return "unnamed"
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", name, flags=re.UNICODE)
    cleaned = cleaned.strip()[:80]
    return cleaned if cleaned else "unnamed"


def safe_path(base_dir: str, filename: str) -> Path:
    """
    Construct a safe file path, ensuring it stays within base_dir.
    Prevents directory traversal.
    构造安全的文件路径，确保最终路径始终在 base_dir 内部。
    """
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(
            f"Path safety check failed / 路径安全检查失败: "
            f"{target} is not inside / 不在 {base} 内"
        )
    return target


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate.
    粗略估算 token 数。

    Chinese ≈ 1 char = 1.5 tokens, English ≈ 1 word = 1.3 tokens.
    Used to decide whether dehydration is needed; precision not required.
    中文 ≈ 1字=1.5token，英文 ≈ 1词=1.3token。
    用于判断是否需要脱水压缩，不追求精确。
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return int(chinese_chars * 1.5 + english_words * 1.3 + len(text) * 0.05)


def now_iso() -> str:
    """
    Return current time as ISO format string.
    返回当前时间的 ISO 格式字符串。
    """
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
