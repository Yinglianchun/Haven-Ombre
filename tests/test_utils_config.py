from datetime import datetime
from zoneinfo import ZoneInfo

from utils import local_date_key, load_config, parse_human_date_reference, strip_human_date_references


def test_load_config_defaults_relationship_weather_off(tmp_path):
    config = load_config(str(tmp_path / "missing-config.yaml"))

    assert config["gateway"]["relationship_weather_interval_rounds"] == 0
    assert config["gateway"]["cooldown_hours"] == 6
    assert config["gateway"]["skip_recent_rounds"] == 5
    assert config["gateway"]["semantic_session_dedupe_enabled"] is True
    assert config["gateway"]["semantic_session_dedupe_threshold"] == 0.90
    assert config["gateway"]["semantic_session_dedupe_lexical_threshold"] == 0.82
    assert config["gateway"]["portrait_memory_include_anchors"] is False
    assert config["gateway"]["operit_context_rewrite_enabled"] is False
    assert config["self_anchor"]["entry_bucket_id"] == ""
    assert config["write_path"]["semantic_search_timeout_seconds"] == 3
    assert config["memory_write_gate"]["auto_sources"] == ["operit", "workflow", "worker", "auto"]
    assert config["memory_write_gate"]["repeat_promote_count"] == 2
    assert config["raw_events"]["db_path"] == ""
    assert config["raw_events"]["max_ingest_batch"] == 1000
    assert config["word_map"]["daily_rebuild_enabled"] is True
    assert config["word_map"]["daily_rebuild_hour"] == 4
    assert config["word_map"]["daily_rebuild_minute"] == 30
    assert config["word_map"]["daily_rebuild_include_archive"] is False
    assert config["word_map"]["daily_rebuild_check_interval_minutes"] == 15
    assert config["reflection"]["enrich_backfill_enabled"] is True
    assert config["reflection"]["enrich_backfill_limit"] == 5
    assert config["reflection"]["edge_backfill_limit"] == 5
    assert config["reflection"]["daily_enabled"] is True
    assert config["reflection"]["daily_min_memory_items"] == 5
    assert config["reflection"]["daily_conversation_turn_limit"] == 0
    assert config["reflection"]["daily_chat_memory_mode"] == "auto"
    assert config["reflection"]["daily_chat_memory_hour"] == 0
    assert config["reflection"]["daily_chat_memory_turn_limit"] == 0
    assert config["reflection"]["daily_chat_memory_max_per_day"] == 3
    assert config["reflection"]["daily_chat_memory_min_confidence"] == 0.68
    assert config["reflection"]["daily_chat_memory_review_max_per_day"] == 10
    assert config["reflection"]["daily_chat_memory_review_min_confidence"] == 0.55
    assert config["reflection"]["daily_chat_memory_summary_enabled"] is True
    assert config["reflection"]["daily_chat_memory_summary_window_turns"] == 30
    assert config["reflection"]["daily_chat_memory_summary_stride_turns"] == 10
    assert config["reflection"]["daily_chat_memory_api_key_env"] == "HANDOFF_SUMMARIZER_API_KEY_2"
    assert config["reflection"]["daily_chat_memory_base_url"] == "https://api.siliconflow.cn/v1"
    assert config["reflection"]["daily_chat_memory_timeout_seconds"] == 180
    assert config["reflection"]["daily_chat_memory_summary_model"] == "Qwen/Qwen3.5-4B"
    assert config["reflection"]["daily_chat_memory_summary_max_tokens"] == 1800
    assert config["reflection"]["daily_chat_memory_candidate_model"] == "Qwen/Qwen3.5-4B"
    assert config["reflection"]["daily_chat_memory_candidate_max_tokens"] == 2400
    assert "daily_chat_memory_summary_thinking_mode" not in config["reflection"]
    assert "daily_chat_memory_candidate_thinking_mode" not in config["reflection"]
    assert "daily_chat_memory_thinking_budget" not in config["reflection"]
    assert config["reflection"]["memory_affect_anchor_enabled"] is True
    assert config["reflection"]["relationship_weather_affect_anchor_enabled"] is True
    assert config["portrait"]["enabled"] is True
    assert config["portrait"]["auto_enabled"] is True
    assert config["portrait"]["daily_enabled"] is True
    assert config["portrait"]["state_path"] == ""
    assert config["dream"]["old_echo_enabled"] is True
    assert config["dream"]["old_echo_min_age_hours"] == 72


def test_load_config_reads_runtime_config_before_env_override(tmp_path, monkeypatch):
    runtime_path = tmp_path / "state" / "config.runtime.yaml"
    runtime_path.parent.mkdir()
    runtime_path.write_text(
        "dream:\n  enabled: false\n  base_url: https://runtime.example\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMBRE_STATE_DIR", str(runtime_path.parent))
    monkeypatch.setenv("OMBRE_DREAM_BASE_URL", "https://env.example")

    config = load_config(str(tmp_path / "missing-config.yaml"))

    assert config["dream"]["enabled"] is False
    assert config["dream"]["base_url"] == "https://env.example"


def test_parse_human_date_reference_accepts_common_memory_formats():
    now = datetime(2026, 6, 15, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert parse_human_date_reference("2026.06.15", now=now)["date"] == "2026-06-15"
    assert parse_human_date_reference("2026-06-15", now=now)["date"] == "2026-06-15"
    assert parse_human_date_reference("2026/6/15", now=now)["date"] == "2026-06-15"
    assert parse_human_date_reference("2026年6月15日", now=now)["date"] == "2026-06-15"
    assert parse_human_date_reference("25年6月15日", now=now)["date"] == "2025-06-15"
    assert parse_human_date_reference("6月15日聊了什么", now=now)["date"] == "2026-06-15"
    assert local_date_key("2026.06.15") == "2026-06-15"
    assert local_date_key("2026-06-14T18:30:00+00:00") == "2026-06-15"
    assert strip_human_date_references("2026.06.15聊求职") == " 聊求职"
