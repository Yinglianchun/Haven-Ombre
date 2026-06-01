from utils import load_config


def test_load_config_defaults_relationship_weather_off(tmp_path):
    config = load_config(str(tmp_path / "missing-config.yaml"))

    assert config["gateway"]["relationship_weather_interval_rounds"] == 0
    assert config["gateway"]["cooldown_hours"] == 6
    assert config["gateway"]["skip_recent_rounds"] == 5
    assert config["write_path"]["semantic_search_timeout_seconds"] == 3
    assert config["memory_write_gate"]["auto_sources"] == ["operit", "workflow", "worker", "auto"]
    assert config["memory_write_gate"]["repeat_promote_count"] == 2
    assert config["reflection"]["enrich_backfill_enabled"] is True
    assert config["reflection"]["enrich_backfill_limit"] == 5
    assert config["reflection"]["daily_enabled"] is True
    assert config["reflection"]["memory_affect_anchor_enabled"] is True
    assert config["reflection"]["relationship_weather_affect_anchor_enabled"] is True
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
