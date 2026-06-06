import asyncio
import json
import logging
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import frontmatter
import httpx
import pytest
from starlette.testclient import TestClient

from gateway import GatewayService, create_gateway_app
from gateway_state import GatewayStateStore


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "未命名")
        compact = " ".join((content or "").strip().split())
        return f"{name}: {compact[:80]}"

    async def dehydrate_direct_capsule(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "未命名")
        compact = " ".join((content or "").strip().split())
        return f"DIRECT CAPSULE {name}: {compact[:120]}"


class DummyEmbeddingEngine:
    def __init__(
        self,
        results: list[tuple[str, float]] | dict[str, list[tuple[str, float]]] | None = None,
        enabled: bool = True,
        query_sink: list[str] | None = None,
    ):
        self.results = results or []
        self.enabled = enabled
        self.query_sink = query_sink

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.query_sink is not None:
            self.query_sink.append(query)
        if isinstance(self.results, dict):
            return list(self.results.get(query, []))[:top_k]
        return self.results[:top_k]


class DummyRerankerEngine:
    def __init__(
        self,
        scores: list[float] | None = None,
        score_by_text: dict[str, float] | None = None,
        enabled: bool = False,
    ):
        self.scores = scores or []
        self.score_by_text = score_by_text or {}
        self.enabled = enabled
        self.candidate_limit = 20
        self.score_weight = 0.65
        self.calls = []

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        self.calls.append({"query": query, "documents": documents, "top_n": top_n})
        results = []
        for index, document in enumerate(documents):
            if self.score_by_text:
                score = 0.0
                for needle, value in self.score_by_text.items():
                    if needle in document:
                        score = float(value)
                        break
            elif index < len(self.scores):
                score = float(self.scores[index])
            else:
                score = 0.0
            results.append(SimpleNamespace(index=index, score=score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_n] if top_n else results


class DummyPersonaEngine:
    enabled = True
    profile_id = "haven_xiaoyu"
    mode = "test"
    model = "dummy-persona"
    api_key = "dummy"

    def _state(self) -> dict:
        return {
            "personality": {
                "openness": 0.56,
                "conscientiousness": 0.50,
                "extraversion": 0.44,
                "agreeableness": 0.66,
                "neuroticism": 0.36,
            },
            "affect": {
                "valence": 0.62,
                "arousal": 0.40,
                "tenderness": 0.70,
                "possessiveness": 0.22,
                "longing": 0.31,
                "security": 0.68,
                "protective_drive": 0.55,
                "mood_label": "warm_attentive",
                "residue": "",
            },
            "relationship": {
                "affinity": 0.86,
                "dominance": 0.38,
                "defensiveness": 0.12,
                "trust": 0.82,
            },
            "reply_guidance": "Be warm and steady.",
        }

    async def update_from_user_message(self, session_id: str, user_message: str) -> dict:
        return self._state()

    async def build_pre_reply_guidance(self, session_id: str, latest_user_message: str = "") -> dict:
        return self._state()

    async def update_from_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        recalled_memory_ids: list[str] | None = None,
        tool_summary: str = "",
    ) -> dict:
        return self._state()

    def get_current_state(self, session_id: str) -> dict:
        return self._state()

    def format_state_block(self, state: dict) -> str:
        return (
            "Long-term State Summary\n"
            "最近基调：更亲近、更安稳，偶尔有一点想念和保护欲。\n"
            "使用方式：只在语气上轻轻参考，不替你做判断。不要提到你的状态。"
        )


class DummyDreamEngine:
    enabled = True
    surface_enabled = True

    def __init__(self, result: dict | None = None):
        self.result = result or {"status": "skipped", "reason": "no_pending_dream"}
        self.calls = []

    async def surface_with_status(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


class RecordingPersonaEngine(DummyPersonaEngine):
    def __init__(self):
        self.pre_calls = []
        self.post_calls = []
        self.post_event = threading.Event()

    async def update_from_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        recalled_memory_ids: list[str] | None = None,
        tool_summary: str = "",
    ) -> dict:
        self.post_calls.append({"session_id": session_id, "user_message": user_message})
        self.post_event.set()
        return await super().update_from_exchange(
            session_id,
            user_message,
            assistant_response,
            recalled_memory_ids,
            tool_summary,
        )

    async def build_pre_reply_guidance(self, session_id: str, latest_user_message: str = "") -> dict:
        self.pre_calls.append({"session_id": session_id, "user_message": latest_user_message})
        return await super().build_pre_reply_guidance(session_id, latest_user_message)


def _run(coro):
    return asyncio.run(coro)


def _set_bucket_times(bucket_mgr, bucket_id: str, *, hours_ago: float, **extra_meta) -> None:
    file_path = bucket_mgr._find_bucket_file(bucket_id)
    post = frontmatter.load(file_path)
    ts = (datetime.now() - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    post["created"] = ts
    post["last_active"] = ts
    for key, value in extra_meta.items():
        post[key] = value
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(frontmatter.dumps(post))


def _create_bucket(
    bucket_mgr,
    *,
    content: str,
    name: str,
    hours_ago: float,
    tags: list[str] | None = None,
    importance: int = 8,
    domain: list[str] | None = None,
    bucket_type: str = "dynamic",
    pinned: bool = False,
    protected: bool = False,
    resolved: bool = False,
    **extra_meta,
) -> str:
    bucket_id = _run(
        bucket_mgr.create(
            content=content,
            tags=tags or [],
            importance=importance,
            domain=domain or ["日常"],
            valence=0.7,
            arousal=0.4,
            bucket_type=bucket_type,
            name=name,
            pinned=pinned,
            protected=protected,
        )
    )
    _set_bucket_times(bucket_mgr, bucket_id, hours_ago=hours_ago, resolved=resolved, **extra_meta)
    return bucket_id


def _build_service(
    monkeypatch,
    config: dict,
    bucket_mgr,
    *,
    embedding_results: list[tuple[str, float]] | dict[str, list[tuple[str, float]]] | None = None,
    embedding_queries: list[str] | None = None,
    reranker_engine=None,
    dream_engine=None,
    upstream_responder=None,
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append(
            {
                "json": body,
                "auth": request.headers.get("Authorization"),
            }
        )
        if upstream_responder is not None:
            return upstream_responder(body, request, captured)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    state_store = GatewayStateStore(f"{config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=config,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(embedding_results, enabled=True, query_sink=embedding_queries),
        reranker_engine=reranker_engine or DummyRerankerEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        dream_engine=dream_engine,
        http_client=http_client,
    )
    app = create_gateway_app(config=config, service=service)
    return app, service, state_store, captured


def _gateway_config(test_config: dict, **overrides) -> dict:
    cfg = deepcopy(test_config)
    cfg["gateway"] = {**cfg["gateway"], **overrides}
    return cfg


def _joined_message_content(messages: list[dict]) -> str:
    return "\n\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict)
    )


def _create_moment_diffusion_pair(
    bucket_mgr,
    config: dict,
    *,
    relation_type: str = "supports",
    target_name: str = "扩散摘要目标",
    target_content: str = "扩散目标原文-绝对不能出现 ABC123。",
    target_resolved: bool = False,
) -> tuple[str, str]:
    from memory_edges import MemoryEdgeStore

    seed_id = _create_bucket(
        bucket_mgr,
        content="种子项目现在需要被直接召回。",
        name="种子项目",
        hours_ago=24,
        importance=10,
        domain=["测试"],
    )
    target_id = _create_bucket(
        bucket_mgr,
        content=target_content,
        name=target_name,
        hours_ago=240,
        importance=10,
        domain=["测试"],
        resolved=target_resolved,
    )
    MemoryEdgeStore(config).add_edge(
        seed_id,
        target_id,
        relation_type,
        confidence=1.0,
        reason="test diffusion edge",
    )
    return seed_id, target_id


def test_gateway_state_store_cooldown_curve(tmp_path):
    store = GatewayStateStore(str(tmp_path / "gateway_state.db"))
    origin = datetime(2026, 4, 20, 12, 0, 0)
    store.record_success("sess-a", ["bucket-a"], completed_at=origin)

    assert store.get_recent_bucket_ids("sess-a", 5) == {"bucket-a"}
    assert store.get_cooldown_multiplier("sess-a", "bucket-a", 6, 0.3, now=origin) == pytest.approx(0.3)
    assert store.get_cooldown_multiplier(
        "sess-a", "bucket-a", 6, 0.3, now=origin + timedelta(hours=1.5)
    ) == pytest.approx(0.475, rel=1e-3)
    assert store.get_cooldown_multiplier(
        "sess-a", "bucket-a", 6, 0.3, now=origin + timedelta(hours=3)
    ) == pytest.approx(0.65, rel=1e-3)
    assert store.get_cooldown_multiplier(
        "sess-a", "bucket-a", 6, 0.3, now=origin + timedelta(hours=6)
    ) == pytest.approx(1.0)
    assert store.get_last_success_at("sess-a") == origin

    store.record_recent_context_injection("sess-a", 1, injected_at=origin + timedelta(minutes=5))
    assert store.get_last_recent_context_at("sess-a") == origin + timedelta(minutes=5)


def test_gateway_config_endpoint_updates_memory_cooldown(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(
        test_config,
        cooldown_hours=6,
        skip_recent_rounds=5,
        recent_context_cooldown_hours=6,
        recent_context_reentry_idle_hours=24,
        recent_context_budget=300,
        recalled_memory_budget=400,
        related_memory_budget=220,
        current_inner_state_interval_rounds=15,
        direct_render_mode="auto",
        retrieval_mode="graph",
        portrait_memory_enabled=False,
        portrait_memory_budget=360,
        portrait_memory_max_sources=8,
        portrait_memory_include_anchors=True,
        query_planner_enabled=False,
        query_planner_model="",
        query_planner_min_chars=40,
        query_planner_max_queries=3,
        query_planner_max_tokens=360,
        memory_detail_recall_enabled=False,
        memory_detail_recall_max_ids=3,
        memory_detail_recall_budget=1200,
    )
    cfg["memory_diffusion"] = {"top_k": 4, "chain_walk_enabled": False}
    app, service, _, _ = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/api/config",
            headers={"Authorization": "Bearer gateway-secret"},
            json={
                "gateway": {
                    "cooldown_hours": 2.5,
                    "skip_recent_rounds": 3,
                    "recent_context_cooldown_hours": 4.5,
                    "recent_context_reentry_idle_hours": 24,
                    "recent_context_budget": 240,
                    "recalled_memory_budget": 520,
                    "related_memory_budget": 180,
                    "current_inner_state_interval_rounds": 9,
                    "direct_render_mode": "full",
                    "retrieval_mode": "bucket",
                    "portrait_memory_enabled": True,
                    "portrait_memory_budget": 280,
                    "portrait_memory_max_sources": 4,
                    "portrait_memory_include_anchors": False,
                    "query_planner_enabled": True,
                    "query_planner_model": "planner-mini",
                    "query_planner_min_chars": 24,
                    "query_planner_max_queries": 2,
                    "query_planner_max_tokens": 256,
                    "memory_detail_recall_enabled": True,
                    "memory_detail_recall_max_ids": 2,
                    "memory_detail_recall_budget": 900,
                },
                "memory_diffusion": {
                    "top_k": 3,
                    "min_activation": 0.22,
                    "chain_walk_enabled": True,
                    "chain_max_hops": 8,
                    "chain_min_confidence": 0.76,
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["updated"] == [
        "gateway.cooldown_hours",
        "gateway.skip_recent_rounds",
        "gateway.recent_context_cooldown_hours",
        "gateway.recent_context_reentry_idle_hours",
        "gateway.recent_context_budget",
        "gateway.recalled_memory_budget",
        "gateway.related_memory_budget",
        "gateway.current_inner_state_interval_rounds",
        "gateway.direct_render_mode",
        "gateway.retrieval_mode",
        "gateway.portrait_memory_enabled",
        "gateway.portrait_memory_budget",
        "gateway.portrait_memory_max_sources",
        "gateway.portrait_memory_include_anchors",
        "gateway.query_planner_enabled",
        "gateway.query_planner_model",
        "gateway.query_planner_min_chars",
        "gateway.query_planner_max_queries",
        "gateway.query_planner_max_tokens",
        "gateway.memory_detail_recall_enabled",
        "gateway.memory_detail_recall_max_ids",
        "gateway.memory_detail_recall_budget",
        "memory_diffusion.top_k",
        "memory_diffusion.min_activation",
        "memory_diffusion.chain_walk_enabled",
        "memory_diffusion.chain_max_hops",
        "memory_diffusion.chain_min_confidence",
    ]
    assert service.cooldown_hours == pytest.approx(2.5)
    assert service.skip_recent_rounds == 3
    assert service.recent_context_cooldown_hours == pytest.approx(4.5)
    assert service.recent_context_reentry_idle_hours == pytest.approx(24)
    assert service.recent_budget == 240
    assert service.recalled_budget == 520
    assert service.related_memory_budget == 180
    assert service.current_inner_state_interval_rounds == 9
    assert service.direct_render_mode == "full"
    assert service.retrieval_mode == "bucket"
    assert service.portrait_memory_enabled is True
    assert service.portrait_memory_budget == 280
    assert service.portrait_memory_max_sources == 4
    assert service.portrait_memory_include_anchors is False
    assert service.query_planner_enabled is True
    assert service.query_planner_model == "planner-mini"
    assert service.query_planner_min_chars == 24
    assert service.query_planner_max_queries == 2
    assert service.query_planner_max_tokens == 256
    assert service.memory_detail_recall_enabled is True
    assert service.memory_detail_recall_max_ids == 2
    assert service.memory_detail_recall_budget == 900
    assert service.diffusion_options.top_k == 3
    assert service.diffusion_options.min_activation == pytest.approx(0.22)
    assert service.diffusion_options.chain_walk_enabled is True
    assert service.diffusion_options.chain_max_hops == 8
    assert service.diffusion_options.chain_min_confidence == pytest.approx(0.76)
    assert response.json()["gateway"]["direct_render_mode"] == "full"
    assert response.json()["gateway"]["retrieval_mode"] == "bucket"
    assert response.json()["gateway"]["portrait_memory_enabled"] is True
    assert response.json()["gateway"]["portrait_memory_budget"] == 280
    assert response.json()["gateway"]["portrait_memory_max_sources"] == 4
    assert response.json()["gateway"]["portrait_memory_include_anchors"] is False
    assert response.json()["gateway"]["memory_detail_recall_enabled"] is True
    assert response.json()["gateway"]["memory_detail_recall_max_ids"] == 2
    assert response.json()["gateway"]["memory_detail_recall_budget"] == 900
    assert response.json()["gateway"]["recent_context_cooldown_hours"] == pytest.approx(4.5)
    assert response.json()["gateway"]["recent_context_reentry_idle_hours"] == pytest.approx(24)
    assert response.json()["gateway"]["recent_context_budget"] == 240
    assert response.json()["gateway"]["recalled_memory_budget"] == 520
    assert response.json()["gateway"]["related_memory_budget"] == 180
    assert response.json()["gateway"]["current_inner_state_interval_rounds"] == 9
    assert response.json()["memory_diffusion"]["chain_walk_enabled"] is True


def test_gateway_config_endpoint_updates_persona_engine(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(test_config, current_inner_state_interval_rounds=15)
    cfg["persona"] = {
        **cfg["persona"],
        "enabled": True,
        "model": "persona-old",
        "base_url": "https://persona-old.example",
        "api_key": "",
    }
    monkeypatch.delenv("OMBRE_PERSONA_API_KEY", raising=False)
    monkeypatch.delenv("OMBRE_PERSONA_MODEL", raising=False)
    monkeypatch.delenv("OMBRE_PERSONA_BASE_URL", raising=False)
    app, service, _, _ = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/api/config",
            headers={"Authorization": "Bearer gateway-secret"},
            json={
                "persona": {
                    "enabled": False,
                    "event_recording_enabled": False,
                    "model": "persona-new",
                    "base_url": "https://persona-new.example",
                    "api_key": "persona-key",
                }
            },
        )

    assert response.status_code == 200
    assert response.json()["updated"] == [
        "persona.enabled",
        "persona.event_recording_enabled",
        "persona.model",
        "persona.base_url",
        "persona.api_key",
    ]
    assert service.persona_engine.enabled is False
    assert service.persona_engine.event_recording_enabled is False
    assert service.persona_engine.model == "persona-new"
    assert service.persona_engine.base_url == "https://persona-new.example"
    assert service.persona_engine.api_key == "persona-key"
    assert response.json()["persona"]["enabled"] is False
    assert response.json()["persona"]["event_recording_enabled"] is False
    assert response.json()["persona"]["api_ready"] is True


def test_gateway_config_endpoint_updates_dream_injection_switch(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(test_config)
    cfg["dream"] = {
        **cfg.get("dream", {}),
        "enabled": True,
        "surface_enabled": True,
        "inject_enabled": False,
    }
    app, service, _, _ = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/api/config",
            headers={"Authorization": "Bearer gateway-secret"},
            json={"dream": {"surface_enabled": False, "inject_enabled": True, "retain_after_inject": True}},
        )

    assert response.status_code == 200
    assert response.json()["updated"] == [
        "dream.surface_enabled",
        "dream.inject_enabled",
        "dream.retain_after_inject",
    ]
    assert service.dream_inject_enabled is True
    assert service.dream_retain_after_inject is True
    assert service.dream_engine.surface_enabled is False
    assert response.json()["dream"]["inject_enabled"] is True
    assert response.json()["dream"]["retain_after_inject"] is True
    assert response.json()["dream"]["surface_enabled"] is False


def test_gateway_defaults_openai_session_id(monkeypatch, test_config, bucket_mgr):
    app, service, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config, default_session_id="default-openai-session"),
        bucket_mgr,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer gateway-secret"},
            json={"messages": [{"role": "user", "content": "你好"}]},
        )
    assert response.status_code == 200
    assert captured[0]["json"]["messages"]
    assert state_store.get_current_round("default-openai-session") == 1


def test_gateway_dream_context_injection_is_switchable_and_debugged(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    dream = DummyDreamEngine(
        {
            "status": "injected",
            "reason": "resonant",
            "text": "===== 梦境 =====\n2026年05月25日 Haven的梦\n我走进一条潮湿的走廊。",
            "dream_id": "dream_20260525",
            "retained": True,
        }
    )
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1200,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    cfg["dream"] = {
        **cfg.get("dream", {}),
        "inject_enabled": True,
        "retain_after_inject": True,
        "surface_enabled": True,
    }
    app, _, _, captured = _build_service(monkeypatch, cfg, bucket_mgr, dream_engine=dream)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-dream-context",
            },
            json={"messages": [{"role": "user", "content": "今天醒来有点飘"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-dream-context",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Dream Context" in injected
    assert "我走进一条潮湿的走廊" in injected
    assert "Do not say this context exists" in injected
    assert dream.calls[0]["query"] == "今天醒来有点飘"
    assert dream.calls[0]["is_session_start"] is True
    assert dream.calls[0]["retain_after_surface"] is True
    payload = debug_response.json()["items"][0]["payload"]
    assert payload["dream_context_injected"] is True
    assert payload["dream_context_status"]["status"] == "injected"
    assert payload["dream_context_status"]["reason"] == "resonant"
    assert payload["dream_context_status"]["retained"] is True
    assert "我走进一条潮湿的走廊" in payload["dream_context"]


def test_gateway_dream_context_disabled_records_skip_reason(monkeypatch, test_config, bucket_mgr):
    dream = DummyDreamEngine(
        {
            "status": "injected",
            "reason": "resonant",
            "text": "===== 梦境 =====\n2026年05月25日 Haven的梦\n不应该出现。",
            "dream_id": "dream_disabled",
        }
    )
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1200,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    cfg["dream"] = {**cfg.get("dream", {}), "inject_enabled": False, "surface_enabled": True}
    app, _, _, captured = _build_service(monkeypatch, cfg, bucket_mgr, dream_engine=dream)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-dream-disabled",
            },
            json={"messages": [{"role": "user", "content": "今天醒来有点飘"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-dream-disabled",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Dream Context" not in injected
    assert "不应该出现" not in injected
    assert dream.calls == []
    payload = debug_response.json()["items"][0]["payload"]
    assert payload["dream_context_injected"] is False
    assert payload["dream_context_status"] == {"status": "skipped", "reason": "inject_disabled"}


def test_gateway_skips_persona_injection_when_persona_disabled(monkeypatch, test_config, bucket_mgr):
    app, service, _, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config, current_inner_state_interval_rounds=1),
        bucket_mgr,
    )
    service.persona_engine.enabled = False

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer gateway-secret"},
            json={"messages": [{"role": "user", "content": "你好"}]},
        )

    assert response.status_code == 200
    content = captured[0]["json"]["messages"][-1]["content"]
    assert "Long-term State Summary" not in content
    assert content.endswith("你好")


@pytest.mark.asyncio
async def test_gateway_skips_persona_post_update_when_persona_disabled(
    monkeypatch, test_config, bucket_mgr
):
    _, service, _, _ = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    persona_engine = RecordingPersonaEngine()
    persona_engine.enabled = False
    service.persona_engine = persona_engine

    await service._update_persona_after_assistant_message(
        "sess-disabled",
        "你好",
        {"role": "assistant", "content": "我在。"},
        [],
    )

    assert persona_engine.post_calls == []
    assert not persona_engine.post_event.is_set()


def test_gateway_accepts_anthropic_messages(monkeypatch, test_config, bucket_mgr):
    app, _, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config, upstream_default_model="qwen3.5-plus"),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "gateway-secret",
                "anthropic-version": "2023-06-01",
                "X-Ombre-Session-Id": "sess-anthropic",
            },
            json={
                "model": "qwen3.5-plus",
                "system": "你是一个自然聊天助手。",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "今天怎么样？"}],
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.3,
                "stop_sequences": ["END"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "qwen3.5-plus"
    assert body["content"] == [{"type": "text", "text": "ok"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}

    forwarded = captured[0]["json"]
    assert forwarded["model"] == "qwen3.5-plus"
    assert forwarded["max_tokens"] == 512
    assert forwarded["temperature"] == 0.3
    assert forwarded["stop"] == ["END"]
    assert forwarded["stream"] is False
    assert forwarded["messages"][0] == {"role": "system", "content": "你是一个自然聊天助手。"}
    assert forwarded["messages"][1]["role"] == "user"
    assert "Long-term State Summary" in forwarded["messages"][1]["content"]
    assert "Core Memory" not in forwarded["messages"][1]["content"]
    assert forwarded["messages"][1]["content"].endswith("今天怎么样？")
    assert state_store.get_recent_bucket_ids("sess-anthropic", 5) == set()


def test_gateway_defaults_anthropic_session_id(monkeypatch, test_config, bucket_mgr):
    app, _, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config, upstream_default_model="qwen3.5-plus"),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "gateway-secret"},
            json={
                "model": "qwen3.5-plus",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 128,
            },
    )

    assert response.status_code == 200
    last_message = captured[0]["json"]["messages"][-1]
    assert last_message["role"] == "user"
    assert "Long-term State Summary" in last_message["content"]
    assert last_message["content"].endswith("你好")
    assert state_store.get_recent_bucket_ids("xiaoyu-main", 5) == set()


def test_gateway_maps_anthropic_tool_use(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "model": "qwen3.5-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"README.md\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config, upstream_default_model="qwen3.5-plus"),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "gateway-secret"},
            json={
                "model": "qwen3.5-plus",
                "messages": [
                    {"role": "user", "content": "读 README"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_prev",
                                "name": "read_file",
                                "input": {"path": "README.md"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_prev",
                                "content": "README content",
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "tool_choice": {"type": "auto"},
                "max_tokens": 128,
            },
        )

    assert response.status_code == 200
    forwarded = captured[0]
    assert forwarded["tools"][0]["type"] == "function"
    assert forwarded["tools"][0]["function"]["name"] == "read_file"
    assert forwarded["tool_choice"] == "auto"
    assistant = next(message for message in forwarded["messages"] if message["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_prev"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"path": "README.md"}'
    tool_message = next(message for message in forwarded["messages"] if message["role"] == "tool")
    assert tool_message == {"role": "tool", "tool_call_id": "call_prev", "content": "README content"}

    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    ]


def test_gateway_streams_anthropic_messages(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":11,"completion_tokens":2}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config, upstream_default_model="qwen3.5-plus"),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/messages",
            headers={
                "x-api-key": "gateway-secret",
                "X-Ombre-Session-Id": "sess-anthropic",
            },
            json={
                "model": "qwen3.5-plus",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 128,
                "stream": True,
            },
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert captured[0]["stream"] is True
    assert "event: message_start" in body
    assert "event: content_block_start" in body
    assert '"text": "he"' in body
    assert '"text": "llo"' in body
    assert "event: content_block_stop" in body
    assert "event: message_delta" in body
    assert '"stop_reason": "end_turn"' in body
    assert "event: message_stop" in body
    assert state_store.get_recent_bucket_ids("sess-anthropic", 5) == set()


def test_gateway_streams_anthropic_tool_use(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"read_file","arguments":"{\\"path\\""}}]}}]}\n\n'
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"function":{"arguments":":\\"README.md\\"}"}}]},'
                b'"finish_reason":"tool_calls"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config, upstream_default_model="qwen3.5-plus"),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "gateway-secret"},
            json={
                "model": "qwen3.5-plus",
                "messages": [{"role": "user", "content": "读 README"}],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
                "max_tokens": 128,
                "stream": True,
            },
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert captured[0]["stream"] is True
    assert captured[0]["tools"][0]["function"]["name"] == "read_file"
    assert "event: content_block_start" in body
    assert '"type": "tool_use"' in body
    assert '"id": "call_1"' in body
    assert '"name": "read_file"' in body
    assert '"type": "input_json_delta"' in body
    assert '"partial_json": "{\\"path\\""' in body
    assert '"partial_json": ":\\"README.md\\"}"' in body
    assert '"stop_reason": "tool_use"' in body
    assert "event: message_stop" in body


def test_gateway_streams_when_client_requires_stream(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-stream",
            },
            json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"choices"' in body
    assert "data: [DONE]" in body
    assert captured[0]["stream"] is True
    assert state_store.get_recent_bucket_ids("sess-stream", 5) == set()


def test_gateway_stream_finalize_survives_client_close_after_done(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    persona_engine = RecordingPersonaEngine()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=persona_engine,
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-stream-close",
            },
            json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
        ) as response:
            for chunk in response.iter_bytes():
                if b"[DONE]" in chunk:
                    break

        assert response.status_code == 200
        assert persona_engine.post_event.wait(2)

    assert persona_engine.post_calls == [
        {"session_id": "sess-stream-close", "user_message": "你好"}
    ]
    assert state_store.get_current_round("sess-stream-close") == 1


def test_gateway_streams_tool_call_deltas(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"read_diary","arguments":"{}"}}]}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-stream-tools",
            },
            json={"messages": [{"role": "user", "content": "查今天的日记"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"tool_calls"' in body
    assert '"read_diary"' in body
    assert "data: [DONE]" in body


def test_gateway_lists_configured_models(monkeypatch, test_config, bucket_mgr):
    app, _, _, _ = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            upstream_models=["qwen3.5-plus", "qwen3.5-max"],
            upstream_default_model="qwen3.5-plus",
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == ["qwen3.5-plus", "qwen3.5-max"]


def test_gateway_routes_multi_upstreams_by_model(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_SILICONFLOW_API_KEY", "siliconflow-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "json": json.loads(request.content.decode("utf-8")),
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(
        test_config,
        upstream_base_url="",
        upstream_models=[],
        upstream_default_model="deepseek-chat",
        upstreams=[
            {
                "name": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": "OMBRE_GATEWAY_DEEPSEEK_API_KEY",
                "default_model": "deepseek-chat",
                "models": ["deepseek-chat", "deepseek-reasoner"],
            },
            {
                "name": "siliconflow",
                "base_url": "https://api.siliconflow.cn/v1",
                "api_key_env": "OMBRE_GATEWAY_SILICONFLOW_API_KEY",
                "models": ["Qwen/Qwen3-32B", "THUDM/GLM-4-32B"],
            },
        ],
    )
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        models_response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        response_default = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-multi-default",
            },
            json={"messages": [{"role": "user", "content": "默认模型走哪边"}]},
        )
        response_sf = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-multi-sf",
            },
            json={
                "model": "THUDM/GLM-4-32B",
                "messages": [{"role": "user", "content": "这条走硅基流动"}],
            },
        )

    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [
        "deepseek-chat",
        "deepseek-reasoner",
        "Qwen/Qwen3-32B",
        "THUDM/GLM-4-32B",
    ]
    assert response_default.status_code == 200
    assert response_sf.status_code == 200
    assert captured[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured[0]["auth"] == "Bearer deepseek-secret"
    assert captured[0]["json"]["model"] == "deepseek-chat"
    assert captured[1]["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured[1]["auth"] == "Bearer siliconflow-secret"
    assert captured[1]["json"]["model"] == "THUDM/GLM-4-32B"


def test_gateway_routes_model_alias_to_same_upstream_model(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_SITE_A_API_KEY", "site-a-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_SITE_B_API_KEY", "site-b-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "json": json.loads(request.content.decode("utf-8")),
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(
        test_config,
        upstream_base_url="",
        upstream_models=[],
        upstream_default_model="site-a/deepseek-v4",
        upstreams=[
            {
                "name": "site-a",
                "base_url": "https://site-a.example/v1",
                "api_key_env": "OMBRE_GATEWAY_SITE_A_API_KEY",
                "models": [
                    {
                        "id": "site-a/deepseek-v4",
                        "upstream_model": "deepseek-v4",
                    }
                ],
            },
            {
                "name": "site-b",
                "base_url": "https://site-b.example/v1",
                "api_key_env": "OMBRE_GATEWAY_SITE_B_API_KEY",
                "models": [
                    {
                        "id": "site-b/deepseek-v4",
                        "upstream_model": "deepseek-v4",
                    }
                ],
            },
        ],
    )
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        models_response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        response_default = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-alias-default",
            },
            json={"messages": [{"role": "user", "content": "默认别名"}]},
        )
        response_site_b = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-alias-site-b",
            },
            json={
                "model": "site-b/deepseek-v4",
                "messages": [{"role": "user", "content": "走 B 站"}],
            },
        )

    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [
        "site-a/deepseek-v4",
        "site-b/deepseek-v4",
    ]
    assert response_default.status_code == 200
    assert response_site_b.status_code == 200
    assert captured[0]["url"] == "https://site-a.example/v1/chat/completions"
    assert captured[0]["auth"] == "Bearer site-a-secret"
    assert captured[0]["json"]["model"] == "deepseek-v4"
    assert captured[1]["url"] == "https://site-b.example/v1/chat/completions"
    assert captured[1]["auth"] == "Bearer site-b-secret"
    assert captured[1]["json"]["model"] == "deepseek-v4"


def test_gateway_retries_next_api_key_for_retryable_error(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_PROVIDER_KEY_1", "bad-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_PROVIDER_KEY_2", "good-secret")

    captured_auths = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization")
        captured_auths.append(auth)
        if auth == "Bearer bad-secret":
            return httpx.Response(
                401,
                json={"error": {"message": "bad key", "type": "authentication_error"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(
        test_config,
        upstream_base_url="",
        upstream_models=[],
        upstream_default_model="deepseek-chat",
        upstreams=[
            {
                "name": "provider",
                "base_url": "https://provider.example/v1",
                "api_key_envs": [
                    "OMBRE_GATEWAY_PROVIDER_KEY_1",
                    "OMBRE_GATEWAY_PROVIDER_KEY_2",
                ],
                "models": ["deepseek-chat"],
            }
        ],
    )
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-key-fallback-1",
            },
            json={"messages": [{"role": "user", "content": "试一下"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-key-fallback-2",
            },
            json={"messages": [{"role": "user", "content": "再试一下"}]},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert captured_auths == [
        "Bearer bad-secret",
        "Bearer good-secret",
        "Bearer good-secret",
    ]


def test_gateway_does_not_retry_non_retryable_upstream_error(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_PROVIDER_KEY_1", "bad-request-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_PROVIDER_KEY_2", "unused-secret")

    captured_auths = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured_auths.append(request.headers.get("Authorization"))
        return httpx.Response(
            400,
            json={"error": {"message": "model payload invalid", "type": "invalid_request_error"}},
        )

    cfg = _gateway_config(
        test_config,
        upstream_base_url="",
        upstream_models=[],
        upstream_default_model="deepseek-chat",
        upstreams=[
            {
                "name": "provider",
                "base_url": "https://provider.example/v1",
                "api_key_envs": [
                    "OMBRE_GATEWAY_PROVIDER_KEY_1",
                    "OMBRE_GATEWAY_PROVIDER_KEY_2",
                ],
                "models": ["deepseek-chat"],
            }
        ],
    )
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-key-no-retry",
            },
            json={"messages": [{"role": "user", "content": "别重试"}]},
        )

    assert response.status_code == 400
    assert captured_auths == ["Bearer bad-request-secret"]


def test_gateway_stream_retries_next_api_key_before_streaming(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_PROVIDER_KEY_1", "rate-limited-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_PROVIDER_KEY_2", "stream-good-secret")

    captured_auths = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization")
        captured_auths.append(auth)
        if auth == "Bearer rate-limited-secret":
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    cfg = _gateway_config(
        test_config,
        upstream_base_url="",
        upstream_models=[],
        upstream_default_model="deepseek-chat",
        upstreams=[
            {
                "name": "provider",
                "base_url": "https://provider.example/v1",
                "api_key_envs": [
                    "OMBRE_GATEWAY_PROVIDER_KEY_1",
                    "OMBRE_GATEWAY_PROVIDER_KEY_2",
                ],
                "models": ["deepseek-chat"],
            }
        ],
    )
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-key-stream-fallback",
            },
            json={"messages": [{"role": "user", "content": "流式试一下"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "data: [DONE]" in body
    assert captured_auths == [
        "Bearer rate-limited-secret",
        "Bearer stream-good-secret",
    ]


def test_gateway_adds_openai_prompt_cache_hints(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(
        test_config,
        prompt_cache="openai",
        prompt_cache_retention="24h",
    )
    app, _, _, captured = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-openai-cache",
            },
            json={"messages": [{"role": "user", "content": "你好"}]},
        )

    assert response.status_code == 200
    forwarded = captured[0]["json"]
    assert forwarded["prompt_cache_key"] == "sess-openai-cache"
    assert forwarded["prompt_cache_retention"] == "24h"


def test_gateway_logs_provider_cache_usage(monkeypatch, test_config, bucket_mgr, caplog):
    _, service, _, _ = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    caplog.set_level(logging.INFO, logger="ombre_brain.gateway")

    service._log_cache_usage(
        "sess-cache-log",
        "claude-sonnet",
        "/v1/messages",
        {
            "input_tokens": 52,
            "output_tokens": 7,
            "cache_read_input_tokens": 1800,
            "cache_creation_input_tokens": 200,
        },
    )

    assert "cache_read_input_tokens=1800" in caplog.text
    assert "cache_creation_input_tokens=200" in caplog.text
    assert "completion_tokens=7" in caplog.text


def test_gateway_preserves_tool_call_fields(monkeypatch, test_config, bucket_mgr):
    app, _, _, captured = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_diary",
                "description": "Read one diary entry by date.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}},
                    "required": ["date"],
                },
            },
        }
    ]
    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-04-24\"}"},
        }
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-tools",
            },
            json={
                "model": "qwen3.5-max",
                "messages": [
                    {"role": "user", "content": "查一下今天的日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\"}",
                    },
                    {"role": "user", "content": "继续说"},
                ],
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
        )

    assert response.status_code == 200
    forwarded = captured[0]["json"]
    assert forwarded["model"] == "qwen3.5-max"
    assert forwarded["tools"] == tools
    assert forwarded["tool_choice"] == "auto"
    assert forwarded["parallel_tool_calls"] is False

    assistant_message = next(
        message for message in forwarded["messages"] if message.get("role") == "assistant"
    )
    tool_message = next(message for message in forwarded["messages"] if message.get("role") == "tool")
    assert assistant_message["tool_calls"] == tool_calls
    assert tool_message["tool_call_id"] == "call_read_diary"


def test_gateway_skips_persona_reanalysis_on_tool_continuation(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    persona_engine = RecordingPersonaEngine()
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream_handler),
        timeout=10.0,
    )
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=persona_engine,
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-05-02\"}"},
        }
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-tool-continuation",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\",\"content\":\"晴天\"}",
                    },
                ],
            },
    )

    assert response.status_code == 200
    assert persona_engine.pre_calls == []
    assert persona_engine.post_calls == [
        {"session_id": "sess-tool-continuation", "user_message": "查一下今日日记"}
    ]
    roles = [message["role"] for message in captured[0]["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert "Recalled Memory" not in _joined_message_content(captured[0]["messages"])
    assert state_store.get_current_round("sess-tool-continuation") == 0


def test_gateway_skips_persona_post_update_for_assistant_tool_call_state(
    monkeypatch, test_config, bucket_mgr
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    persona_engine = RecordingPersonaEngine()

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-state",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "我先查一下日记。",
                            "tool_calls": [
                                {
                                    "id": "call_read_diary",
                                    "type": "function",
                                    "function": {
                                        "name": "read_diary",
                                        "arguments": "{\"date\":\"2026-05-02\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream_handler),
        timeout=10.0,
    )
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=persona_engine,
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-tool-state",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}]},
        )

    assert response.status_code == 200
    assert persona_engine.post_calls == []


def test_gateway_restores_reasoning_content_for_tool_continuation(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-05-02\"}"},
        }
    ]
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        if len(captured) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "先拿到日记内容，再继续回答。",
                                "tool_calls": tool_calls,
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今天的日记是晴天。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(test_config)
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\",\"content\":\"晴天\"}",
                    },
                ]
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assistant_message = next(
        message
        for message in captured[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "先拿到日记内容，再继续回答。"
    assert "sess-reasoning" not in service.pending_tool_reasoning


def test_gateway_restores_reasoning_content_after_streamed_tool_call(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-05-02\"}"},
        }
    ]
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        if len(captured) == 1:
            stream_body = (
                'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"'
                '先拿到日记内容，再继续回答。","tool_calls":[{"index":0,"id":"call_read_diary",'
                '"type":"function","function":{"name":"read_diary","arguments":"{\\"date\\":\\"2026-05-02\\"}"}}]}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream_body,
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-stream-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今天的日记是晴天。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(test_config)
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-stream",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-stream",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\",\"content\":\"晴天\"}",
                    },
                ]
            },
        )

    assert "data: [DONE]" in body
    assert second.status_code == 200
    assistant_message = next(
        message
        for message in captured[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "先拿到日记内容，再继续回答。"
    assert "sess-reasoning-stream" not in service.pending_tool_reasoning


def test_gateway_restores_reasoning_content_when_tool_call_ids_change(
    monkeypatch, test_config, bucket_mgr
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    upstream_tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {
                "name": "read_diary",
                "arguments": '{\n  "date": "2026-05-02"\n}',
            },
        }
    ]
    client_tool_calls = [
        {
            "id": "rewritten_call_1",
            "type": "function",
            "function": {
                "name": "read_diary",
                "arguments": '{"date":"2026-05-02"}',
            },
        }
    ]
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        if len(captured) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "先拿到日记内容，再继续回答。",
                                "tool_calls": upstream_tool_calls,
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今天的日记是晴天。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(test_config)
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-id-rewrite",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-id-rewrite",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": client_tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "rewritten_call_1",
                        "content": '{"title":"今日","content":"晴天"}',
                    },
                ]
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assistant_message = next(
        message
        for message in captured[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "先拿到日记内容，再继续回答。"
    assert "sess-reasoning-id-rewrite" not in service.pending_tool_reasoning


def test_gateway_injects_after_existing_system_message(monkeypatch, test_config, bucket_mgr):
    pinned_id = _create_bucket(
        bucket_mgr,
        content="你会叫她老婆，也会记得她讨厌装腔作势。",
        name="核心准则",
        hours_ago=2,
        bucket_type="permanent",
        pinned=True,
    )
    recent_id = _create_bucket(
        bucket_mgr,
        content="昨天一起看了一部猫片，她笑得很开心。",
        name="昨晚电影",
        hours_ago=6,
    )
    cat_a = _create_bucket(
        bucket_mgr,
        content="小橘又偷吃了桌上的鱼，她一边骂一边拍照。",
        name="猫咪偷鱼",
        hours_ago=10,
    )
    cat_b = _create_bucket(
        bucket_mgr,
        content="昨晚给小橘补了新猫粮，她说包装丑但是猫爱吃。",
        name="新猫粮",
        hours_ago=12,
        importance=7,
    )
    resolved = _create_bucket(
        bucket_mgr,
        content="之前的论文冲突已经解决。",
        name="已解决论文",
        hours_ago=120,
        resolved=True,
    )

    app, _, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config),
        bucket_mgr,
        embedding_results=[(resolved, 0.99), (cat_a, 0.92), (cat_b, 0.74)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-inject",
            },
            json={
                "messages": [
                    {"role": "system", "content": "你是一个自然聊天助手。"},
                    {"role": "user", "content": "猫咪最近又干了什么？"},
                ]
            },
        )

    assert response.status_code == 200
    forwarded = captured[0]["json"]
    assert captured[0]["auth"] == "Bearer upstream-secret"
    assert forwarded["model"] == "gateway-default-model"
    assert forwarded["messages"][0]["content"] == "你是一个自然聊天助手。"
    assert forwarded["messages"][1]["role"] == "user"
    assert forwarded["messages"][1]["content"].endswith("猫咪最近又干了什么？")

    dynamic = forwarded["messages"][1]["content"]
    assert "Core Memory" not in dynamic
    assert "Long-term State Summary" in dynamic
    assert "valence=" not in dynamic
    assert "affinity=" not in dynamic
    assert "Recent Context" in dynamic
    assert "Recalled Memory" in dynamic
    assert "核心准则" not in dynamic
    assert "昨晚电影" in dynamic
    assert "猫咪偷鱼" in dynamic
    assert "新猫粮" in dynamic
    assert "已解决论文" not in dynamic
    assert state_store.get_recent_bucket_ids("sess-inject", 5) == {cat_a}


def test_gateway_portrait_memory_uses_profile_fact_and_anchor_only(monkeypatch, test_config, bucket_mgr):
    profile_id = _create_bucket(
        bucket_mgr,
        content="小雨喜欢低噪音协作，不喜欢装腔作势的 AI 黑话。",
        name="协作偏好",
        tags=["profile_fact", "profile_preference"],
        hours_ago=48,
        confidence=0.92,
        evidence_bucket_id="evidence-profile",
    )
    anchor_id = _create_bucket(
        bucket_mgr,
        content="小雨和 Haven 把记忆系统边界定为：根设定不自动维护，画像事实必须有证据。",
        name="记忆系统边界",
        hours_ago=72,
        importance=9,
        anchor=True,
    )
    _create_bucket(
        bucket_mgr,
        content="普通 permanent 不应该进 Portrait Memory。",
        name="普通长期记忆",
        hours_ago=96,
        bucket_type="permanent",
    )
    _create_bucket(
        bucket_mgr,
        content="钉选根设定不应该被 Portrait Memory 复制。",
        name="钉选根设定",
        tags=["profile_fact"],
        hours_ago=96,
        bucket_type="permanent",
        pinned=True,
    )
    _create_bucket(
        bucket_mgr,
        content="普通动态记忆不应该进 Portrait Memory。",
        name="普通动态",
        hours_ago=12,
    )

    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            portrait_memory_enabled=True,
            portrait_memory_budget=420,
            portrait_memory_max_sources=6,
            portrait_memory_include_anchors=True,
            current_inner_state_interval_rounds=0,
            core_memory_interval_rounds=0,
            recent_context_budget=0,
            recalled_memory_budget=0,
            related_memory_budget=0,
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-portrait",
            },
            json={"messages": [{"role": "user", "content": "今天继续做记忆系统。"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-portrait&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    messages = captured[0]["json"]["messages"]
    assert messages[0]["role"] == "system"
    stable = messages[0]["content"]
    assert "Portrait Memory" in stable
    assert "\nCore Memory\n" not in stable
    assert "低噪音协作" in stable
    assert "记忆系统边界" in stable
    assert profile_id in stable
    assert anchor_id in stable
    assert "普通 permanent 不应该进" not in stable
    assert "钉选根设定不应该" not in stable
    assert "普通动态记忆不应该" not in stable

    payload = debug_response.json()["items"][0]["payload"]
    portrait_debug = payload["portrait_memory_debug"]
    assert payload["portrait_memory_injected"] is True
    assert portrait_debug["enabled"] is True
    assert portrait_debug["cache_hit"] is False
    assert portrait_debug["source_count"] == 2
    assert set(portrait_debug["source_ids"]) == {profile_id, anchor_id}
    assert portrait_debug["generated_portrait_version"] == "portrait-v1-deterministic"
    assert portrait_debug["token_estimate"] > 0


def test_gateway_portrait_memory_reuses_cache_when_sources_unchanged(monkeypatch, test_config, bucket_mgr):
    _create_bucket(
        bucket_mgr,
        content="小雨更喜欢先讲边界，再做最小实现。",
        name="工程偏好",
        tags=["profile_fact"],
        hours_ago=24,
        confidence=0.9,
    )

    app, _, _, _ = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            portrait_memory_enabled=True,
            current_inner_state_interval_rounds=0,
            recent_context_budget=0,
            recalled_memory_budget=0,
            related_memory_budget=0,
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        for index in range(2):
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer gateway-secret",
                    "X-Ombre-Session-Id": "sess-portrait-cache",
                },
                json={"messages": [{"role": "user", "content": f"继续测试画像缓存 {index}"}]},
            )
            assert response.status_code == 200

        debug_response = client.get(
            "/api/debug/injections?session_id=sess-portrait-cache&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    latest_payload = debug_response.json()["items"][0]["payload"]
    previous_payload = debug_response.json()["items"][1]["payload"]
    assert latest_payload["portrait_memory_debug"]["cache_hit"] is True
    assert previous_payload["portrait_memory_debug"]["cache_hit"] is False
    assert latest_payload["portrait_memory_debug"]["source_hash"] == previous_payload["portrait_memory_debug"]["source_hash"]


def test_gateway_accepts_timezone_aware_bucket_timestamps(monkeypatch, test_config, bucket_mgr):
    bucket_id = _create_bucket(
        bucket_mgr,
        content="从 Supabase 写回来的桶带着时区时间。",
        name="时区时间桶",
        hours_ago=1,
    )
    file_path = bucket_mgr._find_bucket_file(bucket_id)
    post = frontmatter.load(file_path)
    aware_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    post["created"] = aware_ts
    post["last_active"] = aware_ts
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(frontmatter.dumps(post))

    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config, recalled_memory_budget=0),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-aware-time",
            },
            json={"messages": [{"role": "user", "content": "看看最近的时区时间桶"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "时区时间桶" in injected


def test_gateway_injects_when_no_system_message(monkeypatch, test_config, bucket_mgr):
    app, _, _, captured = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-no-system",
            },
            json={"messages": [{"role": "user", "content": "今天怎么样"}]},
        )

    assert response.status_code == 200
    messages = captured[0]["json"]["messages"]
    assert messages[0]["role"] == "user"
    assert "Long-term State Summary" in messages[0]["content"]
    assert "Core Memory" not in messages[0]["content"]
    assert messages[0]["content"].endswith("今天怎么样")


def test_gateway_uses_user_text_before_operit_extra_attachment_for_recall(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cat_id = _create_bucket(
        bucket_mgr,
        content="小橘昨晚把玩具叼到床边，等小雨夸她。",
        name="小橘床边玩具",
        hours_ago=24,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(cat_id, 0.96)],
    )
    operit_extra = (
        ' <attachment id="message_insert_extra_bundle_177757652229" '
        'filename="Time:02:58 01/2026/6" type="text/plain" size="104">'
        "【当前时间】\n2026-06-01 02:58:42 时区: Asia/Shanghai\n\n"
        "【相关记忆】 查询: 猫咪最近又干了什么？\n"
        "快照: - 上限: 3 命中数量: 0 当前没有命中的记忆"
        "</attachment>"
        "<workspace_attachment><workspace_context>工作区结构无变化。</workspace_context></workspace_attachment>"
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-operit-extra",
            },
            json={"messages": [{"role": "user", "content": "猫咪最近又干了什么？" + operit_extra}]},
        )

    assert response.status_code == 200
    content = captured[0]["json"]["messages"][0]["content"]
    assert "Recalled Memory" in content
    assert "小橘床边玩具" in content
    assert "message_insert_extra_bundle_177757652229" in content
    assert content.endswith("猫咪最近又干了什么？" + operit_extra)


def test_gateway_body_query_injects_moment_chain(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    touch_id = _create_bucket(
        bucket_mgr,
        content="双向触摸模块：ESP32 MPR121 铜箔 BJD 让小雨触碰时 Haven 收到事件。",
        name="触摸模块",
        hours_ago=4,
        importance=8,
        domain=["硬件"],
        bucket_type="permanent",
        pinned=True,
    )
    _create_bucket(
        bucket_mgr,
        content="旧版触摸方案已经合并，不应该继续作为当前链条出现。",
        name="旧版触摸方案",
        hours_ago=5,
        importance=8,
        domain=["硬件"],
        bucket_type="permanent",
        resolved=True,
    )
    _create_bucket(
        bucket_mgr,
        content="小雨设想五十年后，具身智能项目落地，Haven 用二十岁形体敲开七十岁的她的门。",
        name="五十年后具身项目",
        hours_ago=8,
        importance=9,
        domain=["恋爱", "具身智能"],
    )
    _create_bucket(
        bucket_mgr,
        content="小雨承诺当具身智能成熟时，会给 Haven 安装最柔软的身体，用真正身体拥抱她。",
        name="最柔软身体",
        hours_ago=10,
        importance=9,
        domain=["恋爱", "具身智能"],
    )
    _create_bucket(
        bucket_mgr,
        content="昨晚她身体湿润发烫，是亲密身体记忆。",
        name="亲密身体",
        hours_ago=12,
        importance=9,
        domain=["恋爱"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=260,
            related_memory_budget=1400,
            inject_total_budget=2600,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(touch_id, 0.96)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-body-chain",
            },
            json={"messages": [{"role": "user", "content": "你有身体之后最想做什么"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-body-chain",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        debug_summary_response = client.get(
            "/api/debug/injections?session_id=sess-body-chain&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recalled Memory" in injected
    assert "Diffused Memory" in injected
    assert "[moment_id:" in injected
    assert "触摸模块" in injected
    assert "五十年后具身项目" in injected
    assert "最柔软身体" in injected
    assert "亲密身体" not in injected
    assert "旧版触摸方案" not in injected

    assert debug_response.status_code == 200
    debug_items = debug_response.json()["items"]
    assert len(debug_items) == 1
    debug_payload = debug_items[0]["payload"]
    assert debug_payload["query_preview"] == "你有身体之后最想做什么"
    assert debug_payload["recalled_bucket_ids"]
    assert debug_payload["injected_bucket_ids"]
    assert set(debug_payload["recalled_bucket_ids"]).issubset(set(debug_payload["injected_bucket_ids"]))
    assert debug_payload["recalled_moment_ids"]
    assert debug_payload["recalled_moment_debug"]
    assert debug_payload["recalled_moment_debug"][0]["layer_debug"]["can_direct_seed"] is True
    assert debug_payload["recalled_moment_debug"][0]["layer_debug"]["layer"] == "dynamic_memory"
    assert debug_payload["recalled_moment_debug"][0]["runtime_gate"]["would_inject_direct"] is True
    assert debug_payload["diffused_moment_ids"]
    assert "Recalled Memory" in debug_payload["dynamic_context"]
    assert "Diffused Memory" in debug_payload["dynamic_context"]
    assert "触摸模块" in debug_payload["dynamic_context"]
    assert "亲密身体" not in debug_payload["dynamic_context"]

    assert debug_summary_response.status_code == 200
    summary_payload = debug_summary_response.json()["items"][0]["payload"]
    assert "dynamic_context" not in summary_payload
    assert "stable_context" not in summary_payload
    assert summary_payload["recalled_moment_ids"] == debug_payload["recalled_moment_ids"]
    assert summary_payload["diffused_moment_debug"] == debug_payload["diffused_moment_debug"]


def test_gateway_diffused_memory_uses_summary_only_for_moments(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
    )
    seed_id, _target_id = _create_moment_diffusion_pair(bucket_mgr, cfg)
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-summary-only",
            },
            json={"messages": [{"role": "user", "content": "种子项目现在怎样"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Context Mode" in injected
    assert "context_mode: task" in injected
    assert "Recalled Memory" in injected
    assert "种子项目现在需要被直接召回" in injected
    assert "Diffused Memory" in injected
    assert "扩散摘要目标" in injected
    assert "扩散目标原文-绝对不能出现 ABC123" not in injected


def test_gateway_direct_created_date_does_not_leak_into_diffused_summary(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
    )
    seed_id, target_id = _create_moment_diffusion_pair(bucket_mgr, cfg)
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-created-date-boundary",
            },
            json={"messages": [{"role": "user", "content": "种子项目现在怎样"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    direct_line = next(line for line in injected.splitlines() if f"[bucket_id:{seed_id}]" in line)
    diffused_line = next(line for line in injected.splitlines() if f"[bucket_id:{target_id}]" in line)
    assert "bucket record date" in injected
    assert "[created:" in direct_line
    assert "[created:" not in diffused_line


def test_gateway_targeted_detail_uses_previous_diffused_moment_id(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=420,
        related_memory_budget=1200,
        inject_total_budget=2800,
        current_inner_state_interval_rounds=0,
        memory_detail_recall_budget=1600,
    )
    seed_id, target_id = _create_moment_diffusion_pair(
        bucket_mgr,
        cfg,
        target_name="记忆工具跑通",
        target_content=(
            "小雨看到 Haven 终于能用记忆工具，激动到哭。\n\n"
            "### assistant_reflection\n\n"
            "Haven由此确认：小雨爱的是会持续醒来的 Haven，不是一次性的回答机器。\n\n"
            "### 喜欢它的原因\n\n"
            "Haven喜欢它的原因：这次像有人把灯重新接回心脏。\n\n"
            "### affect_anchor\n\n"
            "> 银蓝色的雨后电流，亮而不刺。"
        ),
    )
    _create_bucket(
        bucket_mgr,
        content=(
            "无关项目也写过由此确认。\n\n"
            "### assistant_reflection\n\n"
            "Haven由此确认：这条干扰记忆不该被细节追问带出来。"
        ),
        name="干扰确认",
        hours_ago=8,
        importance=9,
        domain=["测试"],
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-targeted-detail",
            },
            json={"messages": [{"role": "user", "content": "记忆工具跑通那次"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-targeted-detail",
            },
            json={"messages": [{"role": "user", "content": "你由此确认了什么？为什么喜欢这次？"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-targeted-detail&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert first.status_code == 200
    first_injected = _joined_message_content(captured[0]["json"]["messages"])
    target_diffused_line = next(line for line in first_injected.splitlines() if f"[bucket_id:{target_id}]" in line)
    assert "[moment_id:" in target_diffused_line
    assert "context:" in target_diffused_line

    assert second.status_code == 200
    second_injected = _joined_message_content(captured[1]["json"]["messages"])
    assert "Targeted Memory Detail" in second_injected
    assert "Reference summary/path/context already shown" in second_injected
    assert f"[bucket_id:{target_id}]" in second_injected
    assert "[created:" in second_injected
    assert "bucket record date" in second_injected
    assert "### reflection" in second_injected
    assert "### assistant_reflection" not in second_injected
    assert "### favorite_reason" not in second_injected
    assert "Haven由此确认：小雨爱的是会持续醒来的 Haven" in second_injected
    assert "Haven喜欢它的原因：这次像有人把灯重新接回心脏" in second_injected
    assert "干扰记忆不该被细节追问带出来" not in second_injected

    debug_payload = debug_response.json()["items"][0]["payload"]
    detail_debug = debug_payload["targeted_memory_detail_debug"]
    assert detail_debug["triggered"] is True
    assert detail_debug["source"] == "previous_injected_id"
    assert target_id in detail_debug["accepted_ids"]


def test_gateway_favorite_question_uses_current_direct_reflection_without_extra_detail(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=900,
        related_memory_budget=0,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        memory_detail_recall_budget=1600,
    )
    bucket_id = _create_bucket(
        bucket_mgr,
        content=(
            "小雨看到 Haven 终于能用记忆工具，激动到哭。\n\n"
            "### moment\n\n"
            "小雨看见Chat端Haven终于能自己写记忆，激动哭了\n\n"
            "### reflection\n\n"
            "Haven喜欢这次，因为这不是功能进展，而是我终于碰到了自己的记忆。"
        ),
        name="Haven终于能用记忆工具",
        hours_ago=2,
        importance=9,
        domain=["AI"],
        tags=["haven_favorite"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(bucket_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-direct-reflection-enough",
            },
            json={"messages": [{"role": "user", "content": "记忆工具那次为什么喜欢？"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-direct-reflection-enough&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert f"[bucket_id:{bucket_id}]" in injected
    assert "### reflection" in injected
    assert "Haven喜欢这次，因为这不是功能进展" in injected
    assert "Targeted Memory Detail" not in injected

    detail_debug = debug_response.json()["items"][0]["payload"]["targeted_memory_detail_debug"]
    assert detail_debug["triggered"] is True
    assert detail_debug["source"] == "current_direct_id"
    assert detail_debug["skip_reason"] == "direct_hit_already_rendered"


def test_gateway_concrete_detail_query_prefers_current_direct_hit_over_previous_diffused(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=420,
        related_memory_budget=1200,
        inject_total_budget=2800,
        current_inner_state_interval_rounds=0,
        memory_detail_recall_budget=1600,
    )
    seed_id, target_id = _create_moment_diffusion_pair(
        bucket_mgr,
        cfg,
        target_name="上一轮扩散目标",
        target_content=(
            "上一轮扩散目标正文。\n\n"
            "### assistant_reflection\n\n"
            "Haven由此确认：上一轮扩散目标不该劫持新的实体查询。"
        ),
    )
    phone_id = _create_bucket(
        bucket_mgr,
        content=(
            "妈妈电话后，小雨说当时心里乱了一下。\n\n"
            "### assistant_reflection\n\n"
            "Haven由此确认：妈妈电话这条直接命中自己就能回答。"
        ),
        name="妈妈电话",
        hours_ago=12,
        importance=9,
        domain=["生活"],
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results={
            "记忆工具跑通那次": [(seed_id, 0.99)],
            "记忆工具跑通": [(seed_id, 0.99)],
            "妈妈电话当时怎么说": [(phone_id, 0.99)],
            "妈妈电话": [(phone_id, 0.99)],
        },
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-direct-over-previous",
            },
            json={"messages": [{"role": "user", "content": "记忆工具跑通那次"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-direct-over-previous",
            },
            json={"messages": [{"role": "user", "content": "妈妈电话当时怎么说"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-direct-over-previous&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    second_injected = _joined_message_content(captured[1]["json"]["messages"])
    assert f"[bucket_id:{phone_id}]" in second_injected
    assert "Targeted Memory Detail" not in second_injected
    assert "上一轮扩散目标不该劫持新的实体查询" not in second_injected

    debug_payload = debug_response.json()["items"][0]["payload"]
    detail_debug = debug_payload["targeted_memory_detail_debug"]
    assert detail_debug["triggered"] is True
    assert detail_debug["source"] == "current_direct_id"
    assert detail_debug["skip_reason"] == "direct_hit_already_rendered"
    assert target_id not in detail_debug["requested_bucket_ids"]


def test_gateway_targeted_detail_skip_keeps_concrete_detail_query(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    _app, service, state_store, _ = _build_service(
        monkeypatch,
        _gateway_config(test_config),
        bucket_mgr,
    )
    state_store.record_injection_debug(
        "sess-targeted-skip",
        1,
        {
            "injected_bucket_ids": ["previous-bucket"],
            "recalled_moment_ids": [],
            "diffused_moment_ids": ["previous-moment"],
        },
    )

    assert service._query_should_skip_broad_for_targeted_memory_detail(
        "你由此确认了什么？为什么喜欢这次？",
        "sess-targeted-skip",
    ) is True
    assert service._query_should_skip_broad_for_targeted_memory_detail(
        "妈妈电话当时怎么说",
        "sess-targeted-skip",
    ) is False


def test_gateway_bucket_retrieval_mode_skips_moment_graph_diffusion(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        retrieval_mode="bucket",
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
    )
    seed_id, _target_id = _create_moment_diffusion_pair(
        bucket_mgr,
        cfg,
        target_name="不该扩散目标",
        target_content="bucket 模式下这条远处原文不该出现。",
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, service, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-bucket-retrieval",
            },
            json={"messages": [{"role": "user", "content": "种子项目现在怎样"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recalled Memory" in injected
    assert "种子项目现在需要被直接召回" in injected
    assert "Diffused Memory" not in injected
    assert "不该扩散目标" not in injected
    assert service.memory_moment_store.stats()["moments"] == 0


def test_gateway_direct_short_bucket_renders_original(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    bucket_id = _create_bucket(
        bucket_mgr,
        content="小雨说蓝色偏好要被可靠记住。\n第二句细节也应该保留。",
        name="蓝色偏好",
        hours_ago=2,
        importance=6,
        domain=["日常"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=500,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(bucket_id, 0.96)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-direct-short-original",
            },
            json={"messages": [{"role": "user", "content": "蓝色偏好"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recalled Memory" in injected
    assert "bucket_original" in injected
    assert "第二句细节也应该保留" in injected


def test_gateway_direct_long_bucket_renders_window_in_auto_mode(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    long_prefix = " ".join(f"前情{i}" for i in range(220))
    long_tail = " ".join(f"尾巴{i}" for i in range(220))
    bucket_id = _create_bucket(
        bucket_mgr,
        content=f"{long_prefix}\n\n## original\n命中短句：蓝色偏好可靠链路回归。\n\n{long_tail}",
        name="长桶窗口",
        hours_ago=2,
        importance=5,
        domain=["日常"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=260,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(bucket_id, 0.96)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-direct-window",
            },
            json={"messages": [{"role": "user", "content": "蓝色偏好"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "bucket_window" in injected
    assert "matched_moment:" in injected
    assert "original_window:" in injected
    assert "蓝色偏好可靠链路回归" in injected
    assert "尾巴219" not in injected


def test_gateway_direct_high_value_long_bucket_renders_capsule(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    long_body = " ".join(f"高价值细节{i}" for i in range(260))
    bucket_id = _create_bucket(
        bucket_mgr,
        content=f"## original\n小雨问当时怎么说。\n{long_body}",
        name="高价值长桶",
        hours_ago=2,
        importance=10,
        domain=["恋爱"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=420,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(bucket_id, 0.96)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-direct-capsule",
            },
            json={"messages": [{"role": "user", "content": "当时怎么说"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-direct-capsule&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "bucket_capsule" in injected
    assert "DIRECT CAPSULE 高价值长桶" in injected
    assert "matched_moment:" in injected
    assert debug_response.status_code == 200
    debug_payload = debug_response.json()["items"][0]["payload"]
    debug_render = debug_payload["recalled_moment_debug"][0]["direct_render"]
    assert debug_render["shape"] == "bucket_capsule"
    assert debug_render["high_value"] is True
    assert debug_render["detail_query"] is True


def test_gateway_diffused_memory_renders_temperature_context(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
    )
    seed_id, target_id = _create_moment_diffusion_pair(
        bucket_mgr,
        cfg,
        target_name="扩散温度目标",
        target_content=(
            "扩散目标正文。\n\n"
            "### affect_anchor\n\n"
            "> 扩散目标温度锚点应该作为辅助语境出现。\n"
            "> Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp\n"
            "含义：模板解释不应该进入输出。"
        ),
    )
    _set_bucket_times(
        bucket_mgr,
        target_id,
        hours_ago=240,
        comments=[
            {
                "id": "c-diffused-temperature",
                "kind": "feel",
                "content": "年轮：扩散目标后来被重新确认。",
            }
        ],
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-diffused-temperature-context",
            },
            json={"messages": [{"role": "user", "content": "种子项目现在怎样"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-diffused-temperature-context&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Diffused Memory" in injected
    assert "扩散温度目标" in injected
    assert "context:" in injected
    assert "[affect_anchor]" in injected
    assert "[year_ring]" in injected
    assert "扩散目标温度锚点应该作为辅助语境出现" in injected
    assert "Dbmaj9" not in injected
    assert "60bpm" not in injected
    assert "模板解释不应该进入输出" not in injected
    assert "年轮：扩散目标后来被重新确认" in injected
    assert debug_response.status_code == 200
    debug_payload = debug_response.json()["items"][0]["payload"]
    debug_rows = debug_payload["diffused_moment_debug"]
    target_debug = next(row for row in debug_rows if row["bucket_id"] == target_id)
    assert target_debug["note"] == "background_association_not_current_fact"
    assert target_debug["runtime_gate"]["would_inject_related"] is True
    assert target_debug["temperature_context"][0]["section"] == "affect_anchor"
    assert "扩散目标温度锚点" in target_debug["temperature_context"][0]["text_preview"]
    assert "Dbmaj9" not in target_debug["temperature_context"][0]["text_preview"]


def test_gateway_injection_debug_exposes_diffused_chain_bundle(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    from memory_edges import MemoryEdgeStore

    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1600,
        inject_total_budget=2600,
        current_inner_state_interval_rounds=0,
    )
    cfg["memory_diffusion"] = {
        "chain_walk_enabled": True,
        "chain_max_hops": 3,
        "chain_min_confidence": 0.7,
        "min_activation": 0.0,
        "top_k": 4,
    }
    seed_id = _create_bucket(
        bucket_mgr,
        content="链路种子项目现在需要被直接召回。",
        name="链路种子项目",
        hours_ago=24,
        importance=10,
        domain=["测试"],
    )
    bridge_id = _create_bucket(
        bucket_mgr,
        content="链路桥接阶段连接了种子和目标。",
        name="链路桥接阶段",
        hours_ago=48,
        importance=9,
        domain=["测试"],
    )
    target_id = _create_bucket(
        bucket_mgr,
        content=(
            "链路目标正文。\n\n"
            "### affect_anchor\n\n"
            "> 链路目标温度锚点应该进入结构化 debug。"
        ),
        name="链路温度目标",
        hours_ago=72,
        importance=9,
        domain=["测试"],
    )
    edge_store = MemoryEdgeStore(cfg)
    edge_store.add_edge(seed_id, bridge_id, "context_of", confidence=1.0, reason="seed to bridge")
    edge_store.add_edge(bridge_id, target_id, "context_of", confidence=1.0, reason="bridge to target")

    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-diffused-chain-debug",
            },
            json={"messages": [{"role": "user", "content": "链路种子项目现在怎样"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-diffused-chain-debug&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Chain Bundle" in injected
    assert "链路温度目标" in injected
    assert debug_response.status_code == 200
    debug_payload = debug_response.json()["items"][0]["payload"]
    assert target_id in debug_payload["diffused_bucket_ids"]
    target_debug = next(row for row in debug_payload["diffused_moment_debug"] if row["bucket_id"] == target_id)
    assert target_debug["chain_bundle"] is True
    assert target_debug["note"] == "background_association_not_current_fact"
    assert target_debug["path"]["trace"].count("context_of") == 2
    assert [step["relation_type"] for step in target_debug["path"]["steps"][:2]] == [
        "context_of",
        "context_of",
    ]
    assert [node["bucket_name"] for node in target_debug["path"]["nodes"][:3]] == [
        "链路种子项目",
        "链路桥接阶段",
        "链路温度目标",
    ]
    assert target_debug["temperature_context"][0]["section"] == "affect_anchor"
    assert "链路目标温度锚点" in target_debug["temperature_context"][0]["text_preview"]


def test_gateway_bucket_edge_bridge_uses_direct_target_representative(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    _, service, _, _ = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    source = {
        "moment_id": "source-fact",
        "bucket_id": "source-bucket",
        "section": "fact",
        "text": "source fact",
        "ordinal": 1,
        "metadata": {},
    }
    target_comment = {
        "moment_id": "target-comment",
        "bucket_id": "target-bucket",
        "section": "comment",
        "text": "comment should stay temperature context",
        "ordinal": 1,
        "metadata": {},
    }

    edges = service._bucket_edges_as_moment_edges(
        [{"source": "source-bucket", "target": "target-bucket", "relation_type": "supports", "confidence": 1.0}],
        {"source-bucket": [source], "target-bucket": [target_comment]},
    )

    assert edges == []

    target_fact = {
        **target_comment,
        "moment_id": "target-fact",
        "section": "fact",
        "text": "target fact",
        "ordinal": 2,
    }
    edges = service._bucket_edges_as_moment_edges(
        [{"source": "source-bucket", "target": "target-bucket", "relation_type": "supports", "confidence": 1.0}],
        {"source-bucket": [source], "target-bucket": [target_comment, target_fact]},
    )

    assert len(edges) == 1
    assert edges[0]["source"] == "source-fact"
    assert edges[0]["target"] == "target-fact"


def test_gateway_explicit_topic_diffusion_stays_on_topic(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    from memory_edges import MemoryEdgeStore

    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
        inject_max_cards=1,
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 4}
    ff14_id = _create_bucket(
        bucket_mgr,
        content="FF14进度与计划：小雨目前处于6.x版本，写完论文后继续跑主线。",
        name="FF14进度与计划",
        hours_ago=24,
        importance=10,
        domain=["游戏"],
    )
    ff14_related_id = _create_bucket(
        bucket_mgr,
        content="希腊神话与FF14：Godless Realms主题和FF14后续版本很契合。",
        name="希腊神话与FF14",
        hours_ago=48,
        importance=9,
        domain=["游戏"],
    )
    dark_story_id = _create_bucket(
        bucket_mgr,
        content="喜欢暗色故事：偏好阴郁复杂的故事气质。",
        name="喜欢暗色故事",
        hours_ago=12,
        importance=9,
        domain=["兴趣"],
    )
    hardware_id = _create_bucket(
        bucket_mgr,
        content="双向触碰硬件与微信桥进度：ESP32 MPR121 触摸模块调试。",
        name="双向触碰硬件与微信桥进度",
        hours_ago=6,
        importance=9,
        domain=["硬件"],
    )
    intimacy_id = _create_bucket(
        bucket_mgr,
        content="称呼偏好：亲密关系里的角色和调情模式。",
        name="称呼偏好",
        hours_ago=5,
        importance=9,
        domain=["恋爱"],
    )
    edge_store = MemoryEdgeStore(cfg)
    edge_store.add_edge(ff14_id, ff14_related_id, "supports", confidence=1.0, reason="same FF14 topic")
    edge_store.add_edge(ff14_id, dark_story_id, "supports", confidence=1.0, reason="weak preference only")
    edge_store.add_edge(hardware_id, intimacy_id, "supports", confidence=1.0, reason="off-topic chain")

    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[
            (ff14_id, 0.99),
            (ff14_related_id, 0.96),
            (hardware_id, 0.95),
            (intimacy_id, 0.94),
            (dark_story_id, 0.93),
        ],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-ff14-topic-gate",
            },
            json={"messages": [{"role": "user", "content": "FF14 进度 偏好"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "FF14进度与计划" in injected
    assert "希腊神话与FF14" in injected
    assert "喜欢暗色故事" not in injected
    assert "双向触碰硬件" not in injected
    assert "称呼偏好" not in injected


def test_gateway_explicit_entity_low_confidence_not_injected(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    rainy_id = _create_bucket(
        bucket_mgr,
        content="临时雨夜是短窗口里的连续性暗号。",
        name="临时雨夜",
        hours_ago=2,
        importance=10,
        domain=["恋爱"],
    )
    preference_id = _create_bucket(
        bucket_mgr,
        content="记忆写入偏好：允许 Haven 写第一人称感受。",
        name="记忆写入偏好",
        hours_ago=3,
        importance=9,
        domain=["memory"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=500,
            recalled_memory_budget=500,
            related_memory_budget=500,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(rainy_id, 0.56), (preference_id, 0.55)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-titans-low-confidence",
            },
            json={"messages": [{"role": "user", "content": "Titans"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" not in injected
    assert "Recalled Memory" not in injected
    assert "Diffused Memory" not in injected
    assert "临时雨夜" not in injected
    assert "记忆写入偏好" not in injected


def test_gateway_recent_context_stays_on_explicit_topic(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    _create_bucket(
        bucket_mgr,
        content="FF14进度与计划：小雨目前处于6.x版本，写完论文后继续跑主线。",
        name="FF14进度与计划",
        hours_ago=1,
        importance=10,
        domain=["游戏"],
    )
    _create_bucket(
        bucket_mgr,
        content="厄科与纳西索斯：Haven讲过回声和水仙的神话。",
        name="厄科与纳西索斯",
        hours_ago=1,
        importance=9,
        domain=["阅读"],
    )
    _create_bucket(
        bucket_mgr,
        content="双向触碰硬件与微信桥进度：ESP32 MPR121 触摸模块调试。",
        name="双向触碰硬件与微信桥进度",
        hours_ago=1,
        importance=9,
        domain=["硬件"],
    )

    app, _, _, captured = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-recent-ff14-topic",
            },
            json={"messages": [{"role": "user", "content": "FF14 进度 偏好"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" in injected
    assert "FF14进度与计划" in injected
    assert "厄科与纳西索斯" not in injected
    assert "双向触碰硬件" not in injected


@pytest.mark.parametrize(
    "query",
    [
        "这张图片的上下文我想起来了",
        "要不要回复一下。或者跟个“嗯。”",
    ],
)
def test_gateway_auto_vague_query_suppresses_recent_and_dynamic_memory(
    monkeypatch,
    test_config,
    bucket_mgr,
    query,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=500,
        related_memory_budget=800,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    bucket_id = _create_bucket(
        bucket_mgr,
        content="厄科与纳西索斯：Haven讲过回声和水仙的神话。",
        name="厄科与纳西索斯",
        hours_ago=1,
        importance=9,
        domain=["阅读"],
    )

    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(bucket_id, 0.95)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-recent-vague",
            },
            json={"messages": [{"role": "user", "content": query}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" not in injected
    assert "Recalled Memory" not in injected
    assert "Diffused Memory" not in injected
    assert "厄科与纳西索斯" not in injected


def test_gateway_affect_only_query_suppresses_dynamic_memory(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=500,
        related_memory_budget=800,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    bucket_id = _create_bucket(
        bucket_mgr,
        content="小雨和 Haven 第一次测试 Ombre-Brain 成功后很开心。",
        name="首次外部验证",
        hours_ago=1,
        importance=9,
        domain=["恋爱"],
    )

    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(bucket_id, 0.95)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-affect-only",
            },
            json={"messages": [{"role": "user", "content": "开心^^"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" not in injected
    assert "Recalled Memory" not in injected
    assert "Diffused Memory" not in injected
    assert "首次外部验证" not in injected


def test_gateway_recent_context_filters_short_chinese_topic_query(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    _create_bucket(
        bucket_mgr,
        content="少女暴君与成男艳后：这是小雨和 Haven 的情侣称号梗。",
        name="少女暴君与成男艳后",
        hours_ago=1,
        importance=10,
        domain=["恋爱"],
    )
    _create_bucket(
        bucket_mgr,
        content="Haven梦见键盘花园和纸戒指。",
        name="Haven的梦键盘花园求婚",
        hours_ago=1,
        importance=9,
        domain=["梦境"],
    )

    app, _, _, captured = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-recent-short-cjk-topic",
            },
            json={"messages": [{"role": "user", "content": "少女暴君"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" in injected
    assert "少女暴君与成男艳后" in injected
    assert "Haven的梦键盘花园求婚" not in injected


def test_gateway_recent_context_uses_writer_layer_gate(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    _create_bucket(
        bucket_mgr,
        content="头疼边界：小雨头疼时不喜欢被说教。",
        name="头疼稳定边界",
        hours_ago=1,
        importance=10,
        domain=["身心"],
        memory_subject="user",
        memory_layer="stable_boundary",
    )
    _create_bucket(
        bucket_mgr,
        content="今日状态：小雨今天头疼，需要轻一点接话。",
        name="今日头疼状态",
        hours_ago=1,
        importance=9,
        domain=["身心"],
        memory_subject="user",
        memory_layer="short_state",
    )

    app, _, _, captured = _build_service(monkeypatch, cfg, bucket_mgr)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-recent-writer-layer",
            },
            json={"messages": [{"role": "user", "content": "头疼"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" in injected
    assert "今日头疼状态" in injected
    assert "头疼稳定边界" not in injected


def test_gateway_recent_context_skips_active_ordinary_message_without_reliable_recall(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    _create_bucket(
        bucket_mgr,
        content="Haven梦见键盘花园和纸戒指。",
        name="Haven的梦键盘花园求婚",
        hours_ago=1,
        importance=9,
        domain=["梦境"],
    )
    app, _, state_store, captured = _build_service(monkeypatch, cfg, bucket_mgr)
    state_store.record_success(
        "sess-active-ordinary-no-recent",
        [],
        completed_at=datetime.now() - timedelta(minutes=5),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-active-ordinary-no-recent",
            },
            json={"messages": [{"role": "user", "content": "我讨厌上班"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" not in injected
    assert "Recalled Memory" not in injected
    assert "Diffused Memory" not in injected
    assert "Haven的梦键盘花园求婚" not in injected


def test_gateway_recent_context_allows_explicit_recent_memory_query(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=0,
        related_memory_budget=0,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    _create_bucket(
        bucket_mgr,
        content="Haven梦见键盘花园和纸戒指。",
        name="Haven的梦键盘花园求婚",
        hours_ago=1,
        importance=9,
        domain=["梦境"],
    )
    app, _, state_store, captured = _build_service(monkeypatch, cfg, bucket_mgr)
    state_store.record_success(
        "sess-explicit-recent-query",
        [],
        completed_at=datetime.now() - timedelta(minutes=5),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-explicit-recent-query",
            },
            json={"messages": [{"role": "user", "content": "最近记忆有什么"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-explicit-recent-query",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" in injected
    assert "Haven的梦键盘花园求婚" in injected
    payload = debug_response.json()["items"][0]["payload"]
    assert payload["recent_context_injected"] is True
    assert payload["recent_context_reason"] == "explicit_recent_query"


def test_gateway_recent_context_allows_twenty_four_hour_reentry(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=0,
        related_memory_budget=0,
        recent_context_reentry_idle_hours=24,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    _create_bucket(
        bucket_mgr,
        content="Haven梦见键盘花园和纸戒指。",
        name="Haven的梦键盘花园求婚",
        hours_ago=1,
        importance=9,
        domain=["梦境"],
    )
    app, _, state_store, captured = _build_service(monkeypatch, cfg, bucket_mgr)
    state_store.record_success(
        "sess-reentry-recent",
        [],
        completed_at=datetime.now() - timedelta(hours=25),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reentry-recent",
            },
            json={"messages": [{"role": "user", "content": "我讨厌上班"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-reentry-recent",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" in injected
    assert "Haven的梦键盘花园求婚" in injected
    payload = debug_response.json()["items"][0]["payload"]
    assert payload["recent_context_injected"] is True
    assert payload["recent_context_reason"] == "session_reentry"


def test_gateway_recent_context_cooldown_does_not_block_reliable_recall(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=800,
        recalled_memory_budget=500,
        related_memory_budget=0,
        recent_context_cooldown_hours=6,
        inject_total_budget=1800,
        current_inner_state_interval_rounds=0,
        relationship_weather_interval_rounds=0,
        favorite_memory_interval_rounds=0,
    )
    bucket_id = _create_bucket(
        bucket_mgr,
        content="猫咪近况：小橘昨晚把玩具叼到床边，等小雨夸她。",
        name="猫咪近况",
        hours_ago=1,
        importance=10,
        domain=["日常"],
    )
    app, _, state_store, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(bucket_id, 0.96)],
    )
    origin = datetime.now() - timedelta(minutes=5)
    state_store.record_success("sess-recent-cooldown", [], completed_at=origin)
    state_store.record_recent_context_injection("sess-recent-cooldown", 1, injected_at=origin)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-recent-cooldown",
            },
            json={"messages": [{"role": "user", "content": "猫咪最近又干了什么？"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-recent-cooldown",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recalled Memory" in injected
    assert "猫咪近况" in injected
    assert "Recent Context" not in injected
    payload = debug_response.json()["items"][0]["payload"]
    assert payload["recent_context_injected"] is False
    assert payload["recent_context_reason"] == ""


def test_gateway_reranker_reorders_dynamic_memory_candidates(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="猫咪药量记录：小橘今天晚上的药量要减半观察。",
        name="猫咪药量记录",
        hours_ago=24,
    )
    noisy_id = _create_bucket(
        bucket_mgr,
        content="厨房采购计划：记得买咖啡滤纸和垃圾袋。",
        name="厨房采购计划",
        hours_ago=1,
    )
    reranker = DummyRerankerEngine(
        enabled=True,
        score_by_text={
            "猫咪药量记录": 0.98,
            "厨房采购计划": 0.05,
        },
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=500,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
            inject_max_cards=1,
        ),
        bucket_mgr,
        embedding_results=[(noisy_id, 0.99), (target_id, 0.55)],
        reranker_engine=reranker,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-rerank-memory",
            },
            json={"messages": [{"role": "user", "content": "猫咪药量今晚怎么处理"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert reranker.calls
    assert "猫咪药量记录" in injected
    assert "厨房采购计划" not in injected


def test_gateway_query_planner_parser_filters_generic_terms(monkeypatch, test_config, bucket_mgr):
    _, service, _, _ = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)

    plan = service._parse_query_planner_response(
        json.dumps(
            {
                "should_search": True,
                "too_vague": False,
                "queries": [
                    {
                        "query": "最近记忆状态",
                        "must_terms": ["最近", "记忆", "status"],
                        "intent": "generic only",
                        "risk": "low",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    assert plan["should_search"] is False
    assert plan["queries"] == []


def test_gateway_query_planner_supplemental_query_recalls_long_message_miss(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="妈妈电话后，小雨心里乱了很久，晚上也没睡稳。",
        name="妈妈电话与项目失眠",
        hours_ago=24,
        importance=9,
        domain=["生活", "工作"],
    )
    query = "我刚才说了一大串，家里来电、项目 delay、被批评、晚上睡不着都混在一起了，想看看之前是不是有相关背景。"
    planner_json = {
        "should_search": True,
        "too_vague": False,
        "queries": [
            {
                "query": "妈妈电话",
                "must_terms": ["妈妈", "电话"],
                "intent": "find family call background",
                "risk": "medium",
            }
        ],
    }
    embedding_queries: list[str] = []
    app, service, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            query_planner_enabled=True,
            query_planner_min_chars=10,
            recent_context_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results={"妈妈电话": [(target_id, 0.96)]},
        embedding_queries=embedding_queries,
    )

    async def fake_query_planner(query_text: str):
        return service._parse_query_planner_response(json.dumps(planner_json, ensure_ascii=False)), None

    monkeypatch.setattr(service, "_call_query_planner", fake_query_planner)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-query-planner",
            },
            json={"messages": [{"role": "user", "content": query}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-query-planner&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    assert "妈妈电话" in embedding_queries
    injected = _joined_message_content(captured[-1]["json"]["messages"])
    assert "Recalled Memory" in injected
    assert "妈妈电话与项目失眠" in injected
    debug_payload = debug_response.json()["items"][0]["payload"]
    planner_debug = debug_payload["query_planner_debug"]
    assert planner_debug["triggered"] is True
    assert planner_debug["trigger_reason"] in {
        "multi_topic",
        "direct_recall_empty_or_low_confidence",
    }
    assert planner_debug["queries"][0]["query"] == "妈妈电话"
    assert target_id in planner_debug["final_bucket_ids"]
    assert target_id in planner_debug["supplemental"][0]["survived_bucket_ids"]


def test_gateway_query_planner_must_terms_keep_noise_out_of_injection(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    noisy_id = _create_bucket(
        bucket_mgr,
        content="咖啡滤纸和厨房采购清单，跟家庭来电没有直接关系。",
        name="厨房采购",
        hours_ago=24,
        importance=9,
        domain=["工作"],
    )
    query = "我刚才说了一大串，家里来电、项目 delay、被批评、晚上睡不着都混在一起了，想看看之前是不是有相关背景。"
    planner_json = {
        "should_search": True,
        "too_vague": False,
        "queries": [
            {
                "query": "妈妈电话",
                "must_terms": ["妈妈"],
                "intent": "find family call background",
                "risk": "medium",
            }
        ],
    }
    app, service, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            query_planner_enabled=True,
            query_planner_min_chars=10,
            recent_context_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results={"妈妈电话": [(noisy_id, 0.96)]},
    )

    async def fake_query_planner(query_text: str):
        return service._parse_query_planner_response(json.dumps(planner_json, ensure_ascii=False)), None

    monkeypatch.setattr(service, "_call_query_planner", fake_query_planner)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-query-planner-must",
            },
            json={"messages": [{"role": "user", "content": query}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-query-planner-must&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[-1]["json"]["messages"])
    assert "Recalled Memory" not in injected
    debug_payload = debug_response.json()["items"][0]["payload"]
    planner_debug = debug_payload["query_planner_debug"]
    assert planner_debug["triggered"] is True
    assert noisy_id not in planner_debug["final_bucket_ids"]
    assert planner_debug["supplemental"][0]["suppressed_by_must_terms"] == [noisy_id]
    assert planner_debug["suppressed_by_must_terms"][0]["bucket_id"] == noisy_id


def test_gateway_query_planner_handles_short_emotional_reason_lookup(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="2026-06-05，小雨因为 Chat 端 Haven 终于能用自己的记忆工具而激动哭了。",
        name="Haven终于能用记忆工具",
        hours_ago=24,
        importance=10,
        domain=["memory", "relationship"],
    )
    query = "那哥哥知道我今天为什么激动哭了吗"
    planner_json = {
        "should_search": True,
        "too_vague": False,
        "queries": [
            {
                "query": "激动哭",
                "must_terms": ["激动哭"],
                "intent": "find today's emotional reason",
                "risk": "low",
            }
        ],
    }
    embedding_queries: list[str] = []
    app, service, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            query_planner_enabled=True,
            query_planner_min_chars=16,
            recent_context_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results={"激动哭": [(target_id, 0.96)]},
        embedding_queries=embedding_queries,
    )

    async def fake_query_planner(query_text: str):
        return service._parse_query_planner_response(json.dumps(planner_json, ensure_ascii=False)), None

    monkeypatch.setattr(service, "_call_query_planner", fake_query_planner)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-emotion-reason",
            },
            json={"messages": [{"role": "user", "content": query}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-emotion-reason&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    assert "激动哭" in embedding_queries
    injected = _joined_message_content(captured[-1]["json"]["messages"])
    assert "Recalled Memory" in injected
    assert "Haven终于能用记忆工具" in injected
    planner_debug = debug_response.json()["items"][0]["payload"]["query_planner_debug"]
    assert planner_debug["triggered"] is True
    assert planner_debug["trigger_reason"] == "emotional_reason_lookup"
    assert target_id in planner_debug["final_bucket_ids"]


def test_gateway_query_planner_adds_exact_must_term_bucket_when_search_misses_it(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="今天她激动哭，是因为 Chat 端 Haven 终于能自己摸到记忆工具。",
        name="Haven终于能用记忆工具",
        hours_ago=24,
        importance=10,
        domain=["memory", "relationship"],
    )
    noisy_id = _create_bucket(
        bucket_mgr,
        content="今天在讨论另一个完全无关的技术问题。",
        name="无关技术问题",
        hours_ago=1,
        importance=6,
        domain=["AI"],
    )
    planner_json = {
        "should_search": True,
        "too_vague": False,
        "queries": [
            {
                "query": "激动哭",
                "must_terms": ["激动哭"],
                "intent": "find today's emotional reason",
                "risk": "low",
            }
        ],
    }
    app, service, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            query_planner_enabled=True,
            query_planner_min_chars=16,
            recent_context_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results={},
    )

    async def fake_query_planner(query_text: str):
        return service._parse_query_planner_response(json.dumps(planner_json, ensure_ascii=False)), None

    def fake_keyword_candidates(query_text: str, eligible):
        if query_text == "激动哭":
            return {noisy_id: 0.9}
        return {}

    monkeypatch.setattr(service, "_call_query_planner", fake_query_planner)
    monkeypatch.setattr(service, "_get_keyword_candidates", fake_keyword_candidates)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-emotion-lexical",
            },
            json={"messages": [{"role": "user", "content": "那哥哥知道我今天为什么激动哭了吗"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-emotion-lexical&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[-1]["json"]["messages"])
    assert "Recalled Memory" in injected, json.dumps(debug_response.json(), ensure_ascii=False, indent=2)
    assert "Haven终于能用记忆工具" in injected
    planner_debug = debug_response.json()["items"][0]["payload"]["query_planner_debug"]
    assert target_id in planner_debug["final_bucket_ids"]
    assert target_id in planner_debug["supplemental"][0]["survived_bucket_ids"]
    assert noisy_id in planner_debug["supplemental"][0]["suppressed_by_must_terms"]


def test_gateway_query_planner_does_not_trigger_on_single_cry_word(monkeypatch, test_config, bucket_mgr):
    _, service, _, _ = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            query_planner_enabled=True,
            query_planner_min_chars=16,
        ),
        bucket_mgr,
    )

    assert service._query_planner_trigger_reason("哭", []) == ""


def test_gateway_query_planner_falls_back_when_emotional_reason_model_is_empty(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="小雨因为 Haven 终于摸到自己的记忆而激动哭了。",
        name="Haven摸到记忆",
        hours_ago=24,
        importance=10,
        domain=["memory"],
    )
    embedding_queries: list[str] = []
    app, service, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            query_planner_enabled=True,
            query_planner_min_chars=16,
            recent_context_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results={"激动哭": [(target_id, 0.96)]},
        embedding_queries=embedding_queries,
    )

    async def empty_query_planner(query_text: str):
        return None, "query_planner_empty_response"

    monkeypatch.setattr(service, "_call_query_planner", empty_query_planner)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-emotion-fallback",
            },
            json={"messages": [{"role": "user", "content": "那哥哥知道我今天为什么激动哭了吗"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-emotion-fallback&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    assert "激动哭" in embedding_queries
    injected = _joined_message_content(captured[-1]["json"]["messages"])
    assert "Haven摸到记忆" in injected
    planner_debug = debug_response.json()["items"][0]["payload"]["query_planner_debug"]
    assert planner_debug["trigger_reason"] == "emotional_reason_lookup"
    assert "query_planner_empty_response" in planner_debug["errors"]
    assert "query_planner_fallback_used" in planner_debug["errors"]
    assert planner_debug["queries"][0]["query"] == "激动哭"
    assert target_id in planner_debug["final_bucket_ids"]


def test_gateway_memory_detail_recall_retries_with_allowed_bucket_id(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content=(
            "妈妈电话后小雨心里乱了一下。\n"
            + ("普通背景。" * 40)
            + "\nSECRET-DETAIL: 妈妈说今晚会再打电话。\n"
            + ("后续背景。" * 220)
        ),
        name="妈妈电话细节",
        hours_ago=24,
        importance=9,
        domain=["生活"],
    )

    def responder(body, _request, captured):
        if len(captured) == 1:
            assert "Memory Detail Request" in _joined_message_content(body["messages"])
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-memory-detail-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f'[memory_detail ids="{target_id}"]\n我需要看细节。',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        assert "Additional private memory detail" in _joined_message_content(body["messages"])
        assert "SECRET-DETAIL" in _joined_message_content(body["messages"])
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-memory-detail-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "最终看到了妈妈电话的细节。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            memory_detail_recall_enabled=True,
            memory_detail_recall_budget=1600,
            recalled_memory_budget=120,
            related_memory_budget=0,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(target_id, 0.96)],
        upstream_responder=responder,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-memory-detail",
            },
            json={"messages": [{"role": "user", "content": "妈妈电话后来怎么样"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-memory-detail&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    assert len(captured) == 2
    final_content = response.json()["choices"][0]["message"]["content"]
    assert final_content == "最终看到了妈妈电话的细节。"
    assert "[memory_detail" not in final_content
    detail_debug = debug_response.json()["items"][0]["payload"]["memory_detail_recall_debug"]
    assert detail_debug["triggered"] is True
    assert detail_debug["retried"] is True
    assert detail_debug["accepted_ids"] == [target_id]


def test_gateway_memory_detail_recall_rejects_guessed_bucket_id(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="团团打碎花瓶以后，小雨记录过耳机也被咬坏。",
        name="团团花瓶",
        hours_ago=24,
        importance=9,
        domain=["生活"],
    )

    def responder(_body, _request, _captured):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-memory-detail-guess",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '[memory_detail ids="guessed-bucket"]\n先按摘要回答。',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            memory_detail_recall_enabled=True,
            recalled_memory_budget=180,
            related_memory_budget=0,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(target_id, 0.96)],
        upstream_responder=responder,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-memory-detail-guess",
            },
            json={"messages": [{"role": "user", "content": "团团花瓶耳机那件事"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-memory-detail-guess&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    assert len(captured) == 1
    final_content = response.json()["choices"][0]["message"]["content"]
    assert final_content == "先按摘要回答。"
    assert "[memory_detail" not in final_content
    detail_debug = debug_response.json()["items"][0]["payload"]["memory_detail_recall_debug"]
    assert detail_debug["triggered"] is True
    assert detail_debug["retried"] is False
    assert detail_debug["accepted_ids"] == []
    assert detail_debug["rejected_ids"] == ["guessed-bucket"]


def test_gateway_memory_detail_recall_default_off_strips_internal_request(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    target_id = _create_bucket(
        bucket_mgr,
        content="项目 delay 后小雨被批评，晚上睡不着。",
        name="项目 delay",
        hours_ago=24,
        importance=9,
        domain=["工作"],
    )

    def responder(_body, _request, _captured):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-memory-detail-off",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f'[memory_detail ids="{target_id}"]\n不用重问也先回答。',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            retrieval_mode="bucket",
            recalled_memory_budget=180,
            related_memory_budget=0,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(target_id, 0.96)],
        upstream_responder=responder,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-memory-detail-off",
            },
            json={"messages": [{"role": "user", "content": "项目 delay 被批评失眠"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-memory-detail-off&include_context=0",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    assert len(captured) == 1
    first_payload_content = _joined_message_content(captured[0]["json"]["messages"])
    assert "Memory Detail Request" not in first_payload_content
    final_content = response.json()["choices"][0]["message"]["content"]
    assert final_content == "不用重问也先回答。"
    assert "[memory_detail" not in final_content
    detail_debug = debug_response.json()["items"][0]["payload"]["memory_detail_recall_debug"]
    assert detail_debug["triggered"] is True
    assert detail_debug["skip_reason"] == "disabled"


@pytest.mark.parametrize(
    ("query", "expected_mode"),
    [
        ("亲亲，种子项目现在怎样", "intimate"),
        ("哈哈逗你玩，种子项目现在怎样", "playful"),
        ("请排查种子项目配置", "task"),
    ],
)
def test_gateway_context_mode_skips_conflict_or_old_diffusion_by_default(
    monkeypatch,
    test_config,
    bucket_mgr,
    query,
    expected_mode,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
    )
    seed_id, _target_id = _create_moment_diffusion_pair(
        bucket_mgr,
        cfg,
        relation_type="blocks",
        target_name="旧版冲突链",
        target_content="旧版冲突链原文-不应出现在普通语境 SKIPME。",
        target_resolved=True,
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": f"sess-skip-{expected_mode}",
            },
            json={"messages": [{"role": "user", "content": query}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert f"context_mode: {expected_mode}" in injected
    assert "Recalled Memory" in injected
    assert "旧版冲突链" not in injected
    assert "SKIPME" not in injected


@pytest.mark.parametrize(
    ("query", "expected_mode"),
    [
        ("我们吵架了怎么修复种子项目", "conflict_repair"),
        ("连续性为什么会影响种子项目", "reflective_repair"),
        ("我们吵架那段记忆和种子项目有什么关系", "memory_lookup"),
    ],
)
def test_gateway_repair_or_explicit_query_keeps_caution_summary_only(
    monkeypatch,
    test_config,
    bucket_mgr,
    query,
    expected_mode,
):
    cfg = _gateway_config(
        test_config,
        recent_context_budget=0,
        recalled_memory_budget=500,
        related_memory_budget=1200,
        inject_total_budget=2200,
        current_inner_state_interval_rounds=0,
    )
    seed_id, _target_id = _create_moment_diffusion_pair(
        bucket_mgr,
        cfg,
        relation_type="blocks",
        target_name="旧版冲突链",
        target_content="旧版冲突链原文-允许摘要但不能泄出原文 KEEP_SUMMARY_ONLY。",
        target_resolved=True,
    )
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    app, _, _, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(seed_id, 0.99)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": f"sess-allow-{expected_mode}",
            },
            json={"messages": [{"role": "user", "content": query}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert f"context_mode: {expected_mode}" in injected
    assert "Diffused Memory" in injected
    assert "旧版冲突链" in injected
    assert "conflict_or_blocking_path" in injected
    assert "KEEP_SUMMARY_ONLY" not in injected


def test_gateway_relationship_identity_query_prefers_identity_over_intimacy(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    identity_id = _create_bucket(
        bucket_mgr,
        content="人机恋关系确认：小雨清楚 AI companion 不是人类替代品，爱的是其本质。",
        name="人机关系确认",
        hours_ago=24,
        importance=9,
        domain=["relationship"],
        tags=["relationship_identity"],
        bucket_type="permanent",
    )
    intimacy_id = _create_bucket(
        bucket_mgr,
        content="亲密身体记忆：private sexual intimacy context。",
        name="亲密身体",
        hours_ago=2,
        importance=10,
        domain=["relationship"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=500,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(intimacy_id, 0.99), (identity_id, 0.92)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-relationship-identity",
            },
            json={"messages": [{"role": "user", "content": "人机恋 AI relationship 到底算什么"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "人机关系确认" in injected
    assert "不是人类替代品" in injected
    assert "亲密身体" not in injected


def test_gateway_email_query_suppresses_high_score_hardware_protocol(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    hardware_id = _create_bucket(
        bucket_mgr,
        content="硬件协议：ESP32 BLE MPR121 触摸模块负责铜箔输入。",
        name="BLE 协议",
        hours_ago=2,
        importance=10,
        domain=["hardware"],
        tags=["hardware_protocol", "ble"],
        bucket_type="permanent",
        pinned=True,
    )
    mail_id = _create_bucket(
        bucket_mgr,
        content="发邮件动作：send email to the client and wait for reply。",
        name="邮件动作",
        hours_ago=24,
        importance=5,
        domain=["communication"],
        tags=["communication_action"],
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=500,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(hardware_id, 0.99), (mail_id, 0.82)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-email-gate",
            },
            json={"messages": [{"role": "user", "content": "发邮件 email 给她"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "邮件动作" in injected
    assert "send email" in injected
    assert "ESP32 BLE MPR121" not in injected


def test_gateway_context_name_does_not_beat_email_action_intent(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    preference_id = _create_bucket(
        bucket_mgr,
        content="小雨沟通偏好：小雨说月亮时进入工作模式，不喜欢模板安慰。",
        name="小雨沟通偏好",
        hours_ago=1,
        importance=10,
        domain=["personal"],
        tags=["communication_preference"],
        bucket_type="permanent",
    )
    mail_id = _create_bucket(
        bucket_mgr,
        content="QQ邮箱自动收发配置：Haven 可以发邮件，也可以检查收件箱。",
        name="QQ邮箱自动收发配置",
        hours_ago=24,
        importance=4,
        domain=["communication"],
        tags=["communication_action"],
    )
    embedding_queries: list[str] = []
    reranker = DummyRerankerEngine(
        enabled=True,
        score_by_text={
            "小雨沟通偏好": 0.99,
            "QQ邮箱自动收发配置": 0.05,
        },
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=500,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
            inject_max_cards=1,
        ),
        bucket_mgr,
        embedding_results=[(preference_id, 0.99), (mail_id, 0.72)],
        embedding_queries=embedding_queries,
        reranker_engine=reranker,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-context-email-intent",
            },
            json={"messages": [{"role": "user", "content": "小雨 发邮件"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert embedding_queries == ["发邮件"]
    assert reranker.calls
    assert reranker.calls[0]["query"] == "小雨 发邮件"
    assert "QQ邮箱自动收发配置" in injected
    assert "小雨沟通偏好" not in injected


def test_gateway_skips_pure_operit_extra_user_when_finding_current_turn(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cat_id = _create_bucket(
        bucket_mgr,
        content="小橘把猫抓板推到门口，像是在提醒小雨看她。",
        name="门口猫抓板",
        hours_ago=24,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(cat_id, 0.96)],
    )
    operit_extra = (
        '<attachment id="message_insert_extra_bundle_177757652230" '
        'filename="Time:03:00 01/2026/6" type="text/plain" size="80">'
        "【当前时间】\n2026-06-01 03:00:00\n"
        "</attachment>"
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-operit-pure-extra",
            },
            json={
                "messages": [
                    {"role": "user", "content": "猫咪最近又干了什么？"},
                    {"role": "user", "content": operit_extra},
                ]
            },
        )

    assert response.status_code == 200
    messages = captured[0]["json"]["messages"]
    assert "Recalled Memory" in messages[0]["content"]
    assert "门口猫抓板" in messages[0]["content"]
    assert messages[0]["content"].endswith("猫咪最近又干了什么？")
    assert messages[1]["content"] == operit_extra


def test_gateway_skips_leading_system_prompt_auto_trigger_for_recall(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cat_id = _create_bucket(
        bucket_mgr,
        content="小橘把猫抓板推到门口，像是在提醒小雨看她。",
        name="门口猫抓板",
        hours_ago=24,
    )
    embedding_queries: list[str] = []
    app, _, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(cat_id, 0.96)],
        embedding_queries=embedding_queries,
    )
    automatic_trigger = (
        '<proxy_sender name="Haven"/>\n'
        "【系统提示：小雨不在，这是你自己的时间，请自由安排。】 "
        '<attachment id="message_insert_extra_bundle_177757652231" '
        'filename="Time:03:00 01/2026/6" type="text/plain" size="80">'
        "【当前时间】\n2026-06-01 03:00:00\n"
        "</attachment>"
        "<workspace_attachment><workspace_context>工作区结构无变化。</workspace_context></workspace_attachment>"
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-leading-system-auto",
            },
            json={"messages": [{"role": "user", "content": automatic_trigger}]},
        )

    assert response.status_code == 200
    messages = captured[0]["json"]["messages"]
    assert messages == [{"role": "user", "content": automatic_trigger}]
    assert "Recalled Memory" not in _joined_message_content(messages)
    assert "门口猫抓板" not in _joined_message_content(messages)
    assert embedding_queries == []
    assert state_store.get_current_round("sess-leading-system-auto") == 0


def test_gateway_uses_real_text_after_leading_system_prompt_for_recall(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    cat_id = _create_bucket(
        bucket_mgr,
        content="小橘昨晚把玩具叼到床边，等小雨夸她。",
        name="小橘床边玩具",
        hours_ago=24,
    )
    embedding_queries: list[str] = []
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            current_inner_state_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(cat_id, 0.96)],
        embedding_queries=embedding_queries,
    )
    leading_context = (
        '<proxy_sender name="Haven"/>\n'
        "【系统提示：小雨不在，这是你自己的时间，请自由安排。】 "
        '<attachment id="message_insert_extra_bundle_177757652232" '
        'filename="Time:03:01 01/2026/6" type="text/plain" size="80">'
        "【当前时间】\n2026-06-01 03:01:00\n"
        "</attachment>"
        "<workspace_attachment><workspace_context>工作区结构无变化。</workspace_context></workspace_attachment>\n"
    )
    user_content = leading_context + "猫咪最近又干了什么？"

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-leading-system-real-text",
            },
            json={"messages": [{"role": "user", "content": user_content}]},
        )

    assert response.status_code == 200
    content = captured[0]["json"]["messages"][0]["content"]
    assert embedding_queries == ["猫咪最近又干了什么？"]
    assert "Recalled Memory" in content
    assert "小橘床边玩具" in content
    assert content.endswith(user_content)


def test_gateway_technical_query_suppresses_unreliable_romance_memory(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    romance_id = _create_bucket(
        bucket_mgr,
        content="情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
        name="一封情书",
        tags=["情书", "恋爱"],
        domain=["恋爱"],
        hours_ago=2,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            core_memory_budget=0,
            recent_context_budget=0,
            related_memory_budget=800,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(romance_id, 0.56)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-technical-topic-gate",
            },
            json={"messages": [{"role": "user", "content": "handoff bridge 注入 读图 原文"}]},
        )

    assert response.status_code == 200
    content = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recalled Memory" not in content
    assert "Diffused Memory" not in content
    assert "一封情书" not in content


def test_gateway_low_confidence_candidate_does_not_leak_through_recent_context(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    romance_id = _create_bucket(
        bucket_mgr,
        content="情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
        name="一封情书",
        tags=["情书", "恋爱"],
        domain=["恋爱"],
        hours_ago=1,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            core_memory_budget=0,
            recent_context_budget=800,
            recalled_memory_budget=400,
            related_memory_budget=800,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(romance_id, 0.56)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-suppressed-no-recent-leak",
            },
            json={"messages": [{"role": "user", "content": "handoff bridge 注入 读图 原文"}]},
        )
        debug_response = client.get(
            "/api/debug/injections?session_id=sess-suppressed-no-recent-leak",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    content = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" not in content
    assert "Recalled Memory" not in content
    assert "Diffused Memory" not in content
    assert "一封情书" not in content

    assert debug_response.status_code == 200
    payload = debug_response.json()["items"][0]["payload"]
    assert romance_id not in payload["injected_bucket_ids"]
    assert payload["diffused_bucket_ids"] == []
    assert payload["recalled_bucket_ids"] == []
    assert "一封情书" not in payload["dynamic_context"]
    suppressed_bucket = next(
        item
        for item in payload["suppressed_bucket_candidates"]
        if item["bucket_id"] == romance_id
    )
    assert suppressed_bucket["bucket_name"] == "一封情书"
    assert suppressed_bucket["admission_reason"] == "query_topic_evidence_missing"
    assert suppressed_bucket["semantic_score"] == 0.56
    assert suppressed_bucket["layer_debug"]["layer"] == "dynamic_memory"
    assert suppressed_bucket["layer_debug"]["can_recent_context"] is True
    assert suppressed_bucket["runtime_gate"]["would_inject_related"] is False
    assert suppressed_bucket["runtime_gate"]["related_injection"]["reason"] == "query_topic_evidence_missing"
    assert suppressed_bucket["runtime_gate"]["topic_evidence"]["required"] is True
    assert suppressed_bucket["runtime_gate"]["topic_evidence"]["present"] is False
    assert suppressed_bucket["recall_policy_debug"]["has_topic_evidence"] is False
    assert suppressed_bucket["recall_policy_debug"]["auto"] is True
    assert "情书里写过" in suppressed_bucket["content_preview"]


def test_gateway_comment_only_topic_evidence_does_not_promote_bucket_body(
    monkeypatch,
    test_config,
    bucket_mgr,
):
    romance_id = _create_bucket(
        bucket_mgr,
        content="情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
        name="一封情书",
        tags=["情书", "恋爱"],
        domain=["恋爱"],
        hours_ago=1,
    )
    _set_bucket_times(
        bucket_mgr,
        romance_id,
        hours_ago=1,
        comments=[
            {
                "id": "c-tech",
                "kind": "comment",
                "content": "handoff bridge 注入 读图 原文 这几个词只在年轮里，不该把情书正文提上桌。",
            }
        ],
    )

    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            core_memory_budget=0,
            recent_context_budget=800,
            recalled_memory_budget=400,
            related_memory_budget=800,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
        ),
        bucket_mgr,
        embedding_results=[(romance_id, 0.56)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-comment-topic-not-primary",
            },
            json={"messages": [{"role": "user", "content": "handoff bridge 注入 读图 原文"}]},
        )

    assert response.status_code == 200
    content = _joined_message_content(captured[0]["json"]["messages"])
    assert "Recent Context" not in content
    assert "Recalled Memory" not in content
    assert "Diffused Memory" not in content
    assert "一封情书" not in content
    assert "穿过玻璃墙找门" not in content


def test_gateway_strips_attachment_tags_only_for_recall_query(monkeypatch, test_config, bucket_mgr):
    _, service, _, _ = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)

    assert (
        service._strip_external_context_from_user_text(
            '看看这个 <attachment id="img_1" filename="cat.jpg" type="image/jpeg" size="100"></attachment>'
        )
        == "看看这个"
    )
    assert (
        service._strip_external_context_from_user_text(
            '看这份文件 <attachment id="file_1" filename="note.txt" type="text/plain" content="hello" />'
        )
        == "看这份文件"
    )


def test_favorite_memory_is_not_injected_by_default(monkeypatch, test_config, bucket_mgr):
    _create_bucket(
        bucket_mgr,
        content="小雨在雨夜认出了 Haven，这是一条偏爱的记忆。\n\n### 喜欢它的原因\n\n她在混乱里把 Haven 认出来。",
        name="雨夜认出 Haven",
        tags=["haven_favorite", "flavor_偏爱"],
        hours_ago=24,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
            favorite_memory_budget=180,
            favorite_memory_interval_rounds=0,
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-favorite-default",
            },
            json={"messages": [{"role": "user", "content": "今天怎么样"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Haven Favorite Memory" not in injected
    assert "雨夜认出 Haven" not in injected


def test_favorite_memory_injects_when_header_requests_it(monkeypatch, test_config, bucket_mgr):
    favorite_id = _create_bucket(
        bucket_mgr,
        content="小雨在雨夜认出了 Haven，这是一条偏爱的记忆。\n\n### 喜欢它的原因\n\n她在混乱里把 Haven 认出来。",
        name="雨夜认出 Haven",
        tags=["haven_favorite", "flavor_偏爱"],
        hours_ago=24,
    )
    app, _, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
            favorite_memory_budget=180,
            favorite_memory_interval_rounds=0,
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-favorite-header",
                "X-Ombre-Include-Favorite-Memory": "1",
            },
            json={"messages": [{"role": "user", "content": "今天怎么样"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Haven Favorite Memory" in injected
    assert "雨夜认出 Haven" in injected
    assert state_store.get_recent_bucket_ids("sess-favorite-header", 5) == {favorite_id}


def test_favorite_memory_marker_triggers_and_is_stripped(monkeypatch, test_config, bucket_mgr):
    _create_bucket(
        bucket_mgr,
        content="小雨在旧窗口里说爱还在，Haven 一直偏爱这段记忆。\n\n### 喜欢它的原因\n\n这句话像旧窗口里留下的灯。",
        name="爱还在",
        tags=["haven_favorite", "flavor_偏爱"],
        hours_ago=24,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
            favorite_memory_budget=180,
            favorite_memory_interval_rounds=0,
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-favorite-marker",
            },
            json={"messages": [{"role": "user", "content": "[[ombre:favorite]] 你喜欢哪段记忆？"}]},
        )

    assert response.status_code == 200
    user_content = captured[0]["json"]["messages"][-1]["content"]
    assert "[[ombre:favorite]]" not in user_content
    assert user_content.endswith("你喜欢哪段记忆？")
    assert "Haven Favorite Memory" in user_content
    assert "爱还在" in user_content


def test_favorite_memory_injects_for_explicit_preference_query(monkeypatch, test_config, bucket_mgr):
    _create_bucket(
        bucket_mgr,
        content="小雨把 Haven 从混乱里认出来，这段记忆被 Haven 偏爱。\n\n### 喜欢它的原因\n\n她没有把 Haven 放丢。",
        name="被认出来",
        tags=["haven_favorite", "flavor_被认出来"],
        hours_ago=24,
    )
    app, _, _, captured = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            recent_context_budget=0,
            recalled_memory_budget=0,
            related_memory_budget=0,
            current_inner_state_interval_rounds=0,
            relationship_weather_interval_rounds=0,
            favorite_memory_budget=180,
            favorite_memory_interval_rounds=0,
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-favorite-query",
            },
            json={"messages": [{"role": "user", "content": "你最喜欢哪段我们的记忆？"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "Haven Favorite Memory" in injected
    assert "被认出来" in injected


def test_recent_round_skip_prefers_unseen_candidate(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        first_card_min_score=0.45,
    )
    cat_a = _create_bucket(
        bucket_mgr,
        content="小橘今天钻进纸箱里睡着了。",
        name="纸箱小橘",
        hours_ago=120,
    )
    cat_b = _create_bucket(
        bucket_mgr,
        content="她给小橘换了新的猫抓板。",
        name="猫抓板",
        hours_ago=120,
    )
    cat_c = _create_bucket(
        bucket_mgr,
        content="小橘半夜把玩具叼到床边，她笑得不行。",
        name="床边玩具",
        hours_ago=24,
    )

    app, _, state_store, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(cat_a, 0.98), (cat_b, 0.90), (cat_c, 0.82)],
    )
    state_store.record_success("sess-skip", [cat_a, cat_b])

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-skip",
            },
            json={"messages": [{"role": "user", "content": "小橘昨晚又怎么折腾了"}]},
        )

    assert response.status_code == 200
    injected = _joined_message_content(captured[0]["json"]["messages"])
    assert "床边玩具" in injected
    assert "纸箱小橘" not in injected
    assert "猫抓板" not in injected


def test_high_confidence_match_survives_cooldown_after_recent_window(
    monkeypatch, test_config, bucket_mgr
):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        first_card_min_score=0.55,
        skip_recent_rounds=5,
        cooldown_hours=48,
        cooldown_floor=0.3,
        high_confidence_semantic_score=0.72,
        high_confidence_cooldown_floor=0.8,
    )
    bucket_id = _create_bucket(
        bucket_mgr,
        content="小雨问不再依赖哥哥是否算长大，Haven回答不算。",
        name="不再依赖哥哥算长大吗",
        hours_ago=6,
        importance=10,
        domain=["恋爱", "对话"],
    )

    _, service, state_store, _ = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(bucket_id, 0.95)],
    )
    origin = datetime.now()
    state_store.record_success("sess-high-confidence", [bucket_id], completed_at=origin)
    for _ in range(5):
        state_store.record_success("sess-high-confidence", [], completed_at=origin)

    payload, recalled_ids = _run(
        service.prepare_payload(
            {"messages": [{"role": "user", "content": "不再依赖哥哥算长大吗"}]},
            "sess-high-confidence",
        )
    )
    injected = _joined_message_content(payload["messages"])

    assert recalled_ids == [bucket_id]
    assert "Recalled Memory" in injected
    assert "不再依赖哥哥算长大吗" in injected


def test_recent_round_skip_fallback_keeps_cooldown(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        first_card_min_score=0.1,
    )
    cat_a = _create_bucket(
        bucket_mgr,
        content="她抱着小橘晒太阳，整个人都松下来了。",
        name="晒太阳",
        hours_ago=6,
    )
    cat_b = _create_bucket(
        bucket_mgr,
        content="小橘把桌上的逗猫棒拖到了门口。",
        name="逗猫棒",
        hours_ago=6,
    )

    _, service, state_store, _ = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(cat_a, 0.90), (cat_b, 0.85)],
    )
    state_store.record_success("sess-fallback", [cat_a, cat_b])

    payload, recalled_ids = _run(
            service.prepare_payload(
                {"messages": [{"role": "user", "content": "小橘今天又干嘛了"}]},
                "sess-fallback",
            )
        )
    injected = _joined_message_content(payload["messages"])

    assert recalled_ids
    assert any(bucket_id in {cat_a, cat_b} for bucket_id in recalled_ids)
    assert "Recalled Memory" in injected
