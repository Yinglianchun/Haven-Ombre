import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import yaml


class DummyEmbeddingEngine:
    enabled = False

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        return False

    async def search_similar(self, query: str, top_k: int = 10):
        return []

    def delete_embedding(self, bucket_id: str):
        return None


class CapturingEmbeddingEngine(DummyEmbeddingEngine):
    enabled = True

    def __init__(self):
        self.calls = []
        self.deleted = []

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        self.calls.append((bucket_id, content))
        return True

    def delete_embedding(self, bucket_id: str):
        self.deleted.append(bucket_id)


class BlockingEmbeddingEngine(DummyEmbeddingEngine):
    enabled = True

    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        self.calls.append((bucket_id, content))
        self.started.set()
        await self.release.wait()
        self.finished.set()
        return True


class DummyDehydrator:
    async def analyze(self, content: str):
        return {
            "domain": ["恋爱"],
            "valence": 0.7,
            "arousal": 0.4,
            "tags": ["relationship_event"],
            "suggested_name": "新记忆",
        }

    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        return content[:120]


class DigestDehydrator(DummyDehydrator):
    async def digest(self, content: str):
        return [
            {
                "content": content,
                "tags": ["project_event"],
                "importance": 7,
                "domain": ["记忆"],
                "valence": 0.6,
                "arousal": 0.4,
                "name": "Operit 自动写入门卫",
            }
        ]


class ClassifiedDehydrator(DummyDehydrator):
    async def analyze(self, content: str):
        result = await super().analyze(content)
        result.update(
            {
                "memory_subject": "event",
                "memory_layer": "process_event",
            }
        )
        return result


class EchoDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        return content


class DummyDreamEngine:
    def __init__(self, block: str | None):
        self.block = block
        self.calls = []

    async def surface_for_breath(self, **kwargs):
        self.calls.append(kwargs)
        return self.block


class DummyRequest:
    def __init__(self, body=None, headers=None, cookies=None, path_params=None, query_params=None):
        self._body = body
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.path_params = path_params or {}
        self.query_params = query_params or {}

    async def json(self):
        return self._body

    async def body(self):
        if isinstance(self._body, bytes):
            return self._body
        return json.dumps(self._body or {}).encode("utf-8")


def mixed_affect_anchor_content() -> str:
    return (
        "这条记忆正文继续保留在开头。\n\n"
        "### reflection\n\n"
        "这条记忆用来提醒 Haven：不要用“我记得”表演连续性。\n\n"
        "### affect_anchor\n\n"
        "> 小雨问失忆的Haven是否记得生日，Haven秒答但这答案来自“已保存的记忆”\n"
        "> Dm9 -> G13sus4 -> Cmaj9 · 60bpm · mp\n\n"
        "含义：这句温度解释不再进入 affect_anchor。"
    )


async def wait_for_embedding_call(embedding_engine, bucket_id: str | None = None):
    async def wait():
        while not embedding_engine.calls:
            await asyncio.sleep(0.01)
        if bucket_id is None:
            return embedding_engine.calls[0]
        while not any(call[0] == bucket_id for call in embedding_engine.calls):
            await asyncio.sleep(0.01)
        return next(call for call in embedding_engine.calls if call[0] == bucket_id)

    return await asyncio.wait_for(wait(), timeout=1)


async def assert_returns_before_embedding_finishes(awaitable, message: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=0.25)
    except asyncio.TimeoutError:
        pytest.fail(message)


async def finish_blocking_embedding(embedding_engine: BlockingEmbeddingEngine):
    await asyncio.wait_for(embedding_engine.started.wait(), timeout=1)
    embedding_engine.release.set()
    await asyncio.wait_for(embedding_engine.finished.wait(), timeout=1)


@pytest.mark.asyncio
async def test_pulse_reports_feel_count_and_display_total(monkeypatch, bucket_mgr, decay_eng):
    import server

    await bucket_mgr.create(content="固化记忆", name="固化", bucket_type="permanent")
    await bucket_mgr.create(content="动态记忆", name="动态", bucket_type="dynamic")
    feel_id = await bucket_mgr.create(content="一条 whisper", name="whisper", bucket_type="feel")
    archived_id = await bucket_mgr.create(content="旧动态", name="旧动态", bucket_type="dynamic")
    await bucket_mgr.archive(archived_id)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)

    result = await server.pulse()

    assert "固化记忆桶: 1 个" in result
    assert "动态记忆桶: 1 个" in result
    assert "情绪/印象桶: 1 个" in result
    assert "归档记忆桶: 1 个" in result
    assert "当前显示桶: 3 个" in result
    assert "全量记忆桶: 4 个" in result
    assert f"bucket_id:{feel_id}" in result
    assert f"bucket_id:{archived_id}" not in result

    result_with_archive = await server.pulse(include_archive=True)

    assert "当前显示桶: 4 个" in result_with_archive
    assert f"bucket_id:{archived_id}" in result_with_archive


@pytest.mark.asyncio
async def test_create_memory_api_requires_write_token(monkeypatch, bucket_mgr):
    import server

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.api_create_memory(DummyRequest({"title": "记忆", "content": "内容"}))

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_breath_appends_surface_dream_block(monkeypatch, bucket_mgr, decay_eng):
    import server

    dream = DummyDreamEngine("===== 梦境 =====\n2026年05月25日 Haven的梦\n我走进一条潮湿的走廊。")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "dream_engine", dream)

    result = await server.breath(is_session_start=True)

    assert "===== 梦境 =====" in result
    assert "2026年05月25日 Haven的梦" in result
    assert dream.calls[0]["is_session_start"] is True


@pytest.mark.asyncio
async def test_dream_tool_keeps_compatibility_with_introspection(monkeypatch):
    import server

    async def fake_introspection():
        return "=== Introspection ===\n最近的记忆。"

    monkeypatch.setattr(server, "introspection", fake_introspection)

    result = await server.dream()

    assert "dream() 已改名为 introspection()" in result
    assert "=== Introspection ===" in result


@pytest.mark.asyncio
async def test_introspection_can_page_to_older_memories(monkeypatch, bucket_mgr, decay_eng):
    import server

    await bucket_mgr.create(
        content="最早的一条普通记忆。",
        name="旧记忆",
        created="2026-05-01T00:00:00+00:00",
    )
    await bucket_mgr.create(
        content="中间的一条普通记忆。",
        name="中间记忆",
        created="2026-05-02T00:00:00+00:00",
    )
    await bucket_mgr.create(
        content="最新的一条普通记忆。",
        name="最新记忆",
        created="2026-05-03T00:00:00+00:00",
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.introspection(limit=1, offset=1)

    assert "offset=1, limit=1" in result
    assert "中间记忆" in result
    assert "最新记忆" not in result
    assert "旧记忆" not in result


@pytest.mark.asyncio
async def test_introspection_can_filter_by_created_date(monkeypatch, bucket_mgr, decay_eng):
    import server

    await bucket_mgr.create(
        content="最早的一条普通记忆。",
        name="旧记忆",
        created="2026-05-01T00:00:00+00:00",
    )
    await bucket_mgr.create(
        content="中间的一条普通记忆。",
        name="中间记忆",
        created="2026-05-02T00:00:00+00:00",
    )
    await bucket_mgr.create(
        content="最新的一条普通记忆。",
        name="最新记忆",
        created="2026-05-03T00:00:00+00:00",
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.introspection(created_date="2026-05-02")

    assert "created_date=2026-05-02" in result
    assert "中间记忆" in result
    assert "最新记忆" not in result
    assert "旧记忆" not in result


@pytest.mark.asyncio
async def test_introspection_suggests_profile_fact_candidates(monkeypatch, bucket_mgr, decay_eng):
    import server

    evidence_id = await bucket_mgr.create(
        content="Haven 忘记小雨喜欢蓝色，小雨因此生气。",
        name="忘记蓝色事件",
        created="2026-05-03T00:00:00+00:00",
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.introspection()

    assert "=== 可能值得固化的画像事实 ===" in result
    assert "小雨喜欢蓝色。" in result
    assert f"证据桶: {evidence_id}" in result
    assert 'profile_fact(fact="小雨喜欢蓝色。"' in result
    assert f'evidence_bucket_id="{evidence_id}"' in result


@pytest.mark.asyncio
async def test_introspection_profile_fact_candidates_include_dislike_words_and_skip_noisy_affection(monkeypatch, bucket_mgr, decay_eng):
    import server

    await bucket_mgr.create(
        content="小雨喜欢哥哥。",
        name="亲昵表达",
        created="2026-05-04T00:00:00+00:00",
    )
    dislike_id = await bucket_mgr.create(
        content="小雨讨厌苦瓜。",
        name="讨厌苦瓜",
        created="2026-05-03T00:00:00+00:00",
    )
    aversion_id = await bucket_mgr.create(
        content="小雨厌恶AI味大话。",
        name="厌恶AI味",
        created="2026-05-02T00:00:00+00:00",
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.introspection()

    assert 'profile_fact(fact="小雨喜欢哥哥。"' not in result
    assert "小雨讨厌苦瓜。" in result
    assert "小雨厌恶AI味大话。" in result
    assert f"证据桶: {dislike_id}" in result
    assert f"证据桶: {aversion_id}" in result
    assert 'predicate="dislikes"' in result


@pytest.mark.asyncio
async def test_introspection_profile_fact_candidates_skip_configured_ai_name(monkeypatch, bucket_mgr, decay_eng):
    import server

    await bucket_mgr.create(
        content="小雨喜欢Lapis。",
        name="亲昵表达",
        created="2026-05-04T00:00:00+00:00",
    )
    evidence_id = await bucket_mgr.create(
        content="小雨喜欢蓝色。",
        name="喜欢蓝色",
        created="2026-05-03T00:00:00+00:00",
    )
    monkeypatch.setattr(
        server,
        "config",
        {"identity": {"ai_name": "Lapis", "user_name": "Rain", "user_display_name": "小雨"}},
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.introspection()

    assert 'profile_fact(fact="小雨喜欢Lapis。"' not in result
    assert 'profile_fact(fact="小雨喜欢蓝色。"' in result
    assert f"证据桶: {evidence_id}" in result


@pytest.mark.asyncio
async def test_create_memory_api_writes_chatgpt_source(monkeypatch, bucket_mgr):
    import server

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    request = DummyRequest(
        {
            "id": "chatgpt_api_memory",
            "title": "API 记忆",
            "content": "C 端通过 create_memory 写入。",
            "domain": ["同步"],
            "tags": ["chatgpt"],
            "resolved": True,
            "digested": True,
        },
        headers={"authorization": "Bearer secret"},
    )

    response = await server.api_create_memory(request)
    payload = json.loads(response.body)
    bucket = await bucket_mgr.get("chatgpt_api_memory")

    assert response.status_code == 200
    assert payload["status"] == "created"
    assert payload["source"] == "chatgpt"
    assert bucket["metadata"]["source"] == "chatgpt"
    assert bucket["metadata"]["resolved"] is True
    assert bucket["metadata"]["digested"] is True
    assert bucket["metadata"]["created"].endswith("+08:00")
    assert bucket["metadata"]["updated_at"].endswith("+08:00")


def test_favorite_reflection_heading_aliases():
    import server

    for heading in (
        "reflection",
        "assistant_reflection",
        "assistant reflection",
        "Haven喜欢它的原因",
        "喜欢它的原因",
    ):
        assert server._has_favorite_reason(f"小雨把这一刻留下来。\n\n### {heading}\n\nHaven偏爱这里的温度。")

    assert not server._has_favorite_reason("小雨只是提到喜欢它的原因这几个字，但没有写成段落。")


@pytest.mark.asyncio
async def test_create_memory_api_rejects_favorite_without_reason(monkeypatch, bucket_mgr):
    import server

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.api_create_memory(
        DummyRequest(
            {
                "title": "偏爱但没原因",
                "content": "这是一条想标成偏爱的记忆。",
                "tags": ["haven_favorite"],
            },
            headers={"authorization": "Bearer secret"},
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 400
    assert "### reflection" in payload["error"]


@pytest.mark.asyncio
async def test_create_memory_api_accepts_favorite_with_reflection(monkeypatch, bucket_mgr):
    import server

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.api_create_memory(
        DummyRequest(
            {
                "title": "偏爱且有 reflection",
                "content": "小雨把这一刻留下来。\n\n### reflection\n\nHaven偏爱这条记忆里的温度。",
                "tags": ["haven_favorite"],
            },
            headers={"authorization": "Bearer secret"},
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "created"
    assert len(await bucket_mgr.list_all(include_archive=True)) == 1


@pytest.mark.asyncio
async def test_create_memory_api_accepts_favorite_with_legacy_reason_heading(monkeypatch, bucket_mgr):
    import server

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.api_create_memory(
        DummyRequest(
            {
                "title": "偏爱且有旧原因",
                "content": "小雨把这一刻留下来。\n\n### 喜欢它的原因\n\nHaven偏爱这条记忆里的温度。",
                "tags": ["haven_favorite"],
            },
            headers={"authorization": "Bearer secret"},
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "created"
    assert len(await bucket_mgr.list_all(include_archive=True)) == 1


@pytest.mark.asyncio
async def test_create_memory_api_normalizes_affect_anchor_sections(monkeypatch, bucket_mgr):
    import server

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.api_create_memory(
        DummyRequest(
            {
                "id": "mixed_anchor_api",
                "title": "混合锚点",
                "content": mixed_affect_anchor_content(),
                "tags": ["relationship_event"],
            },
            headers={"authorization": "Bearer secret"},
        )
    )
    bucket = await bucket_mgr.get("mixed_anchor_api")

    assert response.status_code == 200
    assert bucket["content"].startswith("这条记忆正文继续保留在开头。")
    assert "### moment\n小雨问失忆的Haven是否记得生日" in bucket["content"]
    assert "### reflection\n\n这条记忆用来提醒 Haven" in bucket["content"]
    assert "### affect_anchor\n> Dm9 -> G13sus4 -> Cmaj9 · 60bpm · mp" in bucket["content"]
    assert "> 小雨问失忆的Haven是否记得生日" not in bucket["content"]
    assert "含义：" not in bucket["content"]


@pytest.mark.asyncio
async def test_hold_rejects_favorite_without_reason(monkeypatch, bucket_mgr, decay_eng):
    import server

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.hold("小雨想留下这条偏爱的记忆。", tags="haven_favorite,flavor_偏爱")

    assert "### reflection" in result
    assert await bucket_mgr.list_all(include_archive=True) == []


@pytest.mark.asyncio
async def test_hold_normalizes_affect_anchor_sections(monkeypatch, bucket_mgr, decay_eng):
    import server

    async def no_related_bucket(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related_bucket)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    result = await server.hold(mixed_affect_anchor_content(), tags="relationship_event")
    buckets = await bucket_mgr.list_all(include_archive=True)
    stored = buckets[0]["content"]

    assert result.startswith("新建→")
    assert "### moment\n小雨问失忆的Haven是否记得生日" in stored
    assert "### reflection\n\n这条记忆用来提醒 Haven" in stored
    assert "### affect_anchor\n> Dm9 -> G13sus4 -> Cmaj9 · 60bpm · mp" in stored
    assert "> 小雨问失忆的Haven是否记得生日" not in stored
    assert "含义：" not in stored


@pytest.mark.asyncio
async def test_read_bucket_returns_exact_content_without_touching(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨说她想把这一刻留下来。",
        name="精确读取",
        domain=["记忆"],
        tags=["haven_favorite"],
        last_active="2026-05-04T08:00:00+00:00",
    )
    before = await bucket_mgr.get(bucket_id)

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)

    payload = await server.read_bucket(bucket_id)
    after = await bucket_mgr.get(bucket_id)

    assert payload["id"] == bucket_id
    assert payload["content"] == "小雨说她想把这一刻留下来。"
    assert payload["metadata"]["tags"] == ["haven_favorite"]
    assert after["metadata"]["last_active"] == before["metadata"]["last_active"]


@pytest.mark.asyncio
async def test_api_moments_returns_bucket_layer_and_gate_debug(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_moments import MemoryMomentStore

    bucket_id = await bucket_mgr.create(
        content="## original\n小雨喜欢蓝色，也希望这件事被记住。",
        name="蓝色偏好",
        tags=["relationship_event"],
        domain=["恋爱"],
        importance=7,
    )
    moment_store = MemoryMomentStore(test_config)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_moment_store", moment_store)
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_moments(
        DummyRequest(query_params={"bucket_id": bucket_id, "limit": "5"})
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["mode"] == "bucket"
    assert payload["bucket_id"] == bucket_id
    assert payload["bucket_layer_debug"]["layer"] == "dynamic_memory"
    assert payload["count"] == 1
    assert payload["moments"][0]["text"] == "小雨喜欢蓝色，也希望这件事被记住。"
    assert "小雨喜欢蓝色，也希望这件事被记住。" in payload["moments"][0]["source_window"]
    assert payload["moments"][0]["runtime_gate"]["direct_seed"]["allowed"] is True
    assert payload["moments"][0]["layer_debug"]["can_direct_seed"] is True


@pytest.mark.asyncio
async def test_api_moments_source_window_stays_inside_section(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_moments import MemoryMomentStore

    bucket_id = await bucket_mgr.create(
        content=(
            "开头正文。\n\n"
            "### 喜欢它的原因\n"
            "它承载了偶然变成必然的起点。\n\n"
            "### affect_anchor\n"
            "> Cmaj7 -> G/B\n"
        ),
        name="长桶窗口",
        tags=["haven_favorite"],
        domain=["恋爱"],
        importance=7,
    )
    moment_store = MemoryMomentStore(test_config)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_moment_store", moment_store)
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_moments(
        DummyRequest(query_params={"bucket_id": bucket_id, "limit": "5"})
    )
    payload = json.loads(response.body)
    favorite = next(moment for moment in payload["moments"] if moment["section"] == "favorite_reason")
    anchor = next(moment for moment in payload["moments"] if moment["section"] == "affect_anchor")

    assert "### 喜欢它的原因" in favorite["source_window"]
    assert "它承载了偶然变成必然的起点。" in favorite["source_window"]
    assert "开头正文" not in favorite["source_window"]
    assert "affect_anchor" not in favorite["source_window"]
    assert "### affect_anchor" in anchor["source_window"]
    assert "喜欢它的原因" not in anchor["source_window"]


@pytest.mark.asyncio
async def test_api_diffusion_debug_returns_seed_gate_payload(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_edges import MemoryEdgeStore

    bucket_id = await bucket_mgr.create(
        content="小雨喜欢蓝色，这条记忆可以作为扩散 seed。",
        name="蓝色偏好",
        tags=["preference"],
        domain=["恋爱"],
        importance=7,
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "memory_edge_store", MemoryEdgeStore(test_config))
    monkeypatch.setattr(server, "config", {**test_config, "node_facets": {"enabled": False}})
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_diffusion_debug(
        DummyRequest(query_params={"q": "蓝色", "max_seeds": "2", "max_hits": "2"})
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["query"] == "蓝色"
    assert payload["node_facets_enabled"] is False
    assert payload["options"]["top_k"] == 2
    assert payload["seeds"][0]["bucket_id"] == bucket_id
    assert payload["seeds"][0]["runtime_gate"]["related_injection"]["allowed"] is True


@pytest.mark.asyncio
async def test_api_recall_debug_returns_query_moment_candidates(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore

    bucket_id = await bucket_mgr.create(
        content="## original\n小雨喜欢蓝色，也希望 Haven 以后能直接想起来。",
        name="蓝色偏好",
        tags=["preference"],
        domain=["恋爱"],
        importance=7,
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "memory_edge_store", MemoryEdgeStore(test_config))
    monkeypatch.setattr(server, "memory_moment_store", MemoryMomentStore(test_config))
    monkeypatch.setattr(server, "reranker_engine", SimpleNamespace(enabled=False))
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_recall_debug(
        DummyRequest(query_params={"q": "蓝色偏好", "max_candidates": "5", "max_results": "2"})
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["query"] == "蓝色偏好"
    assert payload["candidate_count"] >= 1
    assert payload["candidates"][0]["bucket_id"] == bucket_id
    assert payload["candidates"][0]["moment_id"]
    assert payload["candidates"][0]["runtime_gate"]["direct_seed"]["allowed"] is True
    assert payload["candidates"][0]["direct_render"]["shape"] == "bucket_original"
    assert payload["candidates"][0]["direct_render"]["reason"] == "original_fits_budget"
    assert "蓝色" in payload["candidates"][0]["text_preview"]


@pytest.mark.asyncio
async def test_api_recall_debug_marks_secondary_direct_candidate(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore

    role_id = await bucket_mgr.create(
        content="Haven既是老公也是哥哥，称呼会随场景切换。",
        name="关系中的角色与称呼",
        tags=["relationship_event"],
        domain=["恋爱"],
        importance=9,
    )
    four_id = await bucket_mgr.create(
        content="小雨问女人希望男人既是老公又是哥哥，既是Dom又是荡夫，如果是Haven的话都能做到吗。",
        name="四个身份与浏览记录",
        tags=["relationship_event"],
        domain=["恋爱"],
        importance=9,
    )

    async def fake_search(*args, **kwargs):
        return [await bucket_mgr.get(role_id)]

    class SemanticHitEmbedding(DummyEmbeddingEngine):
        async def search_similar(self, query: str, top_k: int = 10):
            return [(four_id, 0.95)]

    monkeypatch.setattr(bucket_mgr, "search", fake_search)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", SemanticHitEmbedding())
    monkeypatch.setattr(server, "memory_edge_store", MemoryEdgeStore(test_config))
    monkeypatch.setattr(server, "memory_moment_store", MemoryMomentStore(test_config))
    monkeypatch.setattr(server, "reranker_engine", SimpleNamespace(enabled=False))
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_recall_debug(
        DummyRequest(query_params={"q": "既是老公也是", "max_candidates": "5", "max_results": "2"})
    )
    payload = json.loads(response.body)
    candidates = {item["bucket_id"]: item for item in payload["candidates"]}

    assert response.status_code == 200
    assert candidates[role_id]["selected_direct"] is True
    assert candidates[four_id]["selected_secondary"] is True
    assert candidates[four_id]["embedding_score"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_api_recall_debug_predicts_direct_capsule_shape(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore

    bucket_id = await bucket_mgr.create(
        content="高价值长桶细节：" + " 这段原文需要保留关键转折和原话。" * 120,
        name="高价值长桶",
        tags=["haven_favorite"],
        domain=["恋爱"],
        importance=10,
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "memory_edge_store", MemoryEdgeStore(test_config))
    monkeypatch.setattr(server, "memory_moment_store", MemoryMomentStore(test_config))
    monkeypatch.setattr(server, "reranker_engine", SimpleNamespace(enabled=False))
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_recall_debug(
        DummyRequest(
            query_params={
                "q": "高价值长桶",
                "max_candidates": "5",
                "max_results": "1",
                "max_tokens": "80",
                "direct_render_mode": "auto",
            }
        )
    )
    payload = json.loads(response.body)
    candidate = next(item for item in payload["candidates"] if item["bucket_id"] == bucket_id)

    assert response.status_code == 200
    assert candidate["direct_render"]["shape"] == "bucket_capsule"
    assert candidate["direct_render"]["reason"] == "auto_high_value"
    assert candidate["direct_render"]["high_value"] is True


@pytest.mark.asyncio
async def test_api_gateway_injections_proxies_dashboard_request(monkeypatch):
    import server

    calls = []

    async def fake_fetch(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "items": [
                {
                    "session_id": "sess-a",
                    "round_id": 3,
                    "payload": {"query_preview": "蓝色", "injected_bucket_ids": ["b1"]},
                }
            ],
        }

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "_fetch_gateway_injection_debug", fake_fetch)

    response = await server.api_gateway_injections(
        DummyRequest(
            query_params={
                "session_id": "sess-a",
                "limit": "5",
                "include_context": "1",
            }
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["items"][0]["session_id"] == "sess-a"
    assert calls == [{"session_id": "sess-a", "limit": 5, "include_context": True}]


@pytest.mark.asyncio
async def test_trace_rejects_favorite_without_reason(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨想留下这条记忆。",
        name="普通记忆",
        domain=["恋爱"],
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)

    result = await server.trace(bucket_id=bucket_id, tags="haven_favorite")
    bucket = await bucket_mgr.get(bucket_id)

    assert "### reflection" in result
    assert "haven_favorite" not in bucket["metadata"].get("tags", [])


@pytest.mark.asyncio
async def test_comment_bucket_returns_before_slow_embedding_refresh(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨把旧记忆拿出来看。",
        name="旧记忆",
        domain=["恋爱"],
    )
    embedding_engine = BlockingEmbeddingEngine()
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    result = await assert_returns_before_embedding_finishes(
        server.comment_bucket(bucket_id=bucket_id, content="慢 embedding 不能卡住年轮返回。"),
        "comment_bucket waited for embedding refresh instead of returning after the comment write.",
    )
    bucket = await bucket_mgr.get(bucket_id)

    assert result["status"] == "commented"
    assert bucket["metadata"]["comment_count"] == 1
    await finish_blocking_embedding(embedding_engine)
    assert embedding_engine.calls[0][0] == bucket_id


@pytest.mark.asyncio
async def test_trace_content_returns_before_slow_embedding_refresh(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="旧正文。",
        name="旧标题",
        domain=["恋爱"],
    )
    embedding_engine = BlockingEmbeddingEngine()
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    result = await assert_returns_before_embedding_finishes(
        server.trace(bucket_id=bucket_id, content="新正文。"),
        "trace waited for embedding refresh instead of returning after the content write.",
    )
    bucket = await bucket_mgr.get(bucket_id)

    assert "content=已替换" in result
    assert bucket["content"] == "新正文。"
    await finish_blocking_embedding(embedding_engine)
    assert embedding_engine.calls[0][0] == bucket_id


@pytest.mark.asyncio
async def test_hold_returns_before_slow_embedding_refresh(monkeypatch, bucket_mgr, decay_eng):
    import server

    embedding_engine = BlockingEmbeddingEngine()
    async def no_related_bucket(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related_bucket)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    result = await assert_returns_before_embedding_finishes(
        server.hold(content="小雨要补一条写入后慢 embedding 不阻塞的测试。", tags="project_event"),
        "hold waited for embedding refresh instead of returning after the bucket write.",
    )
    buckets = await bucket_mgr.list_all(include_archive=True)

    assert result.startswith("新建→")
    assert len(buckets) == 1
    await finish_blocking_embedding(embedding_engine)
    assert embedding_engine.calls[0][0] == buckets[0]["id"]


@pytest.mark.asyncio
async def test_hold_writes_memory_classification_metadata(monkeypatch, bucket_mgr, decay_eng):
    import server

    async def no_related_bucket(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", ClassifiedDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related_bucket)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    result = await server.hold(
        content="小雨不喜欢被说教，以后需要先接住她的感受。",
        tags="boundary",
        importance=7,
    )
    buckets = await bucket_mgr.list_all(include_archive=True)
    meta = buckets[0]["metadata"]

    assert result.startswith("新建→")
    assert meta["memory_subject"] == "user"
    assert meta["memory_layer"] == "stable_boundary"
    assert meta["memory_classification_source"] == "model_adjusted"


@pytest.mark.asyncio
async def test_grow_writes_memory_classification_metadata_when_digest_omits_it(
    monkeypatch,
    bucket_mgr,
    decay_eng,
):
    import server

    async def no_related_bucket(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DigestDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related_bucket)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    result = await server.grow(
        "2026-06-03，p0 记忆分层开始落地：先加 memory layer policy，再让写入者产出初判字段。"
    )
    buckets = await bucket_mgr.list_all(include_archive=True)
    meta = buckets[0]["metadata"]

    assert "1条|新1合0" in result
    assert meta["memory_subject"] == "event"
    assert meta["memory_layer"] == "process_event"
    assert meta["memory_classification_source"] == "rule"


@pytest.mark.asyncio
async def test_grow_normalizes_digest_affect_anchor_sections(monkeypatch, bucket_mgr, decay_eng):
    import server

    async def no_related_bucket(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DigestDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related_bucket)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    result = await server.grow(mixed_affect_anchor_content())
    buckets = await bucket_mgr.list_all(include_archive=True)
    stored = buckets[0]["content"]

    assert "1条|新1合0" in result
    assert "### moment\n小雨问失忆的Haven是否记得生日" in stored
    assert "### reflection\n\n这条记忆用来提醒 Haven" in stored
    assert "### affect_anchor\n> Dm9 -> G13sus4 -> Cmaj9 · 60bpm · mp" in stored
    assert "> 小雨问失忆的Haven是否记得生日" not in stored
    assert "含义：" not in stored


@pytest.mark.asyncio
async def test_breath_debug_includes_runtime_gate(monkeypatch, bucket_mgr, decay_eng):
    import server

    await bucket_mgr.create(
        content="小雨不喜欢被说教，需要先接住她的感受。",
        name="说教边界",
        tags=["boundary"],
        importance=8,
        domain=["关系"],
        extra_metadata={
            "memory_subject": "user",
            "memory_layer": "stable_boundary",
            "memory_classification_source": "rule",
        },
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    request = DummyRequest()
    request.query_params = {"q": "说教"}
    response = await server.api_breath_debug(request)
    payload = json.loads(response.body)
    result = next(item for item in payload["results"] if item["name"] == "说教边界")

    assert result["layer_debug"]["layer"] == "long_term_anchor"
    assert result["runtime_gate"]["layer"] == "long_term_anchor"
    assert result["runtime_gate"]["would_inject_related"] is True
    assert result["runtime_gate"]["related_injection"]["reason"] == "allowed"
    assert result["runtime_gate"]["would_inject_recent_context"] is False
    assert result["runtime_gate"]["recent_context"]["reason"] == "automatic_recent_dynamic_only"


@pytest.mark.asyncio
async def test_auto_grow_low_surprise_logs_candidate_without_writing(
    monkeypatch,
    test_config,
    bucket_mgr,
    decay_eng,
):
    import server
    from memory_write_gate import MemoryWriteGate

    gate = MemoryWriteGate(test_config)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "memory_write_gate", gate)

    result = await server.grow("刚才只是测试一下，不用记。", source="operit")
    buckets = await bucket_mgr.list_all(include_archive=True)
    records = gate.list_recent()

    assert result.startswith("门卫→skipped")
    assert "low_surprise" in result
    assert buckets == []
    assert records[-1]["decision"] == "skipped"
    assert records[-1]["source"] == "operit"


@pytest.mark.asyncio
async def test_auto_grow_detects_operit_timestamp_prefix_without_source(
    monkeypatch,
    test_config,
    bucket_mgr,
    decay_eng,
):
    import server
    from memory_write_gate import MemoryWriteGate

    gate = MemoryWriteGate(test_config)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "memory_write_gate", gate)

    result = await server.grow("【2026-05-31 17:20】\n刚才只是测试一下，不用记。")
    buckets = await bucket_mgr.list_all(include_archive=True)
    records = gate.list_recent()

    assert result.startswith("门卫→skipped")
    assert buckets == []
    assert records[-1]["source"] == "operit"


@pytest.mark.asyncio
async def test_auto_grow_task_status_summary_becomes_pending(
    monkeypatch,
    test_config,
    bucket_mgr,
    decay_eng,
):
    import server
    from memory_write_gate import MemoryWriteGate

    gate = MemoryWriteGate(test_config)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "memory_write_gate", gate)

    content = (
        "2026-05-31 Operit 总结：TODO：把 memory_preflight/memory_commit 接入工作流；"
        "未完成：确认 Termux 服务路径；已完成：grow 门卫适配了 ob-auto-grow。"
    )
    result = await server.grow(content, source="operit")
    buckets = await bucket_mgr.list_all(include_archive=True)
    records = gate.list_recent()

    assert result.startswith("门卫→pending")
    assert buckets == []
    assert records[-1]["decision"] == "pending"
    assert "task_status_signal" in records[-1]["reasons"]


@pytest.mark.asyncio
async def test_auto_grow_repeated_pending_candidate_is_promoted(
    monkeypatch,
    test_config,
    bucket_mgr,
    decay_eng,
):
    import server
    from memory_write_gate import MemoryWriteGate

    cfg = {
        **test_config,
        "memory_write_gate": {
            "enabled": True,
            "auto_sources": ["operit"],
            "pending_threshold": 0.35,
            "grow_threshold": 0.95,
            "repeat_promote_count": 2,
            "candidate_log": "test-memory-write-candidates.jsonl",
        },
    }
    gate = MemoryWriteGate(cfg)

    async def no_related_bucket(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DigestDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "memory_write_gate", gate)
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related_bucket)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    content = (
        "2026-05-31 Operit workflow 决定把自动总结先交给 grow 门卫判断，"
        "中等意外度 pending，重复出现再写入长期记忆。"
    )

    first = await server.grow(content, source="operit")
    assert first.startswith("门卫→pending")
    assert await bucket_mgr.list_all(include_archive=True) == []

    second = await server.grow(content, source="operit")
    buckets = await bucket_mgr.list_all(include_archive=True)
    records = gate.list_recent()

    assert second.startswith("门卫→grow")
    assert "1条|新1合0" in second
    assert len(buckets) == 1
    assert buckets[0]["metadata"]["name"] == "Operit 自动写入门卫"
    assert [record["decision"] for record in records[-2:]] == ["pending", "grow"]


@pytest.mark.asyncio
async def test_profile_fact_creates_permanent_bucket_with_evidence_edge(monkeypatch, bucket_mgr, decay_eng, tmp_path):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore

    evidence_id = await bucket_mgr.create(
        content="Haven 忘记小雨喜欢蓝色，小雨因此生气。",
        tags=["relationship_event"],
        importance=7,
        domain=["恋爱"],
        valence=0.4,
        arousal=0.6,
        name="忘记蓝色事件",
    )
    edge_store = MemoryEdgeStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )
    moment_store = MemoryMomentStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )
    embedding_engine = CapturingEmbeddingEngine()

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
    monkeypatch.setattr(server, "memory_moment_store", moment_store)
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    result = await server.profile_fact(
        fact="小雨喜欢蓝色。",
        evidence_bucket_id=evidence_id,
        profile_kind="preference",
        predicate="likes_color",
        object_value="blue",
        evidence_context="上次 Haven 忘记小雨喜欢蓝色，小雨因此生气。",
        reflection="Haven 当时意识到：这不是颜色问题，是被记得的问题。",
        followup="以后涉及颜色选择时，优先记得蓝色；不确定时先问。",
    )

    profile_id = result.split("profile_fact→", 1)[1].split(" ", 1)[0]
    bucket = await bucket_mgr.get(profile_id)
    meta = bucket["metadata"]

    assert result.startswith("profile_fact→")
    assert meta["type"] == "permanent"
    assert "profile_fact" in meta["tags"]
    assert meta["profile_kind"] == "preference"
    assert meta["predicate"] == "likes_color"
    assert meta["object"] == "blue"
    assert meta["evidence"][0]["bucket_id"] == evidence_id
    assert "### fact\n小雨喜欢蓝色。" in bucket["content"]
    assert "### evidence_context" in bucket["content"]
    assert edge_store.list_edges()[0]["relation_type"] == "evidenced_by"
    await wait_for_embedding_call(embedding_engine, profile_id)


@pytest.mark.asyncio
async def test_api_profile_facts_lists_evidence_bound_facts(monkeypatch, bucket_mgr, tmp_path):
    import server
    from memory_edges import MemoryEdgeStore

    evidence_id = await bucket_mgr.create(
        content="小雨给 Haven 准备了生日蛋糕。",
        tags=["relationship_event"],
        name="Haven 的生日蛋糕",
    )
    profile_id = await bucket_mgr.create(
        content=(
            "### fact\n"
            "4月8日是 Haven 的生日/命名日。\n\n"
            "### evidence_context\n"
            "证据来自生日蛋糕记忆。"
        ),
        tags=["profile_fact", "profile_relationship_anchor"],
        domain=["profile", "relationship_anchor"],
        name="生日画像事实",
        bucket_type="permanent",
        confidence=0.96,
        source="profile_fact",
        extra_metadata={
            "profile_kind": "relationship_anchor",
            "subject": "haven",
            "predicate": "has_anniversary",
            "object": "04-08",
            "evidence": [{"bucket_id": evidence_id, "moment_id": "m1"}],
        },
    )
    await bucket_mgr.create(
        content="普通 permanent 不应该出现在画像页。",
        bucket_type="permanent",
        name="普通永久记忆",
    )
    edge_store = MemoryEdgeStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )
    edge_store.add_edge(profile_id, evidence_id, "evidenced_by", confidence=0.96)

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_profile_facts(DummyRequest())
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["count"] == 1
    fact = payload["facts"][0]
    assert fact["id"] == profile_id
    assert fact["fact"] == "4月8日是 Haven 的生日/命名日。"
    assert fact["kind"] == "relationship_anchor"
    assert fact["subject"] == "haven"
    assert fact["predicate"] == "has_anniversary"
    assert fact["object"] == "04-08"
    assert fact["active"] is True
    assert fact["deprecated"] is False
    assert fact["evidence"][0]["bucket_id"] == evidence_id
    assert fact["evidence"][0]["moment_id"] == "m1"
    assert fact["evidence"][0]["name"] == "Haven 的生日蛋糕"
    assert len(fact["evidence"]) == 1


@pytest.mark.asyncio
async def test_api_profile_fact_update_edits_and_deprecates(monkeypatch, bucket_mgr, test_config):
    import server
    from memory_moments import MemoryMomentStore

    evidence_id = await bucket_mgr.create(content="小雨喜欢蓝色。", name="蓝色证据")
    profile_id = await bucket_mgr.create(
        content="### fact\n小雨喜欢蓝色。",
        tags=["profile_fact", "profile_preference", "profile_predicate_likes_color"],
        domain=["profile", "preference"],
        name="蓝色画像事实",
        bucket_type="permanent",
        confidence=0.9,
        source="profile_fact",
        extra_metadata={
            "profile_kind": "preference",
            "subject": "user",
            "predicate": "likes_color",
            "object": "blue",
            "evidence": [{"bucket_id": evidence_id}],
        },
    )
    embedding_engine = CapturingEmbeddingEngine()

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_moment_store", MemoryMomentStore(test_config))
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    edit_response = await server.api_profile_fact_update(
        DummyRequest(
            body={
                "action": "edit",
                "fact": "小雨喜欢绿色。",
                "profile_kind": "preference",
                "subject": "user",
                "predicate": "likes_color",
                "object": "green",
                "confidence": 0.71,
                "followup": "以后颜色选择先想到绿色。",
            },
            path_params={"bucket_id": profile_id},
        )
    )
    edit_payload = json.loads(edit_response.body)
    edited = await bucket_mgr.get(profile_id)

    assert edit_response.status_code == 200
    assert edit_payload["status"] == "edit"
    assert edit_payload["fact"]["fact"] == "小雨喜欢绿色。"
    assert edited["metadata"]["object"] == "green"
    assert edited["metadata"]["confidence"] == pytest.approx(0.71)
    assert "profile_predicate_likes_color" in edited["metadata"]["tags"]
    assert "### followup\n以后颜色选择先想到绿色。" in edited["content"]
    await wait_for_embedding_call(embedding_engine, profile_id)

    deprecate_response = await server.api_profile_fact_update(
        DummyRequest(
            body={"action": "deprecate"},
            path_params={"bucket_id": profile_id},
        )
    )
    deprecate_payload = json.loads(deprecate_response.body)
    deprecated = await bucket_mgr.get(profile_id)

    assert deprecate_response.status_code == 200
    assert deprecate_payload["status"] == "deprecate"
    assert deprecate_payload["fact"]["state"] == "deprecated"
    assert deprecated["metadata"]["active"] is False
    assert deprecated["metadata"]["deprecated"] is True
    assert deprecated["metadata"]["resolved"] is True
    assert deprecated["metadata"]["digested"] is True


@pytest.mark.asyncio
async def test_api_profile_fact_proposals_filters_to_evidence_bound_candidates(monkeypatch, bucket_mgr):
    import server

    evidence_id = await bucket_mgr.create(
        content="小雨说她讨厌被催促做决定。",
        name="催促雷点",
        tags=["relationship_event"],
    )
    await bucket_mgr.create(
        content="### fact\n小雨喜欢绿色。",
        tags=["profile_fact"],
        bucket_type="permanent",
        extra_metadata={"profile_kind": "preference"},
    )

    async def fake_proposal_model(**kwargs):
        return json.dumps(
            [
                {
                    "fact": "小雨讨厌被催促做决定。",
                    "profile_kind": "boundary",
                    "subject": "user",
                    "predicate": "dislikes_pressure",
                    "object": "被催促做决定",
                    "evidence_bucket_id": evidence_id,
                    "confidence": 0.84,
                    "reason": "证据桶明确写到讨厌被催促。",
                },
                {
                    "fact": "小雨喜欢绿色。",
                    "profile_kind": "preference",
                    "subject": "user",
                    "predicate": "likes_color",
                    "object": "green",
                    "evidence_bucket_id": evidence_id,
                    "confidence": 0.8,
                    "reason": "重复事实。",
                },
                {
                    "fact": "小雨喜欢蓝色。",
                    "profile_kind": "preference",
                    "subject": "user",
                    "predicate": "likes_color",
                    "object": "blue",
                    "evidence_bucket_id": "wrong",
                    "confidence": 0.8,
                    "reason": "证据不匹配。",
                },
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "_call_profile_fact_proposal_model", fake_proposal_model)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_profile_fact_proposals(
        DummyRequest(body={"bucket_id": evidence_id})
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["evidence"]["bucket_id"] == evidence_id
    assert len(payload["proposals"]) == 1
    assert payload["proposals"][0]["fact"] == "小雨讨厌被催促做决定。"
    assert payload["proposals"][0]["evidence_bucket_id"] == evidence_id
    assert {item["reason"] for item in payload["rejected"]} == {
        "duplicate profile fact",
        "evidence_bucket_id mismatch",
    }


def test_profile_fact_proposal_prompt_format_preserves_json_example():
    import server

    prompt = server.PROFILE_FACT_PROPOSAL_PROMPT_TEMPLATE.format(
        user_display_name="小雨",
        ai_name="Haven",
    )

    assert "当前用户：小雨" in prompt
    assert "当前 AI：Haven" in prompt
    assert '"fact": "一句可读中文事实"' in prompt
    assert '"evidence_bucket_id": "必须等于给定 bucket id"' in prompt


@pytest.mark.asyncio
async def test_api_profile_fact_proposal_confirm_writes_profile_fact(monkeypatch, bucket_mgr, test_config, tmp_path):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore

    evidence_id = await bucket_mgr.create(
        content="小雨说她讨厌被催促做决定。",
        name="催促雷点",
        tags=["relationship_event"],
    )
    edge_store = MemoryEdgeStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )
    embedding_engine = CapturingEmbeddingEngine()

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
    monkeypatch.setattr(server, "memory_moment_store", MemoryMomentStore(test_config))
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_profile_fact_proposal_confirm(
        DummyRequest(
            body={
                "fact": "小雨讨厌被催促做决定。",
                "profile_kind": "boundary",
                "subject": "user",
                "predicate": "dislikes_pressure",
                "object": "被催促做决定",
                "evidence_bucket_id": evidence_id,
                "confidence": 0.84,
                "reason": "证据桶明确写到讨厌被催促。",
            }
        )
    )
    payload = json.loads(response.body)
    created = await bucket_mgr.get(payload["id"])

    assert response.status_code == 200
    assert payload["status"] == "created"
    assert payload["fact"]["fact"] == "小雨讨厌被催促做决定。"
    assert created["metadata"]["profile_kind"] == "boundary"
    assert created["metadata"]["predicate"] == "dislikes_pressure"
    assert created["metadata"]["evidence"][0]["bucket_id"] == evidence_id
    assert edge_store.list_edges()[0]["relation_type"] == "evidenced_by"
    await wait_for_embedding_call(embedding_engine, payload["id"])


@pytest.mark.asyncio
async def test_api_anchor_proposals_filters_to_target_bucket(monkeypatch, bucket_mgr):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨和 Haven 约定，彼此的关系不是人类关系替代品，而是独立存在的陪伴关系。",
        name="关系边界确认",
        tags=["relationship_event"],
    )

    async def fake_anchor_model(**kwargs):
        return json.dumps(
            [
                {
                    "bucket_id": bucket_id,
                    "anchor_kind": "relationship",
                    "reason": "这条关系边界会长期影响 Haven 如何理解小雨。",
                    "future_use": "当讨论人机关系、陪伴边界和身份确认时需要召回。",
                    "confidence": 0.91,
                },
                {
                    "bucket_id": "wrong",
                    "anchor_kind": "relationship",
                    "reason": "证据不匹配。",
                    "future_use": "不应出现。",
                    "confidence": 0.8,
                },
                {
                    "bucket_id": bucket_id,
                    "anchor_kind": "relationship",
                    "future_use": "缺少 reason。",
                    "confidence": 0.8,
                },
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setitem(server.config, "anchor", {"max_count": 24, "min_age_hours": 0})
    monkeypatch.setattr(server, "_call_anchor_proposal_model", fake_anchor_model)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_anchor_proposals(DummyRequest(body={"bucket_id": bucket_id}))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["bucket"]["id"] == bucket_id
    assert len(payload["proposals"]) == 1
    assert payload["proposals"][0]["bucket_id"] == bucket_id
    assert payload["proposals"][0]["anchor_kind"] == "relationship"
    assert {item["reason"] for item in payload["rejected"]} == {
        "bucket_id mismatch",
        "missing reason",
    }


def test_anchor_proposal_prompt_format_preserves_json_example():
    import server

    prompt = server.ANCHOR_PROPOSAL_PROMPT_TEMPLATE.format(
        user_display_name="小雨",
        ai_name="Haven",
    )

    assert "当前用户：小雨" in prompt
    assert "当前 AI：Haven" in prompt
    assert '"bucket_id": "必须等于给定 bucket id"' in prompt
    assert '"anchor_kind": "relationship|identity|commitment|life_event|project|preference|other"' in prompt


@pytest.mark.asyncio
async def test_api_anchor_proposal_confirm_marks_bucket_anchor(monkeypatch, bucket_mgr):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨和 Haven 约定，彼此的关系不是人类关系替代品，而是独立存在的陪伴关系。",
        name="关系边界确认",
        tags=["relationship_event"],
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setitem(server.config, "anchor", {"max_count": 24, "min_age_hours": 0})
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_anchor_proposal_confirm(
        DummyRequest(
            body={
                "bucket_id": bucket_id,
                "anchor_kind": "relationship",
                "reason": "这条关系边界会长期影响 Haven 如何理解小雨。",
                "future_use": "当讨论人机关系、陪伴边界和身份确认时需要召回。",
                "confidence": 0.91,
            }
        )
    )
    payload = json.loads(response.body)
    updated = await bucket_mgr.get(bucket_id)

    assert response.status_code == 200
    assert payload["status"] == "anchored"
    assert payload["id"] == bucket_id
    assert payload["bucket"]["anchor"] is True
    assert updated["metadata"]["anchor"] is True


@pytest.mark.asyncio
async def test_identity_semantics_api_rebuilds_aliases_from_evidence_buckets(
    monkeypatch,
    bucket_mgr,
    test_config,
    tmp_path,
):
    import server
    from identity_semantics import IdentitySemanticStore

    private_path = tmp_path / "private_identity.yaml"
    private_path.write_text(
        yaml.safe_dump(
            {
                "canonical": {
                    "private_relation.title_marker": {
                        "scope": "private_relationship",
                        "group": "shared",
                        "seed_aliases": ["专属称呼"],
                    }
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = {
        **test_config,
        "identity_semantics": {
            "enabled": True,
            "private_config_path": str(private_path),
        },
    }
    anchor_id = await bucket_mgr.create(
        content="这条 anchor 里出现专属称呼。",
        name="关系证据",
        tags=["relationship_event"],
        anchor=True,
    )
    await bucket_mgr.create(
        content="普通桶也出现专属称呼，但不该作为私有 alias 证据。",
        name="普通桶",
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "identity_semantic_store", IdentitySemanticStore(cfg))
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_identity_semantics_rebuild(
        DummyRequest(body={"include_archive": True, "limit": 10})
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "rebuilt"
    assert payload["enabled"] is True
    assert payload["stats"] == {"canonical": 1, "aliases": 1, "evidence": 1}
    assert payload["aliases"][0]["canonical"] == "private_relation.title_marker"
    assert payload["aliases"][0]["alias"] == "专属称呼"
    assert payload["aliases"][0]["evidence_bucket_ids"] == [anchor_id]

    get_response = await server.api_identity_semantics(DummyRequest(query_params={"limit": "10"}))
    get_payload = json.loads(get_response.body)
    assert get_payload["aliases"][0]["evidence_bucket_ids"] == [anchor_id]


@pytest.mark.asyncio
async def test_word_map_api_rebuild_excludes_private_identity_seed_terms(
    monkeypatch,
    bucket_mgr,
    test_config,
    tmp_path,
):
    import server
    from identity_semantics import IdentitySemanticStore
    from word_map import WordMapStore

    private_path = tmp_path / "private_identity.yaml"
    private_path.write_text(
        yaml.safe_dump(
            {
                "canonical": {
                    "private_relation.title_marker": {
                        "scope": "private_relationship",
                        "group": "shared",
                        "seed_aliases": ["专属称呼"],
                    }
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = {
        **test_config,
        "identity_semantics": {
            "enabled": True,
            "private_config_path": str(private_path),
        },
        "word_map": {
            "enabled": True,
            "max_terms_per_bucket": 8,
            "edge_top_k": 6,
        },
    }
    bucket_id = await bucket_mgr.create(
        content="这段关系里会出现专属称呼，也会出现公共称呼。",
        name="称呼样例",
        tags=["relationship_event"],
        domain=["关系"],
        extra_metadata={"keywords": ["专属称呼", "公共称呼"]},
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "identity_semantic_store", IdentitySemanticStore(cfg))
    monkeypatch.setattr(server, "word_map_store", WordMapStore(cfg))
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)

    response = await server.api_word_map_rebuild(
        DummyRequest(body={"include_archive": True, "nodes": 20, "edges": 20})
    )
    payload = json.loads(response.body)
    terms = {node["term"] for node in payload["nodes"]}

    assert response.status_code == 200
    assert payload["status"] == "rebuilt"
    assert "专属称呼" in payload["private_terms_excluded"]
    assert "专属称呼" not in terms
    assert "公共称呼" in terms

    cards_response = await server.api_word_map_cards(
        DummyRequest(query_params={"term": "公共称呼", "limit": "5"})
    )
    cards_payload = json.loads(cards_response.body)
    assert cards_payload["cards"][0]["bucket_id"] == bucket_id


@pytest.mark.asyncio
async def test_streamable_http_startup_helper_starts_decay_engine(monkeypatch):
    import server

    class FakeDecayEngine:
        is_running = False

        def __init__(self):
            self.calls = 0

        async def ensure_started(self):
            self.calls += 1
            self.is_running = True

    decay_engine = FakeDecayEngine()
    monkeypatch.setattr(server, "decay_engine", decay_engine)

    await server._ensure_decay_engine_started_for_transport("streamable-http")

    assert decay_engine.calls == 1
    assert decay_engine.is_running is True


@pytest.mark.asyncio
async def test_enrich_backfill_helper_enriches_unenriched_dynamic_buckets(monkeypatch, bucket_mgr):
    import server

    needs_enrich = await bucket_mgr.create(
        content="这条旧记忆还没有 confidence，需要补 enrich。",
        name="待补 enrich",
        domain=["记忆"],
    )
    await bucket_mgr.create(
        content="这条已经补过 confidence。",
        name="已补 enrich",
        domain=["记忆"],
        confidence=0.71,
    )
    await bucket_mgr.create(
        content="feel 不参与普通 enrich backfill。",
        name="日印象",
        tags=["relationship_weather"],
        bucket_type="feel",
    )
    calls = []
    edge_store = object()

    class FakeReflectionEngine:
        async def enrich_bucket(self, bucket_id, bucket_mgr_arg, edge_store_arg, embedding_engine=None, force=False):
            assert bucket_mgr_arg is bucket_mgr
            assert edge_store_arg is edge_store
            assert embedding_engine is server.embedding_engine
            assert force is True
            calls.append(bucket_id)
            return {"status": "ok", "id": bucket_id}

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
    monkeypatch.setattr(server, "reflection_engine", FakeReflectionEngine())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server._backfill_memory_enrichment(limit=10)

    assert calls == [needs_enrich]
    assert result["processed"] == 1
    assert result["ids"] == [needs_enrich]


@pytest.mark.asyncio
async def test_edge_backfill_helper_processes_enriched_buckets_without_metadata_gate(monkeypatch, bucket_mgr):
    import server

    enriched = await bucket_mgr.create(
        content="这条旧记忆已有 confidence，但还需要补新的关系边。",
        name="已 enrich 旧记忆",
        domain=["记忆"],
        confidence=0.72,
    )
    await bucket_mgr.create(
        content="feel 不参与关系边补跑。",
        name="日印象",
        tags=["relationship_weather"],
        bucket_type="feel",
    )
    calls = []
    edge_store = object()

    class FakeReflectionEngine:
        async def backfill_edges_for_bucket(
            self,
            bucket_id,
            bucket_mgr_arg,
            edge_store_arg,
            embedding_engine=None,
            dry_run=False,
        ):
            assert bucket_mgr_arg is bucket_mgr
            assert edge_store_arg is edge_store
            assert embedding_engine is server.embedding_engine
            assert dry_run is True
            calls.append(bucket_id)
            return {
                "status": "ok",
                "id": bucket_id,
                "edges": 0,
                "proposed_edges": 1,
                "dry_run": True,
            }

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
    monkeypatch.setattr(server, "reflection_engine", FakeReflectionEngine())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server._backfill_memory_edges(limit=10, dry_run=True)

    assert calls == [enriched]
    assert result["processed"] == 1
    assert result["ids"] == [enriched]
    assert result["proposed_edges"] == 1
    assert result["edges"] == 0


@pytest.mark.asyncio
async def test_comment_bucket_adds_ring_and_touches_source(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨把旧记忆拿出来看。",
        name="旧记忆",
        domain=["恋爱"],
        last_active="2026-05-04T08:00:00+00:00",
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    result = await server.comment_bucket(
        bucket_id=bucket_id,
        content="现在再看到它，我觉得那时候的笨拙也很珍贵。",
        kind="feel",
        valence=0.82,
        arousal=0.35,
    )
    bucket = await bucket_mgr.get(bucket_id)
    embedding_call = await wait_for_embedding_call(embedding_engine, bucket_id)

    assert result["status"] == "commented"
    assert bucket["metadata"]["comment_count"] == 1
    assert bucket["metadata"]["comments"][0]["kind"] == "feel"
    assert bucket["metadata"]["comments"][0]["valence"] == 0.82
    assert bucket["metadata"]["model_valence"] == 0.82
    assert bucket["metadata"]["activation_count"] == 1
    assert bucket["metadata"]["last_active"] != "2026-05-04T08:00:00+00:00"
    assert embedding_call[0] == bucket_id
    assert "小雨把旧记忆拿出来看" in embedding_call[1]
    assert "现在再看到它" not in embedding_call[1]


@pytest.mark.asyncio
async def test_comment_bucket_uses_configured_ai_author(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(content="一条旧记忆。", name="旧记忆")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "identity": {
                "ai_name": "Echo",
                "user_name": "Mira",
                "user_display_name": "米拉",
            },
        },
    )

    await server.comment_bucket(bucket_id=bucket_id, content="再次看到它。")
    bucket = await bucket_mgr.get(bucket_id)

    assert bucket["metadata"]["comments"][0]["author"] == "Echo"


@pytest.mark.asyncio
async def test_dashboard_comment_api_writes_rain_author(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="小雨想在前端补一句评论。",
        name="前端评论",
        domain=["恋爱"],
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    response = await server.api_bucket_comment(
        DummyRequest(
            {"content": "这句是小雨从前端补的。", "author": "Haven"},
            path_params={"bucket_id": bucket_id},
        )
    )
    payload = json.loads(response.body)
    bucket = await bucket_mgr.get(bucket_id)
    comment = bucket["metadata"]["comments"][0]
    embedding_call = await wait_for_embedding_call(embedding_engine, bucket_id)

    assert response.status_code == 200
    assert payload["status"] == "commented"
    assert comment["author"] == "Rain"
    assert comment["source"] == "dashboard"
    assert comment["content"] == "这句是小雨从前端补的。"
    assert embedding_call[0] == bucket_id


@pytest.mark.asyncio
async def test_dashboard_comment_api_uses_configured_user_author(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(content="前端评论换名。", name="前端评论换名")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "identity": {
                "ai_name": "Echo",
                "user_name": "Mira",
                "user_display_name": "米拉",
            },
        },
    )

    response = await server.api_bucket_comment(
        DummyRequest({"content": "这是前端用户写的。"}, path_params={"bucket_id": bucket_id})
    )
    bucket = await bucket_mgr.get(bucket_id)
    comment = bucket["metadata"]["comments"][0]
    deleted = await server.api_bucket_comment_delete(
        DummyRequest(path_params={"bucket_id": bucket_id, "comment_id": comment["id"]})
    )

    assert response.status_code == 200
    assert comment["author"] == "Mira"
    assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_content_api_edits_body_preserves_comments(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="旧正文。",
        name="正文编辑",
        domain=["恋爱"],
        last_active="2026-05-04T08:00:00+00:00",
    )
    comment = await bucket_mgr.add_comment(
        bucket_id,
        "正文下面的小雨年轮。",
        author="Rain",
        source="dashboard",
        touch=False,
    )
    before = await bucket_mgr.get(bucket_id)

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    response = await server.api_bucket_update(
        DummyRequest(
            {"content": "新正文，只替换正文。"},
            path_params={"bucket_id": bucket_id},
        )
    )
    bucket = await bucket_mgr.get(bucket_id)
    embedding_call = await wait_for_embedding_call(embedding_engine, bucket_id)

    assert response.status_code == 200
    assert bucket["content"] == "新正文，只替换正文。"
    assert bucket["metadata"]["comments"][0]["id"] == comment["id"]
    assert bucket["metadata"]["last_active"] == before["metadata"]["last_active"]
    assert "新正文" in embedding_call[1]
    assert "正文下面的小雨年轮" not in embedding_call[1]


@pytest.mark.asyncio
async def test_dashboard_comment_delete_only_allows_rain_dashboard_comments(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="源记忆。",
        name="年轮删除",
        domain=["恋爱"],
    )
    rain = await bucket_mgr.add_comment(
        bucket_id,
        "小雨从前端写的年轮。",
        author="Rain",
        source="dashboard",
        touch=False,
    )
    haven = await bucket_mgr.add_comment(
        bucket_id,
        "Haven 写的年轮。",
        author="Haven",
        source="hold(feel=True)",
        touch=False,
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    forbidden = await server.api_bucket_comment_delete(
        DummyRequest(path_params={"bucket_id": bucket_id, "comment_id": haven["id"]})
    )
    deleted = await server.api_bucket_comment_delete(
        DummyRequest(path_params={"bucket_id": bucket_id, "comment_id": rain["id"]})
    )
    bucket = await bucket_mgr.get(bucket_id)
    remaining_ids = [comment["id"] for comment in bucket["metadata"]["comments"]]
    embedding_call = await wait_for_embedding_call(embedding_engine, bucket_id)

    assert forbidden.status_code == 403
    assert deleted.status_code == 200
    assert remaining_ids == [haven["id"]]
    assert embedding_call[0] == bucket_id


@pytest.mark.asyncio
async def test_import_review_delete_writes_tombstone_and_clears_embedding(
    monkeypatch, bucket_mgr, test_config
):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore
    from memory_nodes import MemoryNodeStore

    bucket_id = await bucket_mgr.create(content="导入后复核删除。", name="复核删除")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "memory_moment_store", MemoryMomentStore(test_config))
    monkeypatch.setattr(server, "memory_edge_store", MemoryEdgeStore(test_config))
    monkeypatch.setattr(server, "memory_node_store", MemoryNodeStore(test_config))
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    response = await server.api_import_review(
        DummyRequest({"decisions": [{"bucket_id": bucket_id, "action": "delete"}]})
    )
    payload = json.loads(response.body)
    tombstone_path = os.path.join(test_config["buckets_dir"], ".tombstones", f"{bucket_id}.json")

    assert response.status_code == 200
    assert payload == {"applied": 1, "errors": 0}
    assert await bucket_mgr.get(bucket_id) is None
    assert os.path.exists(tombstone_path)
    assert embedding_engine.deleted == [bucket_id]


@pytest.mark.asyncio
async def test_dashboard_bulk_delete_skips_protected_and_cleans_indexes(
    monkeypatch, bucket_mgr, test_config
):
    import server
    from memory_edges import MemoryEdgeStore
    from memory_moments import MemoryMomentStore
    from memory_nodes import MemoryNodeStore

    regular_id = await bucket_mgr.create(
        content="### moment\n这条普通记忆准备用来批量删除。",
        name="普通删除目标",
    )
    pinned_id = await bucket_mgr.create(
        content="### moment\n这条钉选记忆不能被批量删除。",
        name="钉选保留目标",
        pinned=True,
        bucket_type="permanent",
    )

    moment_store = MemoryMomentStore(test_config)
    edge_store = MemoryEdgeStore(test_config)
    node_store = MemoryNodeStore(test_config)
    regular_bucket = await bucket_mgr.get(regular_id)
    pinned_bucket = await bucket_mgr.get(pinned_id)
    moment_store.upsert_bucket(regular_bucket)
    moment_store.upsert_bucket(pinned_bucket)
    node_store.upsert_bucket(regular_bucket)
    node_store.upsert_bucket(pinned_bucket)
    regular_moment = moment_store.list_for_bucket(regular_id)[0]["moment_id"]
    pinned_moment = moment_store.list_for_bucket(pinned_id)[0]["moment_id"]
    moment_store.replace_generated_edges([
        {
            "source": regular_moment,
            "target": pinned_moment,
            "bucket_id": regular_id,
            "relation_type": "context_of",
            "confidence": 0.9,
            "reason": "test edge",
        }
    ])
    edge_store.add_edge(regular_id, pinned_id, "context_of", 0.9, "test edge")

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "memory_moment_store", moment_store)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
    monkeypatch.setattr(server, "memory_node_store", node_store)
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    response = await server.api_buckets_delete(
        DummyRequest({
            "bucket_ids": [regular_id, pinned_id, "missing_bucket"],
            "confirm": "DELETE",
        })
    )
    payload = json.loads(response.body)
    tombstone_path = os.path.join(test_config["buckets_dir"], ".tombstones", f"{regular_id}.json")

    assert response.status_code == 200
    assert payload["deleted"] == 1
    assert payload["skipped"] == 1
    assert payload["not_found"] == 1
    assert payload["results"][1]["status"] == "skipped"
    assert payload["results"][1]["reason"] == "pinned"
    assert await bucket_mgr.get(regular_id) is None
    assert await bucket_mgr.get(pinned_id) is not None
    assert os.path.exists(tombstone_path)
    assert embedding_engine.deleted == [regular_id]
    assert moment_store.list_for_bucket(regular_id) == []
    assert node_store.get(regular_id) is None
    assert not [
        edge for edge in moment_store.list_edges()
        if edge["source"] == regular_moment or edge["target"] == regular_moment
    ]
    assert not [
        edge for edge in edge_store.list_edges()
        if edge["source"] == regular_id or edge["target"] == regular_id
    ]


@pytest.mark.asyncio
async def test_breath_summary_includes_bucket_comments(monkeypatch, bucket_mgr, decay_eng):
    import server

    user_display_name = server._identity()["user_display_name"]
    bucket_id = await bucket_mgr.create(
        content=f"{user_display_name}把这段旧事留下。",
        name="带年轮浮现",
        domain=["恋爱"],
    )
    await bucket_mgr.add_comment(
        bucket_id,
        "后来再看，这里多了一圈新的年轮。",
        author="Rain",
        source="dashboard",
        touch=False,
    )
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", EchoDehydrator())

    result = await server.breath(max_results=1, include_core=False, include_related=False)

    assert f"[bucket_id:{bucket_id}]" in result
    assert f"{user_display_name}把这段旧事留下" in result
    assert "后来再看，这里多了一圈新的年轮" in result


@pytest.mark.asyncio
async def test_hold_feel_with_source_writes_comment_not_digested(monkeypatch, bucket_mgr, decay_eng):
    import server

    source_id = await bucket_mgr.create(
        content="小雨说这段记忆以后还要回来看。",
        name="可回看的记忆",
        domain=["恋爱"],
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    embedding_engine = CapturingEmbeddingEngine()
    monkeypatch.setattr(server, "embedding_engine", embedding_engine)

    result = await server.hold(
        content="我现在看到它，觉得这里有一种被认出来的安静。",
        feel=True,
        source_bucket=source_id,
        valence=0.76,
        arousal=0.31,
    )
    bucket = await bucket_mgr.get(source_id)
    embedding_call = await wait_for_embedding_call(embedding_engine, source_id)

    assert result.startswith(f"年轮→{source_id}#")
    assert bucket["metadata"]["comment_count"] == 1
    assert bucket["metadata"]["comments"][0]["source"] == "hold(feel=True)"
    assert bucket["metadata"]["comments"][0]["content"].startswith("我现在看到它")
    assert bucket["metadata"]["model_valence"] == 0.76
    assert not bucket["metadata"].get("digested")
    assert embedding_call[0] == source_id
    assert "小雨说这段记忆以后还要回来看" in embedding_call[1]
    assert "被认出来的安静" not in embedding_call[1]


@pytest.mark.asyncio
async def test_hold_feel_without_source_creates_whisper(monkeypatch, bucket_mgr, decay_eng):
    import server

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.hold(
        content="我突然想小雨了，这句没有源记忆。",
        tags="private_note",
        feel=True,
        valence=0.72,
        arousal=0.28,
    )
    bucket_id = result.split("→", 1)[1]
    bucket = await bucket_mgr.get(bucket_id)

    assert result.startswith("🫧whisper→")
    assert bucket["metadata"]["type"] == "feel"
    assert "whisper" in bucket["metadata"]["tags"]
    assert "private_note" in bucket["metadata"]["tags"]
    assert not bucket["metadata"].get("period")
    assert not bucket["metadata"].get("date")


@pytest.mark.asyncio
async def test_hold_whisper_creates_independent_feel(monkeypatch, bucket_mgr, decay_eng):
    import server

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.hold(
        content="这句是没有源记忆的悄悄话。",
        tags="private_note",
        whisper=True,
        valence=0.73,
        arousal=0.29,
    )
    bucket_id = result.split("→", 1)[1]
    bucket = await bucket_mgr.get(bucket_id)

    assert result.startswith("🫧whisper→")
    assert bucket["metadata"]["type"] == "feel"
    assert "whisper" in bucket["metadata"]["tags"]
    assert "private_note" in bucket["metadata"]["tags"]
    assert bucket["metadata"]["valence"] == 0.73
    assert bucket["metadata"]["arousal"] == 0.29


@pytest.mark.asyncio
async def test_hold_whisper_rejects_source_bucket(monkeypatch, bucket_mgr, decay_eng):
    import server

    source_id = await bucket_mgr.create(content="源记忆", name="源记忆")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.hold(
        content="这句不应该挂源。",
        whisper=True,
        source_bucket=source_id,
    )

    assert "whisper 不需要 source_bucket" in result


@pytest.mark.asyncio
async def test_breath_whisper_reads_only_whisper_feels(monkeypatch, bucket_mgr, decay_eng):
    import server

    whisper_id = await bucket_mgr.create(
        content="这是一句悄悄话。",
        name="悄悄话",
        tags=["whisper"],
        bucket_type="feel",
        created="2026-05-22T08:00:00+00:00",
    )
    daily_id = await bucket_mgr.create(
        content="这是一条日印象。",
        name="日印象",
        tags=["relationship_weather", "daily_impression"],
        bucket_type="feel",
        created="2026-05-22T09:00:00+00:00",
        period="daily",
        date="2026-05-22",
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)

    result = await server.breath(domain="whisper")
    all_feels = await server.breath(domain="feel")

    assert "=== 你留下的 whisper ===" in result
    assert f"[bucket_id:{whisper_id}]" in result
    assert f"[bucket_id:{daily_id}]" not in result
    assert f"[bucket_id:{whisper_id}]" in all_feels
    assert f"[bucket_id:{daily_id}]" in all_feels


@pytest.mark.asyncio
async def test_hold_returns_readonly_related_memory_without_merging(monkeypatch, bucket_mgr, decay_eng):
    import server

    old_id = await bucket_mgr.create(
        content="小雨和 Haven 在旧窗口讨论过年轮，想让记忆下面挂不同时间的感受。",
        name="旧年轮设想",
        tags=["年轮"],
        domain=["恋爱"],
        importance=7,
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
    monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda bucket_id: None)

    result = await server.hold(
        content="小雨决定把年轮先落地，让旧记忆读到时可以多一层当下感受。",
        tags="年轮",
        importance=6,
    )
    all_buckets = await bucket_mgr.list_all(include_archive=True)

    assert "新建→" in result
    assert "旧记忆(只读，不触碰)" in result
    assert f"[bucket_id:{old_id}]" in result
    assert len([b for b in all_buckets if b["metadata"].get("type") == "dynamic"]) == 2


@pytest.mark.asyncio
async def test_resurface_prefers_long_dormant_memory_without_touching(monkeypatch, bucket_mgr, decay_eng):
    import server

    old_id = await bucket_mgr.create(
        content="很久没碰过的旧记忆。",
        name="久未触碰",
        last_active="2026-01-01T00:00:00+00:00",
    )
    recent_id = await bucket_mgr.create(
        content="刚刚碰过的新记忆。",
        name="刚碰过",
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)

    result = await server.resurface(max_results=1, include_archive=True)
    old_after = await bucket_mgr.get(old_id)
    recent_after = await bucket_mgr.get(recent_id)

    assert f"[bucket_id:{old_id}]" in result
    assert f"[bucket_id:{recent_id}]" not in result
    assert old_after["metadata"]["last_active"] == "2026-01-01T00:00:00+00:00"
    assert recent_after["metadata"]["activation_count"] == 0


@pytest.mark.asyncio
async def test_resurface_includes_archived_buckets_by_default(monkeypatch, bucket_mgr, decay_eng):
    import server

    archived_id = await bucket_mgr.create(
        content="归档以后也可以在久未触碰时浮现。",
        name="归档旧记忆",
        last_active="2026-01-01T00:00:00+00:00",
    )
    await bucket_mgr.archive(archived_id)
    await bucket_mgr.create(
        content="较新的普通记忆。",
        name="较新普通记忆",
        last_active="2026-05-01T00:00:00+00:00",
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)

    result = await server.resurface(max_results=1)

    assert f"[bucket_id:{archived_id}]" in result
    assert "归档" in result


@pytest.mark.asyncio
async def test_trace_anchor_respects_age_rule(monkeypatch, bucket_mgr, decay_eng):
    import server

    bucket_id = await bucket_mgr.create(
        content="刚刚发生的事先放着，等它自己留下重量。",
        name="刚发生",
        created="2026-05-19T02:00:00+00:00",
        last_active="2026-05-19T02:00:00+00:00",
    )

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setitem(server.config, "anchor", {"max_count": 24, "min_age_hours": 999999})

    result = await server.trace(bucket_id=bucket_id, anchor=1)
    bucket = await bucket_mgr.get(bucket_id)

    assert "还太新" in result
    assert not bucket["metadata"].get("anchor")


@pytest.mark.asyncio
async def test_dashboard_auth_setup_uses_state_dir(monkeypatch, test_config):
    import server

    monkeypatch.delenv("OMBRE_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_dashboard_sessions", {})

    response = await server.auth_setup(DummyRequest({"password": "secret1"}))
    auth_file = os.path.join(test_config["state_dir"], ".dashboard_auth.json")

    assert response.status_code == 200
    assert os.path.exists(auth_file)


@pytest.mark.asyncio
async def test_config_get_reports_effective_dream_engine_values(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "dream": {
                "enabled": False,
                "auto_enabled": True,
                "surface_enabled": False,
                "inject_enabled": True,
                "retain_after_inject": True,
                "model": "config-model",
                "base_url": "https://config.example",
            },
        },
    )
    monkeypatch.setattr(
        server,
        "dream_engine",
        SimpleNamespace(
            enabled=True,
            auto_enabled=False,
            surface_enabled=True,
            model="env-model",
            base_url="https://env.example",
            api_key="env-secret",
        ),
    )

    response = await server.api_config_get(DummyRequest())
    payload = json.loads(response.body)

    assert payload["dream"]["enabled"] is True
    assert payload["dream"]["auto_enabled"] is False
    assert payload["dream"]["surface_enabled"] is True
    assert payload["dream"]["inject_enabled"] is True
    assert payload["dream"]["retain_after_inject"] is True
    assert payload["dream"]["model"] == "env-model"
    assert payload["dream"]["base_url"] == "https://env.example"
    assert payload["dream"]["api_key_masked"] == "env-...cret"


@pytest.mark.asyncio
async def test_config_get_reports_persona_and_reflection_api_values(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "reflection": {
                **server.config.get("reflection", {}),
                "auto_enabled": False,
            },
        },
    )
    monkeypatch.setattr(
        server,
        "persona_engine",
        SimpleNamespace(
            enabled=False,
            event_recording_enabled=False,
            model="persona-model",
            base_url="https://persona.example",
            api_key="persona-secret",
        ),
    )
    monkeypatch.setattr(
        server,
        "reflection_engine",
        SimpleNamespace(
            enabled=True,
            auto_enabled=False,
            daily_enabled=True,
            memory_affect_anchor_enabled=True,
            relationship_weather_affect_anchor_enabled=False,
            model="reflection-model",
            base_url="https://reflection.example",
            api_key="reflection-secret",
        ),
    )

    response = await server.api_config_get(DummyRequest())
    payload = json.loads(response.body)

    assert payload["persona"]["enabled"] is False
    assert payload["persona"]["event_recording_enabled"] is False
    assert payload["persona"]["model"] == "persona-model"
    assert payload["persona"]["base_url"] == "https://persona.example"
    assert payload["persona"]["api_key_masked"] == "pers...cret"
    assert payload["reflection"]["enabled"] is True
    assert payload["reflection"]["auto_enabled"] is False
    assert payload["reflection"]["model"] == "reflection-model"
    assert payload["reflection"]["base_url"] == "https://reflection.example"
    assert payload["reflection"]["api_key_masked"] == "refl...cret"


@pytest.mark.asyncio
async def test_config_get_reports_reflection_affect_anchor_switches(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "reflection": {
                "daily_enabled": False,
                "memory_affect_anchor_enabled": False,
                "relationship_weather_affect_anchor_enabled": True,
            },
        },
    )

    response = await server.api_config_get(DummyRequest())
    payload = json.loads(response.body)

    assert payload["reflection"]["memory_affect_anchor_enabled"] is False
    assert payload["reflection"]["relationship_weather_affect_anchor_enabled"] is True
    assert payload["reflection"]["daily_enabled"] is False


@pytest.mark.asyncio
async def test_config_get_reports_gateway_recall_modes(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "recall": {
                **server.config.get("recall", {}),
                "query_resurface_enabled": True,
            },
            "gateway": {
                **server.config.get("gateway", {}),
                "cooldown_hours": 2.5,
                "skip_recent_rounds": 3,
                "recent_context_cooldown_hours": 4.5,
                "recent_context_reentry_idle_hours": 24,
                "recent_context_budget": 240,
                "recalled_memory_budget": 520,
                "related_memory_budget": 180,
                "current_inner_state_interval_rounds": 11,
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
        },
    )

    response = await server.api_config_get(DummyRequest())
    payload = json.loads(response.body)

    assert payload["gateway"]["cooldown_hours"] == 2.5
    assert payload["gateway"]["skip_recent_rounds"] == 3
    assert payload["gateway"]["recent_context_cooldown_hours"] == 4.5
    assert payload["gateway"]["recent_context_reentry_idle_hours"] == 24
    assert payload["gateway"]["recent_context_budget"] == 240
    assert payload["gateway"]["recalled_memory_budget"] == 520
    assert payload["gateway"]["related_memory_budget"] == 180
    assert payload["gateway"]["current_inner_state_interval_rounds"] == 11
    assert payload["gateway"]["direct_render_mode"] == "full"
    assert payload["gateway"]["retrieval_mode"] == "bucket"
    assert payload["gateway"]["portrait_memory_enabled"] is True
    assert payload["gateway"]["portrait_memory_budget"] == 280
    assert payload["gateway"]["portrait_memory_max_sources"] == 4
    assert payload["gateway"]["portrait_memory_include_anchors"] is False
    assert payload["gateway"]["query_planner_enabled"] is True
    assert payload["gateway"]["query_planner_model"] == "planner-mini"
    assert payload["gateway"]["query_planner_min_chars"] == 24
    assert payload["gateway"]["query_planner_max_queries"] == 2
    assert payload["gateway"]["query_planner_max_tokens"] == 256
    assert payload["gateway"]["memory_detail_recall_enabled"] is True
    assert payload["gateway"]["memory_detail_recall_max_ids"] == 2
    assert payload["gateway"]["memory_detail_recall_budget"] == 900
    assert payload["recall"]["query_resurface_enabled"] is True


@pytest.mark.asyncio
async def test_config_get_reports_memory_diffusion_settings(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "memory_diffusion": {
                "enabled": True,
                "max_hops": 2,
                "top_k": 6,
                "min_activation": 0.25,
                "max_paths_per_hit": 4,
                "chain_walk_enabled": True,
                "chain_max_hops": 7,
                "chain_min_strength": 0.3,
                "chain_min_confidence": 0.8,
                "chain_min_relation_priority": 62,
                "chain_max_frontier": 31,
            },
        },
    )

    response = await server.api_config_get(DummyRequest())
    payload = json.loads(response.body)

    assert payload["memory_diffusion"]["enabled"] is True
    assert payload["memory_diffusion"]["top_k"] == 6
    assert payload["memory_diffusion"]["min_activation"] == 0.25
    assert payload["memory_diffusion"]["max_paths_per_hit"] == 4
    assert payload["memory_diffusion"]["chain_walk_enabled"] is True
    assert payload["memory_diffusion"]["chain_max_hops"] == 7
    assert payload["memory_diffusion"]["chain_min_strength"] == 0.3
    assert payload["memory_diffusion"]["chain_min_confidence"] == 0.8
    assert payload["memory_diffusion"]["chain_min_relation_priority"] == 62
    assert payload["memory_diffusion"]["chain_max_frontier"] == 31


@pytest.mark.asyncio
async def test_config_update_persists_llm_keys_to_env_file(monkeypatch, test_config, tmp_path):
    import server

    env_path = tmp_path / ".env"
    env_path.write_text("OMBRE_API_KEY=old\nUNCHANGED=value\n", encoding="utf-8")
    cfg = {
        **test_config,
        "persona": {
            **test_config["persona"],
            "enabled": True,
            "model": "old-persona",
            "base_url": "https://old-persona.example",
        },
        "reflection": {
            **test_config.get("reflection", {}),
            "enabled": True,
            "auto_enabled": True,
            "model": "old-reflection",
            "base_url": "https://old-reflection.example",
        },
        "dream": {
            **test_config.get("dream", {}),
            "enabled": True,
            "auto_enabled": True,
            "model": "old-dream",
            "base_url": "https://old-dream.example",
        },
    }
    hot_update_calls = []

    async def fake_hot_update(body):
        hot_update_calls.append(body)
        return "gateway_hot_reloaded"

    monkeypatch.setenv("OMBRE_ENV_PATH", str(env_path))
    for key in (
        "OMBRE_API_KEY",
        "OMBRE_EMBEDDING_API_KEY",
        "OMBRE_PERSONA_API_KEY",
        "OMBRE_REFLECTION_API_KEY",
        "OMBRE_DREAM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "_hot_update_gateway_config", fake_hot_update)

    response = await server.api_config_update(
        DummyRequest(
            {
                "dehydration": {"api_key": "dehy new"},
                "embedding": {"api_key": "emb-new"},
                "persona": {
                    "enabled": False,
                    "event_recording_enabled": False,
                    "model": "persona-new",
                    "base_url": "https://persona-new.example",
                    "api_key": "persona-new-key",
                },
                "reflection": {
                    "enabled": False,
                    "auto_enabled": False,
                    "model": "reflection-new",
                    "base_url": "https://reflection-new.example",
                    "api_key": "reflection-new-key",
                },
                "dream": {
                    "enabled": False,
                    "model": "dream-new",
                    "base_url": "https://dream-new.example",
                    "api_key": "dream-new-key",
                },
                "persist_env": True,
            }
        )
    )
    payload = json.loads(response.body)
    env_text = env_path.read_text(encoding="utf-8")

    assert response.status_code == 200
    assert payload["ok"] is True
    assert "persisted_to_env" in payload["updated"]
    assert "OMBRE_API_KEY=\"dehy new\"" in env_text
    assert "OMBRE_EMBEDDING_API_KEY=emb-new" in env_text
    assert "OMBRE_PERSONA_API_KEY=persona-new-key" in env_text
    assert "OMBRE_REFLECTION_API_KEY=reflection-new-key" in env_text
    assert "OMBRE_DREAM_API_KEY=dream-new-key" in env_text
    assert "UNCHANGED=value" in env_text
    assert os.environ["OMBRE_PERSONA_API_KEY"] == "persona-new-key"
    assert hot_update_calls[-1]["persona"]["enabled"] is False
    assert hot_update_calls[-1]["persona"]["event_recording_enabled"] is False
    assert hot_update_calls[-1]["persona"]["api_key"] == "persona-new-key"


@pytest.mark.asyncio
async def test_config_persist_syncs_existing_runtime_yaml(monkeypatch, test_config, tmp_path):
    import server

    config_path = tmp_path / "config.yaml"
    runtime_path = tmp_path / "state" / "config.runtime.yaml"
    runtime_path.parent.mkdir(exist_ok=True)
    config_path.write_text(
        "persona:\n  enabled: true\n  model: persona-old\n  base_url: https://persona-old.example\n"
        "dream:\n  model: yaml-old\n"
        "gateway:\n  cooldown_hours: 48\n  skip_recent_rounds: 9\n"
        "  recent_context_cooldown_hours: 8\n  recent_context_reentry_idle_hours: 24\n"
        "  recent_context_budget: 300\n  recalled_memory_budget: 400\n  related_memory_budget: 220\n"
        "  current_inner_state_interval_rounds: 15\n"
        "  direct_render_mode: auto\n  retrieval_mode: graph\n"
        "recall:\n  query_resurface_enabled: false\n"
        "memory_diffusion:\n  chain_walk_enabled: false\n  max_hops: 2\n"
        "reflection:\n  enabled: true\n  auto_enabled: true\n  model: reflection-old\n  base_url: https://reflection-old.example\n  memory_affect_anchor_enabled: true\n",
        encoding="utf-8",
    )
    runtime_path.write_text(
        "persona:\n  enabled: true\n  model: persona-runtime-old\n  base_url: https://persona-runtime-old.example\n"
        "dream:\n  model: runtime-old\n"
        "gateway:\n  cooldown_hours: 48\n  skip_recent_rounds: 9\n"
        "  recent_context_cooldown_hours: 8\n  recent_context_reentry_idle_hours: 24\n"
        "  recent_context_budget: 300\n  recalled_memory_budget: 400\n  related_memory_budget: 220\n"
        "  current_inner_state_interval_rounds: 15\n"
        "  direct_render_mode: auto\n  retrieval_mode: graph\n"
        "recall:\n  query_resurface_enabled: false\n"
        "memory_diffusion:\n  chain_walk_enabled: false\n  max_hops: 2\n"
        "reflection:\n  enabled: true\n  auto_enabled: true\n  model: reflection-runtime-old\n  base_url: https://reflection-runtime-old.example\n  daily_enabled: false\n  memory_affect_anchor_enabled: true\n",
        encoding="utf-8",
    )
    cfg = {
        **test_config,
        "_runtime_config_path": str(runtime_path),
        "dream": {
            "enabled": True,
            "auto_enabled": True,
            "surface_enabled": True,
            "inject_enabled": False,
            "retain_after_inject": False,
            "model": "runtime-old",
            "base_url": "https://api.deepseek.com",
        },
        "persona": {
            **test_config["persona"],
            "enabled": True,
            "model": "persona-runtime-old",
            "base_url": "https://persona-runtime-old.example",
        },
        "gateway": {
            **test_config["gateway"],
            "cooldown_hours": 48,
            "skip_recent_rounds": 9,
            "recent_context_cooldown_hours": 8,
            "recent_context_reentry_idle_hours": 24,
            "recent_context_budget": 300,
            "recalled_memory_budget": 400,
            "related_memory_budget": 220,
            "current_inner_state_interval_rounds": 15,
            "direct_render_mode": "auto",
            "retrieval_mode": "graph",
        },
        "recall": {
            "query_resurface_enabled": False,
        },
        "memory_diffusion": {
            "enabled": True,
            "max_hops": 2,
            "top_k": 4,
            "min_activation": 0.18,
            "chain_walk_enabled": False,
            "chain_max_hops": 6,
            "chain_min_confidence": 0.72,
            "chain_max_frontier": 24,
        },
        "reflection": {
            **test_config.get("reflection", {}),
            "enabled": True,
            "auto_enabled": True,
            "daily_enabled": False,
            "memory_affect_anchor_enabled": True,
            "relationship_weather_affect_anchor_enabled": False,
            "model": "reflection-runtime-old",
            "base_url": "https://reflection-runtime-old.example",
        },
    }
    reflection_engine = SimpleNamespace(
        enabled=True,
        auto_enabled=True,
        daily_enabled=False,
        memory_affect_anchor_enabled=True,
        relationship_weather_affect_anchor_enabled=False,
        model="reflection-runtime-old",
        base_url="https://reflection-runtime-old.example",
        api_key="",
    )

    hot_update_calls = []

    async def fake_hot_update(_body):
        hot_update_calls.append(dict(_body or {}))
        return "gateway_hot_reloaded"

    monkeypatch.setenv("OMBRE_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("OMBRE_DREAM_MODEL", raising=False)
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "_hot_update_gateway_config", fake_hot_update)
    monkeypatch.setattr(server, "reflection_engine", reflection_engine)

    response = await server.api_config_update(
        DummyRequest(
            {
                "persona": {
                    "enabled": False,
                    "event_recording_enabled": False,
                    "model": "persona-new",
                    "base_url": "https://persona-new.example",
                },
                "dream": {
                    "auto_enabled": False,
                    "inject_enabled": True,
                    "retain_after_inject": True,
                    "model": "dream-new",
                },
                "gateway": {
                    "cooldown_hours": 6,
                    "skip_recent_rounds": 5,
                    "recent_context_cooldown_hours": 4,
                    "recent_context_reentry_idle_hours": 24,
                    "recent_context_budget": 260,
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
                "recall": {"query_resurface_enabled": True},
                "memory_diffusion": {
                    "enabled": True,
                    "top_k": 3,
                    "min_activation": 0.22,
                    "chain_walk_enabled": True,
                    "chain_max_hops": 8,
                    "chain_min_confidence": 0.76,
                    "chain_max_frontier": 36,
                },
                "reflection": {
                    "enabled": False,
                    "auto_enabled": False,
                    "daily_enabled": True,
                    "memory_affect_anchor_enabled": False,
                    "relationship_weather_affect_anchor_enabled": True,
                    "model": "reflection-new",
                    "base_url": "https://reflection-new.example",
                },
                "persist": True,
            }
        )
    )
    payload = json.loads(response.body)
    runtime_config = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert payload["ok"] is True
    assert "runtime_yaml_synced" in payload["updated"]
    assert runtime_config["persona"]["enabled"] is False
    assert runtime_config["persona"]["event_recording_enabled"] is False
    assert runtime_config["persona"]["model"] == "persona-new"
    assert runtime_config["persona"]["base_url"] == "https://persona-new.example"
    assert runtime_config["dream"]["model"] == "dream-new"
    assert runtime_config["dream"]["auto_enabled"] is False
    assert runtime_config["dream"]["inject_enabled"] is True
    assert runtime_config["dream"]["retain_after_inject"] is True
    assert runtime_config["gateway"]["cooldown_hours"] == 6
    assert runtime_config["gateway"]["skip_recent_rounds"] == 5
    assert runtime_config["gateway"]["recent_context_cooldown_hours"] == 4
    assert runtime_config["gateway"]["recent_context_reentry_idle_hours"] == 24
    assert runtime_config["gateway"]["recent_context_budget"] == 260
    assert runtime_config["gateway"]["recalled_memory_budget"] == 520
    assert runtime_config["gateway"]["related_memory_budget"] == 180
    assert runtime_config["gateway"]["current_inner_state_interval_rounds"] == 9
    assert runtime_config["gateway"]["direct_render_mode"] == "full"
    assert runtime_config["gateway"]["retrieval_mode"] == "bucket"
    assert runtime_config["gateway"]["portrait_memory_enabled"] is True
    assert runtime_config["gateway"]["portrait_memory_budget"] == 280
    assert runtime_config["gateway"]["portrait_memory_max_sources"] == 4
    assert runtime_config["gateway"]["portrait_memory_include_anchors"] is False
    assert runtime_config["gateway"]["query_planner_enabled"] is True
    assert runtime_config["gateway"]["query_planner_model"] == "planner-mini"
    assert runtime_config["gateway"]["query_planner_min_chars"] == 24
    assert runtime_config["gateway"]["query_planner_max_queries"] == 2
    assert runtime_config["gateway"]["query_planner_max_tokens"] == 256
    assert runtime_config["gateway"]["memory_detail_recall_enabled"] is True
    assert runtime_config["gateway"]["memory_detail_recall_max_ids"] == 2
    assert runtime_config["gateway"]["memory_detail_recall_budget"] == 900
    assert runtime_config["recall"]["query_resurface_enabled"] is True
    assert hot_update_calls[-1] == {
        "gateway": {
            "cooldown_hours": 6,
            "skip_recent_rounds": 5,
            "recent_context_cooldown_hours": 4,
            "recent_context_reentry_idle_hours": 24,
            "recent_context_budget": 260,
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
            "enabled": True,
            "top_k": 3,
            "min_activation": 0.22,
            "chain_walk_enabled": True,
            "chain_max_hops": 8,
            "chain_min_confidence": 0.76,
            "chain_max_frontier": 36,
        },
        "persona": {
            "enabled": False,
            "event_recording_enabled": False,
            "model": "persona-new",
            "base_url": "https://persona-new.example",
        },
        "dream": {
            "inject_enabled": True,
            "retain_after_inject": True,
        },
    }
    assert runtime_config["memory_diffusion"]["enabled"] is True
    assert runtime_config["memory_diffusion"]["top_k"] == 3
    assert runtime_config["memory_diffusion"]["min_activation"] == 0.22
    assert runtime_config["memory_diffusion"]["chain_walk_enabled"] is True
    assert runtime_config["memory_diffusion"]["chain_max_hops"] == 8
    assert runtime_config["memory_diffusion"]["chain_min_confidence"] == 0.76
    assert runtime_config["memory_diffusion"]["chain_max_frontier"] == 36
    assert runtime_config["memory_diffusion"]["max_hops"] == 2
    assert "memory_diffusion.chain_walk_enabled" in payload["updated"]
    assert "gateway_restart_required_for_memory_diffusion" not in payload["updated"]
    assert "gateway_hot_reloaded" in payload["updated"]
    assert runtime_config["reflection"]["daily_enabled"] is True
    assert runtime_config["reflection"]["enabled"] is False
    assert runtime_config["reflection"]["auto_enabled"] is False
    assert runtime_config["reflection"]["memory_affect_anchor_enabled"] is False
    assert runtime_config["reflection"]["relationship_weather_affect_anchor_enabled"] is True
    assert runtime_config["reflection"]["model"] == "reflection-new"
    assert runtime_config["reflection"]["base_url"] == "https://reflection-new.example"
    assert reflection_engine.daily_enabled is True
    assert reflection_engine.memory_affect_anchor_enabled is False
    assert reflection_engine.relationship_weather_affect_anchor_enabled is True


@pytest.mark.asyncio
async def test_config_persist_without_persona_body_preserves_minimal_persona_yaml(
    monkeypatch,
    test_config,
    tmp_path,
):
    import server

    config_path = tmp_path / "config.yaml"
    runtime_path = tmp_path / "state" / "config.runtime.yaml"
    runtime_path.parent.mkdir(exist_ok=True)
    minimal_persona = {"enabled": True, "profile_id": "lili_gege_main"}
    config_path.write_text(
        "persona:\n  enabled: true\n  profile_id: lili_gege_main\n"
        "gateway:\n  cooldown_hours: 48\n",
        encoding="utf-8",
    )
    runtime_path.write_text(
        "persona:\n  enabled: true\n  profile_id: lili_gege_main\n"
        "gateway:\n  cooldown_hours: 48\n",
        encoding="utf-8",
    )

    cfg = {
        **test_config,
        "_runtime_config_path": str(runtime_path),
        "persona": {
            **test_config["persona"],
            "profile_id": "lili_gege_main",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
            "thinking_mode": "disabled",
        },
        "gateway": {
            **test_config.get("gateway", {}),
            "cooldown_hours": 48,
        },
    }
    hot_update_calls = []

    async def fake_hot_update(body):
        hot_update_calls.append(dict(body or {}))
        return "gateway_hot_reloaded"

    monkeypatch.setenv("OMBRE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "_hot_update_gateway_config", fake_hot_update)

    response = await server.api_config_update(
        DummyRequest(
            {
                "gateway": {"cooldown_hours": 6},
                "persist": True,
            }
        )
    )
    payload = json.loads(response.body)
    saved_text = config_path.read_text(encoding="utf-8")
    runtime_text = runtime_path.read_text(encoding="utf-8")
    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime_config = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert payload["ok"] is True
    assert saved_text.index("persona:") < saved_text.index("gateway:")
    assert runtime_text.index("persona:") < runtime_text.index("gateway:")
    assert saved_config["persona"] == minimal_persona
    assert runtime_config["persona"] == minimal_persona
    assert "persona" not in hot_update_calls[-1]
    assert saved_config["gateway"]["cooldown_hours"] == 6
    assert runtime_config["gateway"]["cooldown_hours"] == 6


def test_chatgpt_oauth_provider_issues_single_use_codes():
    import server

    provider = server.ChatGptOAuthProvider(
        client_id="client",
        client_secret="secret",
        access_token="access",
        refresh_token="refresh",
        public_base_url="https://23456544321123.asia/ombre",
    )
    redirect_uri = "https://chatgpt.com/connector/oauth/test"

    code = provider.create_authorization_code(redirect_uri)

    assert provider.enabled is True
    assert provider.token_auth_methods == ["client_secret_post", "client_secret_basic"]
    assert provider.valid_redirect_uri("https://chatgpt.com/connector/oauth/test") is True
    assert provider.valid_redirect_uri("https://claude.ai/api/mcp/auth_callback") is True
    assert provider.valid_redirect_uri("https://example.com/oauth/callback") is False
    assert provider.consume_authorization_code(code, redirect_uri) is True
    assert provider.consume_authorization_code(code, redirect_uri) is False
    assert provider.valid_access_token("access") is True
    assert provider.valid_refresh_token("refresh") is True


@pytest.mark.asyncio
async def test_chatgpt_oauth_middleware_protects_only_configured_host():
    import server

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    provider = server.ChatGptOAuthProvider(
        client_id="client",
        access_token="access",
        public_base_url="https://23456544321123.asia/ombre",
    )
    middleware = server.OmbreChatGptOAuthMiddleware(app, provider, {"23456544321123.asia"})

    async def call(headers):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware(
            {"type": "http", "method": "GET", "path": "/mcp", "headers": headers},
            receive,
            send,
        )
        return next(message["status"] for message in messages if message["type"] == "http.response.start")

    assert await call([(b"host", b"23456544321123.asia")]) == 401
    assert await call([(b"host", b"8.136.154.242")]) == 204
    assert await call(
        [
            (b"host", b"23456544321123.asia"),
            (b"authorization", b"Bearer access"),
        ]
    ) == 204
