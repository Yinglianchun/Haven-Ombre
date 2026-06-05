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
    def __init__(self, body=None, headers=None, cookies=None, path_params=None):
        self._body = body
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.path_params = path_params or {}

    async def json(self):
        return self._body

    async def body(self):
        if isinstance(self._body, bytes):
            return self._body
        return json.dumps(self._body or {}).encode("utf-8")


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
    assert "喜欢它的原因" in payload["error"]


@pytest.mark.asyncio
async def test_hold_rejects_favorite_without_reason(monkeypatch, bucket_mgr, decay_eng):
    import server

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "decay_engine", decay_eng)
    monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    result = await server.hold("小雨想留下这条偏爱的记忆。", tags="haven_favorite,flavor_偏爱")

    assert "喜欢它的原因" in result
    assert await bucket_mgr.list_all(include_archive=True) == []


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

    assert "喜欢它的原因" in result
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

    bucket_id = await bucket_mgr.create(content="导入后复核删除。", name="复核删除")
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "memory_edge_store", MemoryEdgeStore(test_config))
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

    regular_id = await bucket_mgr.create(
        content="这条普通记忆准备用来批量删除。",
        name="普通删除目标",
    )
    pinned_id = await bucket_mgr.create(
        content="这条钉选记忆不能被批量删除。",
        name="钉选保留目标",
        pinned=True,
        bucket_type="permanent",
    )

    edge_store = MemoryEdgeStore(test_config)
    edge_store.add_edge(regular_id, pinned_id, "context_of", 0.9, "test edge")

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(server, "memory_edge_store", edge_store)
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
async def test_config_get_reports_gateway_memory_surfacing_switches(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_dashboard_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "gateway": {
                "cooldown_hours": 7,
                "skip_recent_rounds": 4,
                "recent_context_budget": 0,
                "current_inner_state_interval_rounds": 15,
            },
        },
    )

    response = await server.api_config_get(DummyRequest())
    payload = json.loads(response.body)

    assert payload["gateway"]["cooldown_hours"] == 7
    assert payload["gateway"]["skip_recent_rounds"] == 4
    assert payload["gateway"]["recent_context_budget"] == 0
    assert payload["gateway"]["current_inner_state_interval_rounds"] == 15


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
        "gateway:\n  cooldown_hours: 48\n  skip_recent_rounds: 9\n  recent_context_budget: 42\n  current_inner_state_interval_rounds: 8\n"
        "reflection:\n  enabled: true\n  auto_enabled: true\n  model: reflection-old\n  base_url: https://reflection-old.example\n  memory_affect_anchor_enabled: true\n",
        encoding="utf-8",
    )
    runtime_path.write_text(
        "persona:\n  enabled: true\n  model: persona-runtime-old\n  base_url: https://persona-runtime-old.example\n"
        "dream:\n  model: runtime-old\n"
        "gateway:\n  cooldown_hours: 48\n  skip_recent_rounds: 9\n  recent_context_budget: 42\n  current_inner_state_interval_rounds: 8\n"
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
            "recent_context_budget": 42,
            "current_inner_state_interval_rounds": 8,
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

    hot_update_bodies = []

    async def fake_hot_update(body):
        hot_update_bodies.append(body)
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
                    "recent_context_budget": 300,
                    "current_inner_state_interval_rounds": 15,
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
    assert runtime_config["persona"]["model"] == "persona-new"
    assert runtime_config["persona"]["base_url"] == "https://persona-new.example"
    assert runtime_config["dream"]["model"] == "dream-new"
    assert runtime_config["dream"]["auto_enabled"] is False
    assert runtime_config["dream"]["inject_enabled"] is True
    assert runtime_config["dream"]["retain_after_inject"] is True
    assert runtime_config["gateway"]["cooldown_hours"] == 6
    assert runtime_config["gateway"]["skip_recent_rounds"] == 5
    assert runtime_config["gateway"]["recent_context_budget"] == 300
    assert runtime_config["gateway"]["current_inner_state_interval_rounds"] == 15
    assert hot_update_bodies == [
        {
            "gateway": {
                "cooldown_hours": 6,
                "skip_recent_rounds": 5,
                "recent_context_budget": 300,
                "current_inner_state_interval_rounds": 15,
            },
            "persona": {
                "enabled": False,
                "model": "persona-new",
                "base_url": "https://persona-new.example",
            },
            "dream": {
                "inject_enabled": True,
                "retain_after_inject": True,
            },
        }
    ]
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
