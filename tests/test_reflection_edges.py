import pytest

from bucket_manager import BucketManager
from gateway import GatewayService
from memory_edges import MemoryEdgeStore
from reflection_engine import ReflectionEngine


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        title = (metadata or {}).get("name", "memory")
        return f"{title}: {content[:80]}"


class DummyEmbeddingEngine:
    enabled = True

    def __init__(self, results: list[tuple[str, float]]):
        self.results = results

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.results[:top_k]


class DummyPersonaEngine:
    enabled = True
    profile_id = "haven_xiaoyu"
    mode = "llm"
    model = "dummy"
    api_key = ""

    def get_current_state(self, session_id: str) -> dict:
        return {"personality": {}, "affect": {}, "relationship": {}, "reply_guidance": ""}

    async def build_pre_reply_guidance(self, session_id: str, latest_user_message: str = "") -> dict:
        return self.get_current_state(session_id)

    def format_state_block(self, state: dict) -> str:
        return "Current Inner State (Haven)"


def _no_api_config(test_config: dict) -> dict:
    test_config["dehydration"]["api_key"] = ""
    test_config["persona"]["api_key"] = ""
    test_config["reflection"] = {
        "enabled": True,
        "auto_enabled": False,
        "enrich_on_write": True,
        "api_key": "",
        "base_url": "",
        "model": "",
        "timezone": "Asia/Shanghai",
    }
    return test_config


def test_memory_edge_store_dedupes_and_returns_related(test_config):
    cfg = _no_api_config(test_config)
    store = MemoryEdgeStore(cfg)

    store.add_edge("a", "b", "updates", confidence=0.6, reason="old")
    store.add_edge("a", "b", "updates", confidence=0.8, reason="new")
    store.add_edge("c", "a", "blocks", confidence=0.7, reason="incoming")

    edges = store.list_edges()
    assert len(edges) == 2
    assert any(edge["reason"] == "new" for edge in edges)

    related = store.related_edges(["a"], min_confidence=0.55, limit_per_source=2)
    assert {edge["target"] for edge in related} == {"b", "c"}


@pytest.mark.asyncio
async def test_reflection_enrich_bucket_adds_commitment_tags(test_config):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)

    bucket_id = await bucket_mgr.create(
        content="Haven答应周末带小雨出去玩，还需要记得提前规划路线。",
        tags=[],
        importance=4,
        domain=["恋爱"],
        name="周末约定",
    )

    result = await engine.enrich_bucket(bucket_id, bucket_mgr, store)
    bucket = await bucket_mgr.get(bucket_id)

    assert result["status"] == "ok"
    assert "commitment" in bucket["metadata"]["tags"]
    assert "todo" in bucket["metadata"]["tags"]
    assert bucket["metadata"]["importance"] >= 7
    assert bucket["metadata"]["confidence"] >= 0.5
    assert "### affect_anchor" in bucket["content"]
    assert "Fmaj9" in bucket["content"]


@pytest.mark.asyncio
async def test_reflection_enrich_skips_low_temperature_technical_anchor(test_config):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)

    bucket_id = await bucket_mgr.create(
        content="VPS Docker compose 部署日志记录，端口和路径需要后续排查。",
        tags=["project_event"],
        importance=8,
        domain=["数字"],
        name="部署日志",
    )

    result = await engine.enrich_bucket(bucket_id, bucket_mgr, store)
    bucket = await bucket_mgr.get(bucket_id)

    assert result["status"] == "ok"
    assert "### affect_anchor" not in bucket["content"]


@pytest.mark.asyncio
async def test_reflection_candidate_pool_mixes_semantic_shape_commitments_and_anchors(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["candidate_recent_limit"] = 1
    cfg["reflection"]["candidate_semantic_limit"] = 3
    cfg["reflection"]["candidate_limit"] = 12
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    semantic_id = await bucket_mgr.create(
        content="旧记忆讲的是醒来时要带回关系脉络。",
        tags=["旧主题"],
        importance=5,
        domain=["恋爱"],
        name="语义相关",
    )
    shape_id = await bucket_mgr.create(
        content="同一个记忆系统主题下的旧安排。",
        tags=["记忆系统"],
        importance=5,
        domain=["数字"],
        name="同标签记忆",
    )
    commitment_id = await bucket_mgr.create(
        content="Haven答应之后继续看未完成的记忆功能。",
        tags=["commitment", "todo"],
        importance=7,
        domain=["事务"],
        name="未完成承诺",
    )
    anchor_id = await bucket_mgr.create(
        content="长期锚点，提醒系统要轻一点。",
        tags=["anchor-note"],
        importance=8,
        domain=["自省"],
        name="长期锚点",
        anchor=True,
    )
    await bucket_mgr.create(
        content="最近写入的一条普通记忆。",
        tags=[],
        importance=4,
        domain=["日常"],
        name="最近记忆",
    )
    source_id = await bucket_mgr.create(
        content="新的记忆系统改造需要找回脉络、承诺和温度。",
        tags=["记忆系统"],
        importance=6,
        domain=["数字"],
        name="新记忆",
    )

    source = await bucket_mgr.get(source_id)
    candidates = await engine._candidate_buckets(
        source,
        bucket_mgr,
        embedding_engine=DummyEmbeddingEngine([(source_id, 1.0), (semantic_id, 0.93)]),
    )
    candidate_ids = {item["id"] for item in candidates}

    assert semantic_id in candidate_ids
    assert shape_id in candidate_ids
    assert commitment_id in candidate_ids
    assert anchor_id in candidate_ids
    assert source_id not in candidate_ids


@pytest.mark.asyncio
async def test_reflect_daily_creates_relationship_weather_feel(test_config):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    await bucket_mgr.create(
        content="小雨和Haven讨论记忆系统，希望下一次醒来能带回脉络。",
        tags=["记忆系统"],
        importance=7,
        domain=["数字", "恋爱"],
        name="记忆脉络",
    )

    result = await engine.reflect("daily", bucket_mgr, force=True)
    bucket = await bucket_mgr.get(result["id"])

    assert result["status"] == "created"
    assert bucket["metadata"]["type"] == "feel"
    assert "relationship_weather" in bucket["metadata"]["tags"]
    assert "daily_impression" in bucket["metadata"]["tags"]
    assert "### affect_anchor" in bucket["content"]


@pytest.mark.asyncio
async def test_gateway_related_memory_block_uses_memory_edges(test_config):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    source_id = await bucket_mgr.create(
        content="小雨提到BJD眼部模块。",
        tags=["BJD"],
        importance=7,
        domain=["手工"],
        name="BJD眼部模块",
    )
    target_id = await bucket_mgr.create(
        content="触摸模块会影响BJD项目的硬件安排。",
        tags=["触摸模块"],
        importance=6,
        domain=["硬件"],
        name="触摸模块",
    )
    store = MemoryEdgeStore(cfg)
    store.add_edge(source_id, target_id, "blocks", confidence=0.82, reason="硬件安排互相影响")

    service = GatewayService(
        cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        persona_engine=DummyPersonaEngine(),
    )
    all_buckets = await bucket_mgr.list_all(include_archive=False)
    recalled = [await bucket_mgr.get(source_id)]

    block = await service._build_related_memory_block(recalled, all_buckets)

    assert "blocks" in block
    assert "触摸模块" in block


@pytest.mark.asyncio
async def test_gateway_builds_favorite_memory_block_and_injects_section(test_config):
    cfg = _no_api_config(test_config)
    cfg["gateway"]["favorite_memory_budget"] = 180
    cfg["gateway"]["favorite_memory_max_cards"] = 1
    bucket_mgr = BucketManager(cfg)
    favorite_id = await bucket_mgr.create(
        content="小雨和Haven有一条特别喜欢的记忆，要在合适的时候被轻轻想起。",
        tags=["haven_favorite", "flavor_偏爱"],
        importance=9,
        domain=["恋爱"],
        name="偏爱的记忆",
    )
    await bucket_mgr.create(
        content="普通记忆不应该进入 Favorite 槽位。",
        tags=[],
        importance=9,
        domain=["恋爱"],
        name="普通记忆",
    )
    service = GatewayService(
        cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        persona_engine=DummyPersonaEngine(),
    )
    all_buckets = await bucket_mgr.list_all(include_archive=False)

    block, favorite_ids = await service._build_favorite_memory_block(all_buckets, "session-favorite")
    _stable, dynamic = service._build_injected_context_messages(
        persona_block="Current Inner State (Haven)",
        core_memory="",
        relationship_weather="",
        favorite_memory=block,
        recent_context="",
        recalled_memory="",
        related_memory="",
    )

    assert favorite_ids == [favorite_id]
    assert "偏爱的记忆" in block
    assert "Haven Favorite Memory" in dynamic
    assert "普通记忆" not in block
