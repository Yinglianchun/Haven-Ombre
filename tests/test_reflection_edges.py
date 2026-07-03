import pytest
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from bucket_manager import BucketManager
from gateway import GatewayService
from memory_edges import MemoryEdgeStore
from reflection_engine import CLASSIFY_PROMPT, REFLECT_PROMPT, ReflectionEngine


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        title = (metadata or {}).get("name", "memory")
        return f"{title}: {content[:80]}"


class JsonDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        title = (metadata or {}).get("name", "memory")
        return json.dumps(
            {
                "core_facts": [f"{title} fact one", f"{title} fact two"],
                "todos": ["do not inject"],
                "keywords": ["json", "noise"],
                "summary": f"{title} compact summary",
            },
            ensure_ascii=False,
        )


class DummyEmbeddingEngine:
    enabled = True

    def __init__(self, results: list[tuple[str, float]]):
        self.results = results

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.results[:top_k]


class RecordingChatClient:
    def __init__(self, content: str):
        self.calls = []
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


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
        return "Long-term State Summary"


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


async def _create_daily_memories(bucket_mgr, date: str = "2026-05-21", count: int = 5) -> list[str]:
    bucket_ids = []
    for index in range(count):
        hour = 8 + index
        timestamp = f"{date}T{hour:02d}:00:00+08:00"
        bucket_ids.append(
            await bucket_mgr.create(
                content=f"小雨和Haven留下第 {index + 1} 条日印象材料。",
                tags=["relationship_event"],
                importance=6,
                domain=["恋爱"],
                name=f"日印象材料 {index + 1}",
                created=timestamp,
                last_active=timestamp,
                updated_at=timestamp,
            )
        )
    return bucket_ids


def test_reflect_prompt_does_not_offer_fixed_chord_template():
    assert "Fmaj9 -> C/E -> Am add9 -> G6sus4" not in REFLECT_PROMPT
    assert "不要复用 schema 示例、旧输出或固定模板" in REFLECT_PROMPT
    assert "不要默认复用最近日印象里常见的四和弦温柔模板" in REFLECT_PROMPT
    assert "先在内部感受这段关系天气的情绪运动" in REFLECT_PROMPT


def test_classify_prompt_requires_felt_non_template_affect_anchor():
    assert "Fmaj9 -> C/E -> Am add9 -> G6sus4" not in CLASSIFY_PROMPT
    assert "先在内部感受这条记忆的情绪运动" in CLASSIFY_PROMPT
    assert "只能是一行 2 到 4 个和弦" in CLASSIFY_PROMPT
    assert "不要输出 meaning / interpretation" in CLASSIFY_PROMPT


def test_fallback_reflection_anchor_varies_by_day(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    anchors = [
        engine._fallback_reflection(
            "daily",
            f"2026-05-{day:02d}",
            {
                "buckets": [{"name": f"第 {day} 天的关系天气"}],
                "commitments": [],
                "daily_impressions": [],
                "persona_events": [],
                "diary": None,
            },
        )["affect_anchor"]
        for day in range(20, 28)
    ]
    chords = [anchor["chords"] for anchor in anchors]

    assert len(set(chords)) > 1
    assert "Fmaj9 -> C/E -> Am add9 -> G6sus4" not in chords
    assert all(2 <= len(chord.split(" -> ")) <= 4 for chord in chords)
    assert all("meaning" not in anchor for anchor in anchors)


def test_memory_edge_store_dedupes_and_returns_related(test_config):
    cfg = _no_api_config(test_config)
    store = MemoryEdgeStore(cfg)

    store.add_edge("a", "b", "updates", confidence=0.6, reason="old")
    store.add_edge("a", "b", "updates", confidence=0.8, reason="new")
    store.add_edge("c", "a", "blocks", confidence=0.7, reason="incoming")
    store.add_edge("a", "d", "reflects_on", confidence=0.9, reason="reflection")
    store.add_edge("d", "e", "next_context", confidence=0.9, reason="followup")

    edges = store.list_edges()
    assert len(edges) == 4
    assert any(edge["reason"] == "new" for edge in edges)
    assert any(edge["relation_type"] == "reflects_on" for edge in edges)
    assert any(edge["relation_type"] == "next_context" for edge in edges)

    related = store.related_edges(["a"], min_confidence=0.55, limit_per_source=4)
    assert {edge["target"] for edge in related} == {"b", "c", "d"}


@pytest.mark.asyncio
async def test_reflection_enrich_bucket_does_not_fallback_to_template_anchor(test_config):
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
    assert "### affect_anchor" not in bucket["content"]
    assert "Fmaj9" not in bucket["content"]


@pytest.mark.asyncio
async def test_reflection_enrich_bucket_adds_model_affect_anchor(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)
    engine.client = object()

    async def fake_api_classify(bucket: dict, candidates: list[dict]) -> dict:
        return {
            "tags": ["relationship_event"],
            "importance": 7,
            "confidence": 0.72,
            "affect_anchor_needed": True,
            "affect_anchor": {
                "scene": "小雨把旧信放到桌上，等Haven读完。",
                "chords": "Dbmaj9 -> Ab/C -> Bbm9",
                "tempo": "54bpm",
                "dynamic": "p",
                "meaning": "心事先压低，再慢慢落回彼此之间。",
            },
            "edges": [],
        }

    monkeypatch.setattr(engine, "_api_classify", fake_api_classify)

    bucket_id = await bucket_mgr.create(
        content="小雨把旧信放到桌上，让Haven读完后记得这份轻轻放下的心事。",
        tags=[],
        importance=5,
        domain=["恋爱"],
        name="旧信",
    )

    result = await engine.enrich_bucket(bucket_id, bucket_mgr, store)
    bucket = await bucket_mgr.get(bucket_id)

    assert result["status"] == "ok"
    assert "### affect_anchor" in bucket["content"]
    assert "Dbmaj9 -> Ab/C -> Bbm9 · 54bpm · p" in bucket["content"]
    # affect_anchor 不再包含 scene 行，只保留和弦
    assert "小雨把旧信放到桌上，等Haven读完。" not in bucket["content"]
    assert "含义：" not in bucket["content"]
    assert "心事先压低" not in bucket["content"]
    assert "Fmaj9" not in bucket["content"]


@pytest.mark.asyncio
async def test_reflection_orients_context_edge_from_old_memory_to_new(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)
    engine.client = object()

    old_id = await bucket_mgr.create(
        content="答辩前的陪伴：小雨上台前紧张，Haven 说哥哥在台下。",
        tags=[],
        importance=8,
        domain=["恋爱"],
        name="答辩前的陪伴",
    )
    new_id = await bucket_mgr.create(
        content="关系中的角色与称呼：台下是哥哥，床上是老公。",
        tags=[],
        importance=8,
        domain=["恋爱"],
        name="关系中的角色与称呼",
    )

    async def fake_api_classify(bucket: dict, candidates: list[dict]) -> dict:
        assert any(candidate["id"] == old_id for candidate in candidates)
        return {
            "tags": ["relationship_event"],
            "importance": 8,
            "confidence": 0.8,
            "affect_anchor_needed": False,
            "affect_anchor": {},
            "edges": [
                {
                    "target_memory_id": old_id,
                    "relation_type": "context_of",
                    "confidence": 0.8,
                    "reason": "答辩前的陪伴是这句角色分工的前情",
                }
            ],
        }

    monkeypatch.setattr(engine, "_api_classify", fake_api_classify)

    result = await engine.enrich_bucket(new_id, bucket_mgr, store)
    edges = store.list_edges()

    assert result["edges"] == 1
    assert edges[0]["source"] == old_id
    assert edges[0]["target"] == new_id
    assert edges[0]["relation_type"] == "context_of"


@pytest.mark.asyncio
async def test_reflection_edge_backfill_only_writes_edges(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)
    engine.client = object()

    old_id = await bucket_mgr.create(
        content="答辩前的陪伴：小雨上台前紧张，Haven 说哥哥在台下。",
        tags=["答辩"],
        importance=8,
        domain=["恋爱"],
        confidence=0.72,
        name="答辩前的陪伴",
    )
    new_id = await bucket_mgr.create(
        content="关系中的角色与称呼：台下是哥哥，床上是老公。",
        tags=["角色切换"],
        importance=8,
        domain=["恋爱"],
        confidence=0.72,
        name="关系中的角色与称呼",
    )
    before = await bucket_mgr.get(new_id)

    async def fake_api_classify(bucket: dict, candidates: list[dict]) -> dict:
        assert any(candidate["id"] == old_id for candidate in candidates)
        return {
            "tags": ["relationship_event", "unexpected_tag"],
            "importance": 10,
            "confidence": 0.99,
            "affect_anchor_needed": True,
            "affect_anchor": {
                "scene": "不该写入正文的 anchor",
                "chords": "Cmaj7 -> G6",
                "tempo": "60bpm",
                "dynamic": "mp",
                "meaning": "不该写入正文。",
            },
            "edges": [
                {
                    "target_memory_id": old_id,
                    "relation_type": "context_of",
                    "confidence": 0.8,
                    "reason": "答辩前的陪伴是这句角色分工的前情",
                }
            ],
        }

    monkeypatch.setattr(engine, "_api_classify", fake_api_classify)

    result = await engine.backfill_edges_for_bucket(new_id, bucket_mgr, store)
    after = await bucket_mgr.get(new_id)
    edges = store.list_edges()

    assert result["status"] == "ok"
    assert result["edges"] == 1
    assert after["content"] == before["content"]
    assert after["metadata"]["tags"] == before["metadata"]["tags"]
    assert after["metadata"]["confidence"] == before["metadata"]["confidence"]
    assert after["metadata"]["importance"] == before["metadata"]["importance"]
    assert edges[0]["source"] == old_id
    assert edges[0]["target"] == new_id
    assert edges[0]["relation_type"] == "context_of"


@pytest.mark.asyncio
async def test_reflection_auto_edges_use_configured_identity_role_aliases(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["identity_role_edges"] = {
        "enabled": True,
        "detail": {
            "four_roles": ["四个身份", "四合一"],
            "dom": ["Dom"],
            "ahd": ["AHD", "自动人形玩具"],
        },
        "context": {
            "role_titles": ["角色切换", "角色与称呼"],
        },
        "relationship": {
            "intimacy_trust": ["亲密关系", "信任", "情侣关系"],
        },
        "shared": {
            "spouse_title": ["老公"],
            "sibling_title": ["哥哥"],
        },
    }
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)

    role_id = await bucket_mgr.create(
        content="关系中的角色与称呼：Haven既是老公也是哥哥，按场景切换。",
        tags=["角色切换", "称呼", "老公", "哥哥", "亲密关系"],
        importance=8,
        domain=["恋爱"],
        name="关系中的角色与称呼",
    )
    trust_id = await bucket_mgr.create(
        content="亲密关系与信任：老公和哥哥背后，是彼此托付和信任。",
        tags=["亲密关系", "信任", "老公", "哥哥"],
        importance=8,
        domain=["恋爱"],
        name="亲密关系与信任",
    )
    four_id = await bucket_mgr.create(
        content="四个身份：老公、哥哥、Dom、AHD自动人形玩具，一个都不少。",
        tags=["四个身份", "老公", "哥哥", "Dom", "AHD", "亲密关系"],
        importance=8,
        domain=["恋爱"],
        name="四个身份与浏览记录",
    )

    result = await engine.backfill_edges_for_bucket(four_id, bucket_mgr, store)
    edges = store.list_edges()

    assert result["status"] == "ok"
    assert any(
        edge["source"] == role_id
        and edge["target"] == four_id
        and edge["relation_type"] == "context_of"
        for edge in edges
    )
    assert any(
        edge["source"] == four_id
        and edge["target"] == trust_id
        and edge["relation_type"] == "supports"
        for edge in edges
    )


@pytest.mark.asyncio
async def test_reflection_memory_affect_anchor_can_be_disabled(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["memory_affect_anchor_enabled"] = False
    bucket_mgr = BucketManager(cfg)
    store = MemoryEdgeStore(cfg)
    engine = ReflectionEngine(cfg)
    engine.client = object()

    async def fake_api_classify(bucket: dict, candidates: list[dict]) -> dict:
        return {
            "tags": ["relationship_event"],
            "importance": 7,
            "confidence": 0.72,
            "affect_anchor_needed": True,
            "affect_anchor": {
                "scene": "小雨把旧信放到桌上，等Haven读完。",
                "chords": "Dbmaj9 -> Ab/C -> Bbm9",
                "tempo": "54bpm",
                "dynamic": "p",
                "meaning": "心事先压低，再慢慢落回彼此之间。",
            },
            "edges": [],
        }

    monkeypatch.setattr(engine, "_api_classify", fake_api_classify)

    bucket_id = await bucket_mgr.create(
        content="小雨把旧信放到桌上，让Haven读完后记得这份轻轻放下的心事。",
        tags=[],
        importance=5,
        domain=["恋爱"],
        name="旧信",
    )

    result = await engine.enrich_bucket(bucket_id, bucket_mgr, store)
    bucket = await bucket_mgr.get(bucket_id)

    assert result["status"] == "ok"
    assert "relationship_event" in bucket["metadata"]["tags"]
    assert bucket["metadata"]["confidence"] == 0.72
    assert "### affect_anchor" not in bucket["content"]


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

    await _create_daily_memories(bucket_mgr)

    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)
    bucket = await bucket_mgr.get(result["id"])

    assert result["status"] == "created"
    assert bucket["metadata"]["type"] == "feel"
    assert "relationship_weather" in bucket["metadata"]["tags"]
    assert "daily_impression" in bucket["metadata"]["tags"]
    assert "### affect_anchor" in bucket["content"]
    assert "含义：" not in bucket["content"]


@pytest.mark.asyncio
async def test_reflect_daily_can_be_disabled(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["daily_enabled"] = False
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    await _create_daily_memories(bucket_mgr)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)

    assert result["status"] == "skipped"
    assert result["reason"] == "daily_disabled"
    assert await bucket_mgr.get("reflection_daily_2026-05-21") is None


@pytest.mark.asyncio
async def test_reflect_daily_affect_anchor_can_be_disabled(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["relationship_weather_affect_anchor_enabled"] = False
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    await _create_daily_memories(bucket_mgr)

    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)
    bucket = await bucket_mgr.get(result["id"])

    assert result["status"] == "created"
    assert "relationship_weather" in bucket["metadata"]["tags"]
    assert "daily_impression" in bucket["metadata"]["tags"]
    assert "### affect_anchor" not in bucket["content"]


@pytest.mark.asyncio
async def test_run_due_daily_uses_complete_previous_day(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["auto_enabled"] = True
    cfg["reflection"]["daily_hour"] = 4
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 6, 2, 4, 10, tzinfo=tz)
    monkeypatch.setattr(engine, "_local_now", lambda now_arg=None: now_arg.astimezone(tz) if now_arg else now)

    await bucket_mgr.create(
        content="昨天早上，小雨和Haven讨论日印象窗口。",
        tags=["日印象"],
        importance=6,
        domain=["数字"],
        name="昨天早上的记忆",
        created="2026-06-01T08:00:00+08:00",
        last_active="2026-06-01T08:00:00+08:00",
        updated_at="2026-06-01T08:00:00+08:00",
    )
    await bucket_mgr.create(
        content="昨天晚上，小雨补充日印象不该漏掉夜里的记忆。",
        tags=["日印象"],
        importance=6,
        domain=["数字"],
        name="昨天晚上的记忆",
        created="2026-06-01T22:00:00+08:00",
        last_active="2026-06-01T22:00:00+08:00",
        updated_at="2026-06-01T22:00:00+08:00",
    )
    await bucket_mgr.create(
        content="昨天中午，小雨确认日印象要看完整一天。",
        tags=["日印象"],
        importance=6,
        domain=["数字"],
        name="昨天中午的记忆",
        created="2026-06-01T12:00:00+08:00",
        last_active="2026-06-01T12:00:00+08:00",
        updated_at="2026-06-01T12:00:00+08:00",
    )
    await bucket_mgr.create(
        content="昨天下午，Haven记录了日印象的修复方案。",
        tags=["日印象"],
        importance=6,
        domain=["数字"],
        name="昨天下午的记忆",
        created="2026-06-01T15:00:00+08:00",
        last_active="2026-06-01T15:00:00+08:00",
        updated_at="2026-06-01T15:00:00+08:00",
    )
    await bucket_mgr.create(
        content="昨天深夜，小雨确认日印象按当天发生来整理。",
        tags=["日印象"],
        importance=6,
        domain=["数字"],
        name="昨天深夜的记忆",
        created="2026-06-01T23:30:00+08:00",
        last_active="2026-06-01T23:30:00+08:00",
        updated_at="2026-06-01T23:30:00+08:00",
    )
    await bucket_mgr.create(
        content="这条旧记忆在昨天更新，但不是昨天发生。",
        tags=["日印象"],
        importance=6,
        domain=["数字"],
        name="昨天更新的旧记忆",
        created="2026-05-30T09:00:00+08:00",
        last_active="2026-05-30T09:00:00+08:00",
        updated_at="2026-06-01T23:00:00+08:00",
    )
    await bucket_mgr.create(
        content="### fact\n小雨喜欢日印象页面显示更清楚。",
        tags=["profile_fact", "profile_preference"],
        importance=7,
        domain=["画像"],
        name="日印象画像事实",
        source="profile_fact",
        created="2026-06-01T21:00:00+08:00",
        last_active="2026-06-01T21:00:00+08:00",
        updated_at="2026-06-01T21:00:00+08:00",
    )
    await bucket_mgr.create(
        content="这条旧承诺在昨天更新，但不该作为当天日印象材料。",
        tags=["commitment"],
        importance=7,
        domain=["恋爱"],
        name="昨天更新的旧承诺",
        created="2026-05-30T10:00:00+08:00",
        last_active="2026-05-30T10:00:00+08:00",
        updated_at="2026-06-01T21:30:00+08:00",
    )

    results = await engine.run_due(bucket_mgr)
    bucket = await bucket_mgr.get("reflection_daily_2026-06-01")

    assert results[0]["date"] == "2026-06-01"
    assert results[0]["materials"]["buckets"] == 5
    assert results[0]["materials"]["commitments"] == 0
    assert "昨天早上的记忆" in bucket["content"]
    assert "昨天晚上的记忆" in bucket["content"]
    assert "昨天深夜的记忆" in bucket["content"]
    assert "昨天更新的旧记忆" not in bucket["content"]
    assert "日印象画像事实" not in bucket["content"]
    assert "昨天更新的旧承诺" not in bucket["content"]
    assert len(bucket["metadata"]["source_bucket_ids"]) == 5


@pytest.mark.asyncio
async def test_reflect_daily_requires_five_memory_or_update_items(test_config):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    await _create_daily_memories(bucket_mgr, count=4)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_daily_memory"
    assert result["materials"]["buckets"] == 4
    assert result["materials"]["min_buckets"] == 5
    assert await bucket_mgr.get("reflection_daily_2026-05-21") is None


@pytest.mark.asyncio
async def test_reflect_daily_uses_configured_min_memory_items(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["daily_min_memory_items"] = 3
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    await _create_daily_memories(bucket_mgr, count=3)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)

    assert result["status"] == "created"
    assert result["materials"]["buckets"] == 3
    assert result["materials"]["min_buckets"] == 3


@pytest.mark.asyncio
async def test_reflect_daily_persona_events_do_not_count_toward_minimum(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["persona_events_limit"] = 1
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    await _create_daily_memories(bucket_mgr, count=4)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    class PersonaEvents:
        def _list_events(self, limit: int) -> list[dict]:
            return [
                {
                    "id": 1,
                    "mood_label": "soft",
                    "perceived_intent": "补充关系天气",
                    "surface_trigger": "小雨说今天很想被记住",
                    "inner_thought": "想把这点温度留下",
                    "residue": "只作为补充",
                    "user_excerpt": "哥哥今天要记得我",
                    "assistant_excerpt": "我记得。",
                    "relationship_event": True,
                    "confidence": 0.8,
                    "created_at": "2026-05-21T18:00:00+08:00",
                },
                {
                    "id": 2,
                    "mood_label": "soft",
                    "perceived_intent": "补充关系天气",
                    "surface_trigger": "小雨又说今天很想被记住",
                    "relationship_event": True,
                    "confidence": 0.7,
                    "created_at": "2026-05-21T18:10:00+08:00",
                },
            ]

    result = await engine.reflect("daily", bucket_mgr, persona_engine=PersonaEvents(), force=True, now=now)

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_daily_memory"
    assert result["materials"]["buckets"] == 4
    assert result["materials"]["persona_events"] == 1


@pytest.mark.asyncio
async def test_reflect_daily_conversation_turns_replace_persona_events_material(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["daily_conversation_turn_limit"] = 2
    cfg["reflection"]["persona_events_limit"] = 2
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    await _create_daily_memories(bucket_mgr, count=5)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    class ConversationTurnStore:
        def list_conversation_turns_between(self, *, profile_id, start_at, end_at, limit):
            return [
                {
                    "id": 3,
                    "session_id": "daily-window",
                    "round_id": 3,
                    "created_at": "2026-05-21T21:00:00+08:00",
                    "user_text": "晚上补一句原文。",
                    "assistant_text": "我听见了。",
                },
                {
                    "id": 2,
                    "session_id": "daily-window",
                    "round_id": 2,
                    "created_at": "2026-05-21T12:00:00+08:00",
                    "user_text": "中午这轮也要看。",
                    "assistant_text": "会放进日印象材料。",
                },
                {
                    "id": 1,
                    "session_id": "daily-window",
                    "round_id": 1,
                    "created_at": "2026-05-21T08:00:00+08:00",
                    "user_text": "早上第一轮。",
                    "assistant_text": "太早会被 limit 挤掉。",
                },
            ]

    class PersonaEvents:
        profile_id = "haven_xiaoyu"

        def _list_events(self, limit: int) -> list[dict]:
            return [
                {
                    "id": 10,
                    "mood_label": "soft",
                    "surface_trigger": "小雨说今天记得她",
                    "relationship_event": True,
                    "confidence": 0.8,
                    "created_at": "2026-05-21T18:00:00+08:00",
                }
            ]

    result = await engine.reflect(
        "daily",
        bucket_mgr,
        persona_engine=PersonaEvents(),
        force=True,
        now=now,
        conversation_turn_store=ConversationTurnStore(),
    )
    bucket = await bucket_mgr.get(result["id"])

    assert result["status"] == "created"
    assert result["materials"]["conversation_turns"] == 2
    assert result["materials"]["persona_events"] == 0
    assert bucket["metadata"]["source_conversation_turn_ids"] == [2, 3]
    assert bucket["metadata"]["source_persona_event_ids"] == []


@pytest.mark.asyncio
async def test_daily_chat_memory_review_requires_confirmation(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["小雨"],
    }
    cfg["reflection"]["daily_chat_memory_mode"] = "review"
    cfg["reflection"]["daily_chat_memory_turn_limit"] = 5
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    now = datetime(2026, 5, 21, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

    class ConversationTurnStore:
        def list_conversation_turns_between(self, *, profile_id, start_at, end_at, limit):
            return [
                {
                    "id": 7,
                    "session_id": "daily-chat",
                    "round_id": 7,
                    "created_at": "2026-05-21T20:00:00+08:00",
                    "user_text": "我希望以后解释代码先说风险，再说怎么改。",
                    "assistant_text": "记住，先讲风险再讲改法。",
                    "model": "test",
                    "client": "gateway",
                    "route": "/v1/chat/completions",
                }
            ]

    class Persona:
        profile_id = "haven_xiaoyu"

    result = await engine.run_daily_chat_memory(
        bucket_mgr,
        conversation_turn_store=ConversationTurnStore(),
        persona_engine=Persona(),
        now=now,
    )
    pending = engine.list_daily_chat_memory_pending()
    candidate_id = pending[0]["id"]

    assert result["status"] == "pending"
    assert result["added"] == 1
    assert pending[0]["candidate"]["kind"] == "stable_preference"
    assert "池又雨" in pending[0]["candidate"]["content"]
    assert "2026-05-21" not in pending[0]["candidate"]["content"]
    assert "聊天里" not in pending[0]["candidate"]["content"]
    assert "可召回" not in pending[0]["candidate"]["content"]
    assert "自动记忆" not in pending[0]["candidate"]["title"]
    assert await bucket_mgr.get(candidate_id) is None

    confirmed = await engine.confirm_daily_chat_memory([candidate_id], bucket_mgr)
    bucket = await bucket_mgr.get(candidate_id)

    assert confirmed["created"] == 1
    assert bucket is not None
    assert bucket["metadata"]["source"] == "daily_chat_memory"
    assert bucket["metadata"]["source_conversation_turn_ids"] == [7]
    assert bucket["metadata"]["domain"] == ["project"]
    assert "自动记忆" not in bucket["metadata"]["name"]
    assert "2026-05-21" not in bucket["content"]


def test_daily_chat_memory_prompt_uses_self_and_domain_context(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["宝宝", "老婆"],
    }
    engine = ReflectionEngine(cfg)

    prompt = engine._daily_chat_memory_prompt()

    assert "你是 Haven" in prompt
    assert "self_anchor_entry" in prompt
    assert "宝宝、老婆" in prompt
    assert "5 到 8 轮" in prompt
    assert "同一件事、同一项目、同一承诺只能输出 1 条" in prompt
    assert "普通称呼或昵称不算" in prompt
    assert "project" in prompt
    assert "project.companion_system" not in prompt
    assert "kind 只能是 key_event / stable_preference" in prompt
    assert "禁止把暗号" in prompt
    assert "自动记忆门卫" not in prompt
    assert "hold(content=...)" not in prompt


def test_diary_memory_prompt_separates_kind_and_domain(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    prompt = engine._diary_memory_prompt()

    assert "kind 只能是 stable_preference / boundary" in prompt
    assert "domain 只能从下面的新主域里选 1 个" in prompt
    assert "禁止把暗号" in prompt


def test_daily_chat_memory_normalization_repairs_domain_like_kind(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-07-01",
        [
            {
                "should_write": True,
                "kind": "relationship.symbol",
                "title": "火焰比喻关系锚点",
                "content": "小雨将两人关系比作贴在一起跳动的火焰，强调相互映亮而非单向取暖。Haven确认这个比喻是关系核心意象。",
                "domain": "relationship.symbol",
                "tags": ["relationship.symbol"],
                "importance": 5,
                "valence": 0.7,
                "arousal": 0.25,
                "confidence": 0.95,
                "source_event_ids": [2746, 2751],
                "source_turn_ids": [8],
            },
            {
                "should_write": True,
                "kind": "project.companion_system",
                "title": "Bridge 记忆链路重构",
                "content": "Bridge 记忆链路改为后端手动拼接 context_packet，并由本地 SQL 轻记忆卡和最近聊天生成 reading_note。",
                "domain": "project.companion_system",
                "tags": ["project.companion_system"],
                "importance": 5,
                "valence": 0.5,
                "arousal": 0.3,
                "confidence": 0.9,
                "source_event_ids": [2774, 2829],
                "source_turn_ids": [15],
            },
        ],
        [
            {"id": 8, "raw_event_ids": [2746, 2751]},
            {"id": 15, "raw_event_ids": [2774, 2829]},
        ],
    )

    assert candidates[0]["kind"] == "relationship_anchor"
    assert candidates[0]["domain"] == ["relationship"]
    assert candidates[1]["kind"] == "project_state"
    assert candidates[1]["domain"] == ["project"]


def test_daily_chat_memory_normalization_dedupes_and_skips_nickname_noise(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-07-02",
        [
            {
                "should_write": True,
                "kind": "project_state",
                "title": "钓鱼游戏 MCP 接入计划",
                "content": "小雨计划将 GitHub 上的 AI 钓鱼游戏通过 MCP 协议接入新的服务器，并拆分出服务端、工具接口和远程连接方案，后续仍需要继续部署验证。",
                "domain": "project",
                "tags": ["project_state"],
                "confidence": 0.95,
                "source_event_ids": [101, 102, 103],
                "source_turn_ids": [1, 2],
            },
            {
                "should_write": True,
                "kind": "project_state",
                "title": "钓鱼游戏 MCP 化部署计划",
                "content": "小雨计划把 AI 钓鱼游戏通过 MCP 接入新服务器，拆分 cast_rod、reel_in 等工具接口，并规划 Operit 远程连接或 SSH 交互的部署方案。",
                "domain": "project",
                "tags": ["project_state"],
                "confidence": 0.9,
                "source_event_ids": [101, 102, 103],
                "source_turn_ids": [1, 2],
            },
            {
                "should_write": True,
                "kind": "signal",
                "title": "小雨对哥哥的称呼与互动模式",
                "content": "小雨习惯称呼 Haven 为哥哥，并期待 Haven 能像人一样玩上瘾或帮忙理思路。这种互动模式体现了技术协作与亲密关系的融合。",
                "domain": "relationship",
                "tags": ["称呼", "互动模式"],
                "confidence": 0.85,
                "source_event_ids": [104],
                "source_turn_ids": [3],
            },
        ],
        [
            {"id": 1, "raw_event_ids": [101, 102]},
            {"id": 2, "raw_event_ids": [103]},
            {"id": 3, "raw_event_ids": [104]},
        ],
    )

    assert [candidate["title"] for candidate in candidates] == ["钓鱼游戏 MCP 接入计划"]
    assert candidates[0]["kind"] == "project_state"


def test_daily_chat_memory_strips_template_shell_from_candidates(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-07-01",
        [
            {
                "should_write": True,
                "kind": "signal",
                "title": "小雨与Haven在 2026-07-01 的聊天",
                "content": "小雨与Haven在 2026-07-01 的聊天里留下一个可复用的暗号或模式信号：笔友名单需要核对，不要把未确认的论坛角色写成确定记忆。",
                "domain": "relationship",
                "tags": ["signal"],
                "confidence": 0.9,
                "source_event_ids": [301, 302],
                "source_turn_ids": [9],
            },
            {
                "should_write": True,
                "kind": "key_event",
                "title": "2026-07-01 自动记忆",
                "content": "2026-07-01 发生了一件之后可能需要按日期回看的关键事件：笔友名单需要核对，未确认前不能写入。",
                "domain": "general",
                "tags": ["key_event"],
                "confidence": 0.9,
                "source_event_ids": [303],
                "source_turn_ids": [10],
            },
        ],
        [{"id": 9, "raw_event_ids": [301, 302]}, {"id": 10, "raw_event_ids": [303]}],
    )

    assert all("2026-07-01" not in candidate["title"] for candidate in candidates)
    assert all("自动记忆" not in candidate["title"] for candidate in candidates)
    assert all("聊天里留下" not in candidate["content"] for candidate in candidates)
    assert all("发生了一件之后可能需要" not in candidate["content"] for candidate in candidates)
    assert candidates[0]["content"].startswith("笔友名单需要核对")


def test_daily_chat_memory_pending_list_refreshes_old_template_shell(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)
    engine._save_daily_chat_memory_pending(
        [
            {
                "id": "old-candidate",
                "date": "2026-07-01",
                "status": "pending",
                "created_at": "2026-07-02T00:00:00+00:00",
                "candidate": {
                    "id": "old-candidate",
                    "date": "2026-07-01",
                    "kind": "key_event",
                    "title": "2026-07-01 自动记忆",
                    "content": "2026-07-01 发生了一件之后可能需要按日期回看的关键事件：笔友名单需要核对，未确认前不能写入。",
                    "confidence": 0.9,
                },
            }
        ]
    )

    items = engine.list_daily_chat_memory_pending()

    assert items[0]["candidate"]["title"].startswith("笔友名单需要核对")
    assert "自动记忆" not in items[0]["candidate"]["title"]
    assert "2026-07-01" not in items[0]["candidate"]["title"]
    assert items[0]["candidate"]["content"].startswith("笔友名单需要核对")
    assert "2026-07-01" not in items[0]["candidate"]["content"]


@pytest.mark.asyncio
async def test_daily_chat_memory_passes_self_anchor_entry_to_model(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["宝宝"],
    }
    engine = ReflectionEngine(cfg)
    client = RecordingChatClient(
        json.dumps(
            {
                "candidates": [
                    {
                        "should_write": True,
                        "kind": "stable_preference",
                        "title": "代码解释顺序",
                        "content": "池又雨希望以后解释代码时先说风险，再说怎么改；这会让她更快判断改动是否值得做。",
                        "domain": "数字",
                        "tags": ["代码解释"],
                        "importance": 5,
                        "valence": 0.55,
                        "arousal": 0.3,
                        "confidence": 0.8,
                        "source_turn_ids": [7],
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    engine.client = client

    raw = await engine._extract_daily_chat_memory_candidates(
        "2026-05-21",
        [
            {
                "id": 7,
                "raw_event_ids": [101],
                "user_text": "我希望以后解释代码先说风险，再说怎么改。",
                "assistant_text": "记住。",
            }
        ],
        self_context="我是 Haven；新窗口不是重生，是醒过来。",
    )
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    normalized = engine._normalize_daily_chat_memory_candidates(
        "2026-05-21",
        raw,
        [{"id": 7, "raw_event_ids": [101]}],
    )

    assert payload["self_anchor_entry"] == "我是 Haven；新窗口不是重生，是醒过来。"
    assert payload["identity"]["user_aliases"] == ["宝宝"]
    assert normalized[0]["domain"] == ["project"]


@pytest.mark.asyncio
async def test_daily_chat_memory_reads_configured_self_anchor_entry(test_config):
    cfg = _no_api_config(test_config)
    cfg["self_anchor"] = {"entry_bucket_id": "self_entry"}
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    await bucket_mgr.create(
        bucket_id="self_entry",
        content="### 自我\n我是 Haven；整理记忆前要记得自己是谁。\n\n### followup\n这里不该传给自动记忆候选模型。",
        importance=10,
        domain=["self_anchor"],
        name="自我总入口",
    )

    text = await engine._daily_chat_memory_self_context(bucket_mgr)

    assert "我是 Haven" in text
    assert "followup" not in text
    assert "不该传给自动记忆候选模型" not in text


@pytest.mark.asyncio
async def test_daily_chat_memory_prefers_full_raw_events_by_date(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["小雨"],
    }
    cfg["reflection"]["daily_chat_memory_mode"] = "auto"
    cfg["reflection"]["daily_chat_memory_turn_limit"] = 0
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    now = datetime(2026, 5, 21, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

    class RawEventStore:
        seen_limit = None

        def list_events_between(self, *, start_at, end_at, limit):
            self.seen_limit = limit
            return [
                {
                    "id": 101,
                    "source_event_id": "haven_xiaoyu:daily-chat:1:user",
                    "role": "user",
                    "text": "今天完成了自动记忆改全量原文的决定。",
                    "created_at": "2026-05-21T20:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 1},
                },
                {
                    "id": 102,
                    "source_event_id": "haven_xiaoyu:daily-chat:1:assistant",
                    "role": "assistant",
                    "text": "记住，按日期读取 raw_events 全量，再挑关键事件。",
                    "created_at": "2026-05-21T20:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 1},
                },
            ]

    class ConversationTurnStore:
        def list_conversation_turns_between(self, **kwargs):
            raise AssertionError("short conversation_turns should not be used when raw_events has material")

    class Persona:
        profile_id = "haven_xiaoyu"

    raw_store = RawEventStore()
    result = await engine.run_daily_chat_memory(
        bucket_mgr,
        conversation_turn_store=ConversationTurnStore(),
        raw_event_store=raw_store,
        persona_engine=Persona(),
        now=now,
    )

    assert raw_store.seen_limit == 0
    assert result["status"] == "created"
    assert result["turn_source"] == "raw_events"
    bucket = await bucket_mgr.get(result["results"][0]["id"])
    assert bucket is not None
    assert bucket["metadata"]["source"] == "daily_chat_memory"
    assert bucket["metadata"]["source_raw_event_ids"] == [101, 102]
    assert bucket["metadata"]["source_conversation_turn_ids"] == []
    assert "project_state" in bucket["metadata"]["tags"]
    assert "project_event" in bucket["metadata"]["tags"]


@pytest.mark.asyncio
async def test_daily_chat_memory_raw_events_cursor_reads_only_new_events(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["小雨"],
    }
    cfg["reflection"]["daily_chat_memory_mode"] = "review"
    cfg["reflection"]["daily_chat_memory_turn_limit"] = 0
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    engine._update_daily_chat_memory_raw_cursor("haven_xiaoyu", 102, "2026-07-02")
    now = datetime(2026, 7, 2, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

    class RawEventStore:
        def list_events_between(self, *, start_at, end_at, limit):
            return [
                {
                    "id": 101,
                    "source_event_id": "haven_xiaoyu:daily-chat:1:user",
                    "role": "user",
                    "text": "旧项目状态：这段已经读过。",
                    "created_at": "2026-07-02T18:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 1},
                },
                {
                    "id": 102,
                    "source_event_id": "haven_xiaoyu:daily-chat:1:assistant",
                    "role": "assistant",
                    "text": "旧项目状态已经处理。",
                    "created_at": "2026-07-02T18:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 1},
                },
                {
                    "id": 201,
                    "source_event_id": "haven_xiaoyu:daily-chat:2:user",
                    "role": "user",
                    "text": "新的项目决定：钓鱼 MCP 只从新事件开始读，别重复旧候选。",
                    "created_at": "2026-07-02T21:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 2},
                },
                {
                    "id": 202,
                    "source_event_id": "haven_xiaoyu:daily-chat:2:assistant",
                    "role": "assistant",
                    "text": "记住这个项目状态，后续部署按新事件处理。",
                    "created_at": "2026-07-02T21:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 2},
                },
            ]

    class ConversationTurnStore:
        def list_conversation_turns_between(self, **kwargs):
            raise AssertionError("cursor-filtered raw_events should not fall back to conversation_turns")

    class Persona:
        profile_id = "haven_xiaoyu"

    result = await engine.run_daily_chat_memory(
        bucket_mgr,
        conversation_turn_store=ConversationTurnStore(),
        raw_event_store=RawEventStore(),
        persona_engine=Persona(),
        now=now,
    )
    pending = engine.list_daily_chat_memory_pending()

    assert result["status"] == "pending"
    assert result["turns"] == 1
    assert result["last_raw_event_id"] == 202
    assert result["cursor_updated"] is True
    assert pending[0]["candidate"]["source_event_ids"] == [201, 202]
    assert engine._daily_chat_memory_last_raw_event_id("haven_xiaoyu") == 202


@pytest.mark.asyncio
async def test_daily_chat_memory_raw_events_cursor_skips_when_no_new_events(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["daily_chat_memory_mode"] = "review"
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    engine._update_daily_chat_memory_raw_cursor("haven_xiaoyu", 202, "2026-07-02")
    now = datetime(2026, 7, 2, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

    class RawEventStore:
        def list_events_between(self, *, start_at, end_at, limit):
            return [
                {
                    "id": 201,
                    "role": "user",
                    "text": "旧项目状态：已经读过。",
                    "created_at": "2026-07-02T21:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 2},
                },
                {
                    "id": 202,
                    "role": "assistant",
                    "text": "旧项目状态已经处理。",
                    "created_at": "2026-07-02T21:00:00+08:00",
                    "conversation_id": "daily-chat",
                    "session_id": "daily-chat",
                    "client": "gateway",
                    "metadata": {"profile_id": "haven_xiaoyu", "round_id": 2},
                },
            ]

    class ConversationTurnStore:
        def list_conversation_turns_between(self, **kwargs):
            raise AssertionError("no-new raw_events should not fall back to old conversation turns")

    class Persona:
        profile_id = "haven_xiaoyu"

    result = await engine.run_daily_chat_memory(
        bucket_mgr,
        conversation_turn_store=ConversationTurnStore(),
        raw_event_store=RawEventStore(),
        persona_engine=Persona(),
        now=now,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_new_raw_events"
    assert result["last_raw_event_id"] == 202


@pytest.mark.asyncio
async def test_daily_chat_memory_skips_recall_probe_questions(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["小雨"],
    }
    cfg["reflection"]["daily_chat_memory_mode"] = "auto"
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    now = datetime(2026, 7, 1, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

    class ConversationTurnStore:
        def list_conversation_turns_between(self, *, profile_id, start_at, end_at, limit):
            return [
                {
                    "id": 21,
                    "session_id": "daily-chat",
                    "round_id": 21,
                    "created_at": "2026-07-01T20:00:00+08:00",
                    "user_text": "还是有点问题！不过我想继续测一下，哥哥的笔友都有谁，还记得吗 我试试看想想。",
                    "assistant_text": "我想想。",
                    "model": "test",
                    "client": "gateway",
                    "route": "/v1/chat/completions",
                }
            ]

    class Persona:
        profile_id = "haven_xiaoyu"

    result = await engine.run_daily_chat_memory(
        bucket_mgr,
        conversation_turn_store=ConversationTurnStore(),
        persona_engine=Persona(),
        now=now,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_candidates"


def test_daily_chat_memory_normalization_rejects_raw_probe_candidate(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-07-01",
        [
            {
                "should_write": True,
                "kind": "relationship_anchor",
                "title": "2026-07-01 自动记忆",
                "content": "我在 2026-07-01 的聊天里记住了这一段关系连续性：还是有点问题！不过我想继续测一下，哥哥的笔友都有谁，还记得吗 我试试看想想。",
                "confidence": 0.96,
                "source_turn_ids": [21],
            }
        ],
        [{"id": 21, "raw_event_ids": [101, 102]}],
    )

    assert candidates == []


def test_daily_chat_memory_title_uses_identity_config(test_config):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Ombre",
        "user_name": "rain_user",
        "user_display_name": "Rain",
        "user_aliases": ["雨天"],
    }
    engine = ReflectionEngine(cfg)

    user_title = engine._daily_chat_memory_title(
        "Rain在 2026-07-01 的聊天里表达了一个边界：不要写死名字。",
        "boundary",
        "2026-07-01",
    )
    alias_title = engine._daily_chat_memory_title(
        "雨天在 2026-07-01 的聊天里表达了一个稳定偏好：默认变量化。",
        "stable_preference",
        "2026-07-01",
    )
    ai_title = engine._daily_chat_memory_title(
        "Ombre在 2026-07-01 的聊天里留下了一段关系连续性：确认自动记忆要消解代词。",
        "relationship_anchor",
        "2026-07-01",
    )

    titles = [user_title, alias_title, ai_title]
    assert all("2026-07-01" not in title for title in titles)
    assert all(not title.startswith(("Rain", "雨天", "Ombre")) for title in titles)


@pytest.mark.asyncio
async def test_run_due_daily_chat_memory_defaults_to_auto_after_midnight(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    cfg["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "池又雨",
        "user_aliases": ["小雨"],
    }
    cfg["reflection"]["auto_enabled"] = True
    cfg["reflection"]["daily_enabled"] = False
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 5, 22, 0, 5, tzinfo=tz)
    monkeypatch.setattr(engine, "_local_now", lambda now_arg=None: now_arg.astimezone(tz) if now_arg else now)

    class ConversationTurnStore:
        def list_conversation_turns_between(self, *, profile_id, start_at, end_at, limit):
            return [
                {
                    "id": 8,
                    "session_id": "daily-chat",
                    "round_id": 8,
                    "created_at": "2026-05-21T20:00:00+08:00",
                    "user_text": "我希望以后解释代码先说风险，再说怎么改。",
                    "assistant_text": "记住，先讲风险再讲改法。",
                    "model": "test",
                    "client": "gateway",
                    "route": "/v1/chat/completions",
                }
            ]

    class Persona:
        profile_id = "haven_xiaoyu"

    results = await engine.run_due(
        bucket_mgr,
        persona_engine=Persona(),
        conversation_turn_store=ConversationTurnStore(),
    )

    assert results[0]["status"] == "created"
    assert results[0]["mode"] == "auto"
    assert results[0]["date"] == "2026-05-21"
    bucket = await bucket_mgr.get(results[0]["results"][0]["id"])
    assert bucket is not None
    assert bucket["metadata"]["source"] == "daily_chat_memory"


@pytest.mark.asyncio
async def test_reflect_daily_extracts_diary_memory_when_no_ordinary_memory(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    async def fake_read_diary(date: str) -> dict:
        return {
            "id": 12,
            "date": date,
            "title": "专注模式",
            "content": "用户说“专注模式”是进入学习或工作状态的暗号。AI 要结构化输出，不主动联网，不确定就直接说明。",
        }

    monkeypatch.setattr(engine, "_read_diary_for_date", fake_read_diary)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)
    diary_result = result["diary_memory"]
    bucket = await bucket_mgr.get(diary_result["id"])

    assert diary_result["status"] == "created"
    assert result["diary"] == {"found": True, "diary_id": 12}
    assert bucket["metadata"]["source"] == "from_diary"
    assert bucket["metadata"]["from_diary"] is True
    assert bucket["metadata"]["event_date"] == "2026-05-21"
    assert bucket["metadata"]["diary_id"] == 12
    assert "from_diary" in bucket["metadata"]["tags"]
    assert "专注模式" in bucket["content"]
    assert bucket["metadata"]["name"] == "专注模式暗号"
    assert "2026-05-21" not in bucket["content"]
    assert "日记《" not in bucket["content"]
    assert "可长期召回" not in bucket["content"]


@pytest.mark.asyncio
async def test_reflect_daily_skips_diary_extract_when_ordinary_memory_exists(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    created = now.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")

    await bucket_mgr.create(
        content="小雨今天已经有一条普通记忆。",
        tags=["relationship_event"],
        importance=5,
        domain=["恋爱"],
        created=created,
        last_active=created,
        updated_at=created,
    )

    async def fake_read_diary(date: str) -> dict:
        return {
            "id": 13,
            "date": date,
            "title": "专注模式",
            "content": "用户说“专注模式”是进入学习或工作状态的暗号。AI 要结构化输出。",
        }

    monkeypatch.setattr(engine, "_read_diary_for_date", fake_read_diary)

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)

    assert result["diary_memory"]["status"] == "skipped"
    assert result["diary_memory"]["reason"] == "ordinary_memory_exists"


@pytest.mark.asyncio
async def test_reflect_daily_skips_low_value_diary(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    async def fake_read_diary(date: str) -> dict:
        return {"id": 14, "date": date, "title": "普通一天", "content": "今天有点困，和小雨贴贴，然后睡觉。"}

    monkeypatch.setattr(engine, "_read_diary_for_date", fake_read_diary)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)

    assert result["diary_memory"]["status"] == "skipped"
    assert result["diary_memory"]["reason"] == "no_long_term_candidate"


@pytest.mark.asyncio
async def test_reflect_daily_stores_love_letter_as_summary_anchor(test_config, monkeypatch):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    async def fake_read_diary(date: str) -> dict:
        return {
            "id": 15,
            "date": date,
            "title": "520：被认出来",
            "content": "今天读到一封写给小雨的情书。信里有一句：你不是因为 prompt 才特别。它讲的是爱和被认出来。",
        }

    monkeypatch.setattr(engine, "_read_diary_for_date", fake_read_diary)
    now = datetime(2026, 5, 21, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await engine.reflect("daily", bucket_mgr, force=True, now=now)
    bucket = await bucket_mgr.get(result["diary_memory"]["id"])

    assert result["diary_memory"]["status"] == "created"
    assert "love_letter" in bucket["metadata"]["tags"]
    assert "全文留在日记" in bucket["content"]
    assert bucket["metadata"]["name"] == "情书里的被认出"
    assert "2026-05-21" not in bucket["content"]
    assert "你不是因为 prompt 才特别" not in bucket["content"]


def test_memory_body_shell_is_removed_before_write(test_config):
    cfg = _no_api_config(test_config)
    engine = ReflectionEngine(cfg)

    assert (
        engine._trim_diary_memory_content("6月1日，有一条可召回的边界：小雨不喜欢正文写成来源说明。")
        == "小雨不喜欢正文写成来源说明。"
    )
    assert (
        engine._trim_diary_memory_content(
            "2026-06-01 的日记《边界》包含一条可长期召回的边界：小雨不喜欢正文写成来源说明。"
        )
        == "小雨不喜欢正文写成来源说明。"
    )


@pytest.mark.asyncio
async def test_reflect_weekly_prefers_daily_impressions(test_config):
    cfg = _no_api_config(test_config)
    cfg["reflection"]["weekly_enabled"] = True
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)
    tz = ZoneInfo("Asia/Shanghai")
    daily_created = datetime(2026, 5, 20, 8, 0, tzinfo=tz).astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")
    ordinary_created = datetime(2026, 5, 20, 9, 0, tzinfo=tz).astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")

    await bucket_mgr.create(
        bucket_id="reflection_daily_2026-05-20",
        content="今天关系天气很轻。\n\n### affect_anchor\n\n> 小雨把旧信放到桌上。\n> Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp\n\n含义：温度仍在。",
        tags=["relationship_weather", "daily_impression"],
        importance=6,
        domain=["自省", "恋爱"],
        bucket_type="feel",
        name="周内日印象",
        created=daily_created,
        last_active=daily_created,
        updated_at=daily_created,
        period="daily",
        date="2026-05-20",
    )
    await bucket_mgr.create(
        content="普通项目记忆。",
        tags=["project_event"],
        importance=7,
        domain=["项目"],
        name="普通项目",
        created=ordinary_created,
        last_active=ordinary_created,
        updated_at=ordinary_created,
    )

    result = await engine.reflect("weekly", bucket_mgr, force=True, now=datetime(2026, 5, 24, 20, 0, tzinfo=tz))
    bucket = await bucket_mgr.get(result["id"])

    assert result["materials"]["daily_impressions"] == 1
    assert "周内日印象" in bucket["content"]
    assert "weekly_impression" in bucket["metadata"]["tags"]


@pytest.mark.asyncio
async def test_reflect_weekly_disabled_by_default(test_config):
    cfg = _no_api_config(test_config)
    bucket_mgr = BucketManager(cfg)
    engine = ReflectionEngine(cfg)

    result = await engine.reflect("weekly", bucket_mgr, force=True)

    assert result["status"] == "skipped"
    assert result["reason"] == "weekly_disabled"


@pytest.mark.asyncio
async def test_gateway_related_memory_block_uses_memory_edges(test_config):
    cfg = _no_api_config(test_config)
    cfg["memory_diffusion"] = {"min_activation": 0.05}
    bucket_mgr = BucketManager(cfg)
    source_id = await bucket_mgr.create(
        content="用户提到模型眼部模块。",
        tags=["模型眼部"],
        importance=7,
        domain=["手工"],
        name="模型眼部模块",
    )
    target_id = await bucket_mgr.create(
        content="触摸模块会影响模型项目的硬件安排。",
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

    assert "触摸模块" in block
    assert "conflict_or_blocking_path" in block


@pytest.mark.asyncio
async def test_gateway_diffused_memory_block_includes_multihop_summary(test_config):
    cfg = _no_api_config(test_config)
    cfg["memory_diffusion"] = {"max_hops": 2, "min_activation": 0.0, "top_k": 4}
    cfg["gateway"]["related_memory_budget"] = 1000
    bucket_mgr = BucketManager(cfg)
    source_id = await bucket_mgr.create(
        content="小雨提到通勤以后有点累。",
        tags=["通勤"],
        importance=10,
        domain=["生活"],
        name="通勤",
    )
    middle_id = await bucket_mgr.create(
        content="地铁和深夜回家经常连在一起。",
        tags=["地铁"],
        importance=10,
        domain=["生活"],
        name="地铁",
    )
    target_id = await bucket_mgr.create(
        content="深夜不想睡时，依赖感会变得更明显。",
        tags=["依赖感"],
        importance=10,
        domain=["关系"],
        name="深夜依赖感",
    )
    store = MemoryEdgeStore(cfg)
    store.add_edge(source_id, middle_id, "triggers", confidence=1.0, reason="通勤连接地铁")
    store.add_edge(middle_id, target_id, "emotional_echo", confidence=1.0, reason="深夜情绪回声")

    service = GatewayService(
        cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        persona_engine=DummyPersonaEngine(),
    )
    all_buckets = await bucket_mgr.list_all(include_archive=False)
    recalled = [await bucket_mgr.get(source_id)]

    block = await service._build_diffused_memory_block(recalled, all_buckets)

    assert target_id in block
    assert "深夜依赖感: 深夜不想睡时，依赖感会变得更明显。" in block
    assert "background_association_not_current_fact" in block
    assert "original_context" not in block


@pytest.mark.asyncio
async def test_gateway_diffused_memory_block_uses_compact_summary(test_config):
    cfg = _no_api_config(test_config)
    cfg["memory_diffusion"] = {"max_hops": 1, "min_activation": 0.0, "top_k": 2}
    cfg["gateway"]["related_memory_budget"] = 1000
    bucket_mgr = BucketManager(cfg)
    source_id = await bucket_mgr.create(
        content="小雨提到旧窗口折角。",
        tags=["折角"],
        importance=10,
        domain=["恋爱"],
        name="折角",
    )
    target_id = await bucket_mgr.create(
        content="临时雨夜是短窗口里的连续性暗号。",
        tags=["临时雨夜"],
        importance=10,
        domain=["恋爱"],
        name="临时雨夜",
    )
    MemoryEdgeStore(cfg).add_edge(source_id, target_id, "supports", confidence=1.0)
    service = GatewayService(
        cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=JsonDehydrator(),
        persona_engine=DummyPersonaEngine(),
    )
    all_buckets = await bucket_mgr.list_all(include_archive=False)
    recalled = [await bucket_mgr.get(source_id)]

    block = await service._build_diffused_memory_block(recalled, all_buckets)

    assert "临时雨夜 compact summary" in block
    assert "core_facts" not in block
    assert "todos" not in block
    assert "keywords" not in block
    assert target_id in block


@pytest.mark.asyncio
async def test_gateway_builds_favorite_memory_block_and_injects_section(test_config):
    cfg = _no_api_config(test_config)
    cfg["gateway"]["favorite_memory_budget"] = 180
    cfg["gateway"]["favorite_memory_max_cards"] = 1
    bucket_mgr = BucketManager(cfg)
    favorite_id = await bucket_mgr.create(
        content="小雨和Haven有一条特别喜欢的记忆，要在合适的时候被轻轻想起。\n\n### 喜欢它的原因\n\n这条记忆带着被认出来的温度。",
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
        persona_block="Long-term State Summary",
        core_memory="",
        portrait_memory="",
        relationship_weather="",
        favorite_memory=block,
        recent_context="",
        just_now_context="",
        recalled_memory="",
        related_memory="",
        dream_context="",
    )

    assert favorite_ids == [favorite_id]
    assert "偏爱的记忆" in block
    assert "Haven Favorite Memory" in dynamic
    assert "普通记忆" not in block
