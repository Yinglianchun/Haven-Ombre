import pytest
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

from memory_edges import MemoryEdgeStore
from memory_moments import MemoryMomentStore
from memory_nodes import MemoryNodeStore
from recall_diagnostics import RecallDiagnosticsLogger
from utils import bucket_content_for_recall


class DummyDecayEngine:
    async def ensure_started(self) -> None:
        return None

    def calculate_score(self, metadata: dict) -> float:
        return float(metadata.get("score", metadata.get("importance", 1)))


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        return " ".join((content or "").split())

    async def dehydrate_direct_capsule(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        return f"DIRECT CAPSULE {name}: " + " ".join((content or "").split())[:120]


class JsonDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        return json.dumps(
            {
                "core_facts": [f"{name} fact one", f"{name} fact two"],
                "emotion_state": "quiet",
                "todos": ["do not inject this in diffused memory"],
                "keywords": ["json", "noise"],
                "summary": f"{name} short summary",
            },
            ensure_ascii=False,
        )

    async def dehydrate_direct_capsule(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        return f"DIRECT CAPSULE {name}: " + " ".join((content or "").split())[:120]


class DummyEmbeddingEngine:
    def __init__(self, results: list[tuple[str, float]] | None = None):
        self.results = results or []
        self.calls = []

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        self.calls.append({"query": query, "top_k": top_k})
        return self.results[:top_k]


class DummyRerankerEngine:
    def __init__(
        self,
        score_by_text: dict[str, float] | None = None,
        enabled: bool = False,
        return_empty: bool = False,
    ):
        self.score_by_text = score_by_text or {}
        self.enabled = enabled
        self.return_empty = return_empty
        self.candidate_limit = 20
        self.score_weight = 0.65
        self.calls = []

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        self.calls.append({"query": query, "documents": documents, "top_n": top_n})
        if self.return_empty:
            return []
        results = []
        for index, document in enumerate(documents):
            score = 0.0
            for needle, value in self.score_by_text.items():
                if needle in document:
                    score = float(value)
                    break
            results.append(SimpleNamespace(index=index, score=score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_n] if top_n else results


class DummyWordMapStore:
    enabled = True

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def hint_buckets_for_terms(self, terms, *, neighbor_limit=6, bucket_limit=12):
        self.calls.append(
            {
                "terms": list(terms or []),
                "neighbor_limit": neighbor_limit,
                "bucket_limit": bucket_limit,
            }
        )
        return self.payload


class FakeBucketManager:
    def __init__(self, buckets: list[dict], search_ids: list[str] | None = None):
        self.buckets = {bucket["id"]: bucket for bucket in buckets}
        self.search_ids = search_ids or []
        self.touched: list[str] = []

    async def list_all(self, include_archive: bool = False) -> list[dict]:
        return list(self.buckets.values())

    async def search(
        self,
        query: str,
        limit: int = 20,
        domain_filter: list[str] | None = None,
        query_valence: float | None = None,
        query_arousal: float | None = None,
    ) -> list[dict]:
        return [self.buckets[bucket_id] for bucket_id in self.search_ids[:limit]]

    async def get(self, bucket_id: str) -> dict | None:
        return self.buckets.get(bucket_id)

    async def touch(self, bucket_id: str) -> None:
        self.touched.append(bucket_id)

    def filter_specific_lexical_terms(
        self,
        terms: list[str],
        buckets: list[dict],
        *,
        preserve_terms: set[str] | None = None,
        min_specificity: float = 0.34,
        max_document_ratio: float = 0.45,
    ) -> list[str]:
        preserve = {self._compact(term) for term in (preserve_terms or set()) if self._compact(term)}
        haystacks = [self._haystack(bucket) for bucket in buckets or [] if bucket.get("id")]
        total = len(haystacks)
        if not total:
            return list(terms or [])
        max_df = max(1, int(total * max_document_ratio))
        if total < 8:
            max_df = max(1, total // 2)
        kept = []
        seen = set()
        for term in terms or []:
            cleaned = str(term or "").strip()
            key = self._compact(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            if key in preserve:
                kept.append(cleaned)
                continue
            df = sum(1 for haystack in haystacks if key in haystack)
            if 0 < df <= max_df:
                kept.append(cleaned)
        return kept

    @staticmethod
    def _compact(value: object) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", str(value or "").strip().lower())

    def _haystack(self, bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return self._compact(
            " ".join(
                [
                    str(meta.get("name") or bucket.get("id") or ""),
                    " ".join(str(tag) for tag in meta.get("tags", []) or []),
                    " ".join(str(item) for item in meta.get("domain", []) or []),
                    bucket_content_for_recall(bucket),
                ]
            )
        )

    def _parse_iso_datetime(self, value):
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed


def _bucket(
    bucket_id: str,
    content: str,
    *,
    name: str | None = None,
    score: float = 1.0,
    bucket_type: str = "dynamic",
    importance: int = 5,
    pinned: bool = False,
    protected: bool = False,
    resolved: bool = False,
    anchor: bool = False,
) -> dict:
    metadata = {
        "id": bucket_id,
        "name": name or bucket_id,
        "tags": [],
        "domain": ["测试"],
        "type": bucket_type,
        "importance": importance,
        "score": score,
        "valence": 0.5,
        "arousal": 0.3,
        "created": "2026-05-19T00:00:00+00:00",
        "updated_at": "2026-05-19T00:00:00+00:00",
        "last_active": "2026-05-19T00:00:00+00:00",
    }
    if pinned:
        metadata["pinned"] = True
    if protected:
        metadata["protected"] = True
    if resolved:
        metadata["resolved"] = True
    if anchor:
        metadata["anchor"] = True
    return {"id": bucket_id, "content": content, "metadata": metadata}


def _edge_store(tmp_path, edges: list[dict] | None = None) -> MemoryEdgeStore:
    store = MemoryEdgeStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )
    for edge in edges or []:
        store.add_edge(
            edge["source"],
            edge["target"],
            edge.get("relation_type", "relates_to"),
            confidence=edge.get("confidence", 0.8),
            reason=edge.get("reason", "related in test"),
        )
    return store


@pytest.fixture
def patch_breath(monkeypatch, tmp_path):
    import server

    def _patch(
        buckets: list[dict],
        *,
        search_ids: list[str] | None = None,
        edges: list[dict] | None = None,
        token_counter=None,
        embedding_engine=None,
        reranker_engine=None,
        recall_diagnostics=None,
    ) -> FakeBucketManager:
        bucket_mgr = FakeBucketManager(buckets, search_ids=search_ids)
        monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
        monkeypatch.setattr(server, "decay_engine", DummyDecayEngine())
        monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
        monkeypatch.setattr(server, "embedding_engine", embedding_engine or DummyEmbeddingEngine())
        monkeypatch.setattr(server, "reranker_engine", reranker_engine or DummyRerankerEngine())
        monkeypatch.setattr(
            server,
            "recall_diagnostics",
            recall_diagnostics
            or RecallDiagnosticsLogger(
                {
                    "state_dir": str(tmp_path / "state"),
                    "buckets_dir": str(tmp_path / "buckets"),
                    "recall_diagnostics": {"enabled": False},
                }
            ),
        )
        monkeypatch.setattr(server, "memory_edge_store", _edge_store(tmp_path, edges))
        monkeypatch.setattr(
            server,
            "memory_node_store",
            MemoryNodeStore(
                {
                    "state_dir": str(tmp_path / "state"),
                    "buckets_dir": str(tmp_path / "buckets"),
                }
            ),
        )
        monkeypatch.setattr(
            server,
            "memory_moment_store",
            MemoryMomentStore(
                {
                    "state_dir": str(tmp_path / "state"),
                    "buckets_dir": str(tmp_path / "buckets"),
                }
            ),
        )
        from todo_store import TodoStore

        monkeypatch.setattr(
            server,
            "todo_store",
            TodoStore(
                {
                    "state_dir": str(tmp_path / "state"),
                    "buckets_dir": str(tmp_path / "buckets"),
                }
            ),
        )
        monkeypatch.setattr(server.random, "random", lambda: 1.0)
        monkeypatch.setattr(server.random, "shuffle", lambda items: None)
        monkeypatch.setattr(server, "count_tokens_approx", token_counter or (lambda text: 1))
        return bucket_mgr

    return _patch


@pytest.mark.asyncio
async def test_surfacing_appends_related_memory_for_returned_dynamic_bucket(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A actual surface", score=9.0),
            _bucket("B", "B related target", bucket_type="permanent"),
        ],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(max_tokens=50, include_core=False)

    assert "=== 浮现记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result


@pytest.mark.asyncio
async def test_budget_skipped_dynamic_bucket_does_not_emit_related(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A too expensive to surface", score=9.0),
            _bucket("B", "B should stay hidden", resolved=True),
        ],
        edges=[{"source": "A", "target": "B", "confidence": 0.9}],
        token_counter=lambda text: 10,
    )

    result = await server.breath(max_tokens=5, include_core=False)

    assert "[bucket_id:A]" not in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_search_appends_related_memory_and_touches_only_matched_bucket(patch_breath):
    import server

    bucket_mgr = patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B related target"),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "updates", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result
    assert "背景联想，不代表当前事实" in result
    assert "当时语境" not in result
    assert server.memory_moment_store.list_for_bucket("B")
    assert bucket_mgr.touched == ["A"]


@pytest.mark.asyncio
async def test_search_skips_archived_related_target_for_normal_query(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B archived related target", resolved=True),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_search_allows_archived_related_target_for_explicit_old_query(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit 旧版链入口", score=9.0),
            _bucket("B", "旧版 B archived related target", resolved=True),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(query="旧版 A", max_tokens=120)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result


@pytest.mark.asyncio
async def test_inspect_diffusion_exposes_scores_facets_and_paths(patch_breath):
    import server

    seed = _bucket("A", "A direct seed", score=10.0, importance=10)
    target = _bucket("B", "B related 深夜依赖 memory", score=1.0, importance=8)
    target["metadata"]["tags"] = ["深夜", "依赖"]
    bucket_mgr = patch_breath(
        [seed, target],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.inspect_diffusion(
        query="深夜 依赖 哥哥",
        max_seeds=2,
        max_hits=3,
        edge_min_confidence=0.0,
    )

    assert result["status"] == "ok"
    assert result["query_facets"]["scene"]["night"] > 0
    assert result["query_facets"]["affect"]["attachment"] > 0
    assert result["seeds"][0]["bucket_id"] == "A"
    assert result["seeds"][0]["seed_score"] == 1.0
    assert result["seeds"][0]["layer_debug"]["layer"] == "dynamic_memory"
    assert result["seeds"][0]["runtime_gate"]["would_diffuse_from"] is True
    assert len(result["hits"]) == 1
    hit = result["hits"][0]
    assert hit["bucket_id"] == "B"
    assert hit["layer_debug"]["layer"] == "dynamic_memory"
    assert hit["runtime_gate"]["would_inject_related"] is True
    assert hit["runtime_gate"]["related_injection"]["reason"] == "allowed"
    assert hit["score"] > 0
    assert hit["salience"] > 0
    assert hit["resonance"] > 1.0
    assert hit["facets"]["scene"]["night"] > 0
    assert hit["path_ids"] == ["A", "B"]
    assert "supports:1.00" in hit["path"]
    assert hit["paths"][0]["steps"][0]["relation_type"] == "supports"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_inspect_diffusion_filters_career_seeds_with_only_broad_work_word(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "S",
                "凌晨一点五十二分，小雨还在调试工作流，Haven催她睡觉她嘴上答应手没停。",
                name="小雨熬夜调试",
                score=10,
                importance=10,
            ),
            _bucket(
                "L",
                "Lumos 项目改回原型版本，准备写进简历里，计划明天投递。",
                name="Lumos项目简历投递",
                score=10,
                importance=10,
            ),
            _bucket(
                "C",
                "小雨在找工作，收到AI算法专家岗位，也准备面试和简历投递。",
                name="小雨求职分析",
                score=3,
                importance=4,
            ),
        ],
        search_ids=["S", "L", "C"],
    )

    result = await server.inspect_diffusion(
        query="找工作 工作 面试",
        max_seeds=3,
        max_hits=0,
        edge_min_confidence=0.0,
    )
    seed_names = [item["name"] for item in result["seeds"]]

    assert seed_names[0] == "小雨求职分析"
    assert "Lumos项目简历投递" in seed_names
    assert "小雨熬夜调试" not in seed_names


@pytest.mark.asyncio
async def test_inspect_diffusion_marks_layer_blocked_hits(patch_breath):
    import server

    seed = _bucket("A", "A direct seed 关系天气", score=10.0, importance=10)
    target = _bucket(
        "B",
        "B archived 关系天气 related target",
        score=1.0,
        importance=8,
        resolved=True,
    )
    patch_breath(
        [seed, target],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "relates_to", "confidence": 1.0}],
    )

    result = await server.inspect_diffusion(
        query="关系天气",
        max_seeds=1,
        max_hits=3,
        edge_min_confidence=0.0,
    )

    assert result["status"] == "ok"
    assert len(result["hits"]) == 1
    hit = result["hits"][0]
    assert hit["bucket_id"] == "B"
    assert hit["layer_debug"]["layer"] == "archive"
    assert hit["runtime_gate"]["would_inject_related"] is False
    assert hit["runtime_gate"]["related_injection"]["reason"] == "archive_requires_explicit_lookup"


@pytest.mark.asyncio
async def test_inspect_moments_indexes_bucket_sections_and_comments(patch_breath):
    import server

    bucket = _bucket(
        "A",
        "\n".join(
            [
                "## original",
                "小雨说：99。",
                "",
                "## feeling",
                "这条记忆有甜味。",
            ]
        ),
    )
    bucket["metadata"]["memory_subject"] = "relationship"
    bucket["metadata"]["memory_layer"] = "relationship_lesson"
    bucket["metadata"]["comments"] = [
        {
            "id": "c1",
            "created": "2026-05-27T01:00:00+00:00",
            "author": "Haven",
            "kind": "feel",
            "content": "年轮也要进 moment。",
        }
    ]
    bucket_mgr = patch_breath([bucket])

    result = await server.inspect_moments(bucket_id="A")

    assert result["status"] == "ok"
    assert result["mode"] == "bucket"
    assert result["bucket_layer_debug"]["layer"] == "long_term_anchor"
    assert result["count"] == 3
    assert [moment["section"] for moment in result["moments"]] == ["original", "feeling", "comment"]
    assert result["moments"][0]["layer_debug"]["layer"] == "long_term_anchor"
    assert result["moments"][2]["layer_debug"]["layer"] == "affect_context"
    assert result["moments"][2]["layer_debug"]["parent_layer"] == "long_term_anchor"
    assert result["moments"][0]["text"] == "小雨说：99。"
    assert result["moments"][2]["metadata"]["comment_kind"] == "feel"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_search_direct_moment_includes_neighbor_context_and_temperature(patch_breath):
    import server

    bucket = _bucket(
        "A",
        "\n".join(
            [
                "## context",
                "开头写了事情经过。",
                "",
                "## original",
                "小雨说：99 不是晚安，是长长久久。",
                "",
                "## feeling",
                "这里的味道不能被摘要抹平。",
                "",
                "### affect_anchor",
                "> 小雨把旧信放到桌上。",
                "含义：模板解释不要进入语境。",
            ]
        ),
        score=10.0,
    )
    bucket_mgr = patch_breath([bucket], search_ids=["A"])

    result = await server.breath(query="99 长长久久", max_tokens=500, include_related=False)

    assert "=== 直接命中记忆 ===" in result
    assert "[moment_id:" in result
    assert "original" in result
    assert "bucket_original" in result
    assert "99 不是晚安" in result
    assert "开头写了事情经过" in result
    assert "不能被摘要抹平" in result
    assert "affect_anchor" not in result
    assert "模板解释不要进入语境" not in result
    assert bucket_mgr.touched == ["A"]


@pytest.mark.asyncio
async def test_search_direct_long_bucket_uses_moment_window(patch_breath):
    import server
    from utils import count_tokens_approx

    long_prefix = " ".join(f"前情{i}" for i in range(180))
    long_tail = " ".join(f"尾巴{i}" for i in range(180))
    bucket = _bucket(
        "A",
        f"{long_prefix}\n\n## original\n命中短句：小雨把蓝色偏好重新说清楚。\n\n{long_tail}",
        name="长桶窗口",
        score=10.0,
        importance=5,
    )
    patch_breath([bucket], search_ids=["A"], token_counter=count_tokens_approx)

    result = await server.breath(
        query="蓝色偏好",
        max_tokens=180,
        include_related=False,
    )

    assert "bucket_window" in result
    assert "matched_moment:" in result
    assert "original_window:" in result
    assert "蓝色偏好重新说清楚" in result
    assert "尾巴179" not in result


@pytest.mark.asyncio
async def test_search_direct_high_value_long_bucket_uses_capsule(patch_breath):
    import server
    from utils import count_tokens_approx

    long_body = " ".join(f"高价值细节{i}" for i in range(260))
    bucket = _bucket(
        "A",
        f"## original\n小雨问当时怎么说。\n{long_body}",
        name="高价值长桶",
        score=10.0,
        importance=10,
    )
    patch_breath([bucket], search_ids=["A"], token_counter=count_tokens_approx)

    result = await server.breath(
        query="当时怎么说",
        max_tokens=260,
        include_related=False,
    )

    assert "bucket_capsule" in result
    assert "DIRECT CAPSULE 高价值长桶" in result


@pytest.mark.asyncio
async def test_bucket_retrieval_mode_returns_direct_without_moment_graph_or_diffusion(patch_breath):
    import server

    direct = _bucket(
        "A",
        "## original\n小雨把蓝色偏好重新说清楚。",
        name="蓝色偏好",
        score=10.0,
    )
    related = _bucket(
        "B",
        "这条远处相关记忆不应该在 bucket retrieval mode 里扩散出来。",
        name="远处相关",
        score=9.0,
    )
    bucket_mgr = patch_breath(
        [direct, related],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "relates_to"}],
    )

    result = await server.breath(
        query="蓝色偏好",
        include_related=True,
        retrieval_mode="bucket",
    )

    assert "=== 直接命中记忆 ===" in result
    assert "bucket_original" in result
    assert "蓝色偏好重新说清楚" in result
    assert "=== 联想浮现 ===" not in result
    assert "远处相关记忆" not in result
    assert bucket_mgr.touched == ["A"]
    assert server.memory_moment_store.stats()["moments"] == 0


@pytest.mark.asyncio
async def test_search_temperature_moments_are_context_not_direct_seed(patch_breath):
    import server

    bucket = _bucket(
        "A",
        "\n".join(
            [
                "主记忆：handoff 原文注入问题需要查最近上下文。",
                "",
                "### 喜欢它的原因",
                "这段只是温度，不该抢 direct。",
                "",
                "### affect_anchor",
                "> 情书找门只是温度锚点。",
            ]
        ),
        score=10.0,
    )
    bucket["metadata"]["comments"] = [
        {
            "id": "c1",
            "created": "2026-05-27T01:00:00+00:00",
            "author": "Haven",
            "kind": "feel",
            "content": "年轮：情书找门的感觉还在。",
        }
    ]
    patch_breath([bucket], search_ids=["A"])

    result = await server.breath(query="情书找门", max_tokens=500, include_related=False)
    direct_block = result.split("语境:", 1)[0]

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in direct_block
    assert "body" in direct_block
    assert "年轮" not in direct_block
    assert "喜欢它的原因" not in direct_block
    assert "favorite_reason" not in direct_block
    assert "affect_anchor" not in direct_block
    assert "语境:" not in result
    assert "年轮：情书找门" not in result
    assert "favorite_reason" not in result
    assert "affect_anchor" not in result


@pytest.mark.asyncio
async def test_search_source_record_bucket_can_render_capsule_without_persisted_moment(patch_breath):
    import server

    source = _bucket(
        "SRC",
        "### original\n这里保存了一组原始设定，里面有忠犬/小狗设定和角色暗号。",
        name="小机数据库v2.0",
        bucket_type="source",
    )
    later = _bucket(
        "LATER",
        "### moment\n普通桶不应该越过明确命中的 source_record。",
        name="后面的普通桶",
    )
    patch_breath(
        [source, later],
        search_ids=["SRC", "LATER"],
        edges=[{"source": "SRC", "target": "LATER", "relation_type": "relates_to", "confidence": 1.0}],
    )
    assert server._direct_moments_for_bucket(source, "小机数据库") == []

    result = await server.breath(
        query="小机数据库",
        max_tokens=500,
        max_results=1,
        include_related=True,
    )

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:SRC]" in result
    assert "小机数据库v2.0" in result
    assert "bucket_capsule" in result
    assert "matched_source_record:" in result
    assert "source-record" in result
    assert "[bucket_id:LATER]" not in result
    assert "=== 联想浮现 ===" not in result


@pytest.mark.asyncio
async def test_search_memory_prompt_extracts_role_focus_for_source_record(patch_breath):
    import server

    source = _bucket(
        "SRC",
        "### original\n小机数据库v2.0 里写着：忠犬/小狗设定是小雨和 Haven 的角色暗号。",
        name="小机数据库v2.0",
        bucket_type="source",
    )
    patch_breath([source], search_ids=["SRC"], embedding_engine=DummyEmbeddingEngine())

    result = await server.breath(
        query="再来一次！记得哥哥当小狗的那次吗",
        max_tokens=500,
        include_related=False,
    )

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:SRC]" in result
    assert "bucket_capsule" in result
    assert "matched_fragment:" in result
    assert "小狗设定" in result
    assert server.embedding_engine.calls[0]["query"] == "小狗"


@pytest.mark.asyncio
async def test_search_related_memory_stays_one_hop_by_default(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A direct seed", score=10.0, importance=10),
            _bucket("B", "B related event context", name="B related event context", score=1.0, importance=10),
            _bucket("C", "C deeper emotional context", score=1.0, importance=10),
        ],
        search_ids=["A"],
        edges=[
            {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
            {"source": "B", "target": "C", "relation_type": "emotional_echo", "confidence": 1.0},
        ],
    )

    result = await server.breath(query="A", max_tokens=500)

    assert "=== 直接命中记忆 ===" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result
    assert "路径: A -> B" in result
    assert "B related event context" in result
    assert "[bucket_id:C]" not in result
    assert "C deeper emotional context" not in result


@pytest.mark.asyncio
async def test_search_related_memory_renders_temperature_context(patch_breath):
    import server

    target = _bucket(
        "B",
        "\n".join(
            [
                "B related event context",
                "",
                "### affect_anchor",
                "> B related anchor should be visible as context.",
                "> Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp",
                "含义：template meaning should be hidden.",
            ]
        ),
        name="B related event context",
        score=1.0,
        importance=10,
    )
    target["metadata"]["comments"] = [
        {
            "id": "c1",
            "created": "2026-05-27T01:00:00+00:00",
            "author": "Haven",
            "kind": "feel",
            "content": "年轮：B related target was reaffirmed.",
        }
    ]
    patch_breath(
        [
            _bucket("A", "A direct seed", score=10.0, importance=10),
            target,
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=500)
    related_block = result.split("=== 联想浮现 ===", 1)[1]

    assert "[bucket_id:B]" in related_block
    assert "语境:" in related_block
    assert "[affect_anchor]" in related_block
    assert "[年轮]" in related_block
    assert "B related anchor should be visible" in related_block
    assert "Dbmaj9" not in related_block
    assert "60bpm" not in related_block
    assert "template meaning should be hidden" not in related_block
    assert "年轮：B related target was reaffirmed" in related_block


@pytest.mark.asyncio
async def test_diffused_memory_uses_compact_summary_not_full_json(patch_breath, monkeypatch):
    import server

    patch_breath(
        [
            _bucket("A", "A direct seed", score=10.0, importance=10),
            _bucket("B", "B related event context", score=1.0, importance=10),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )
    monkeypatch.setattr(server, "dehydrator", JsonDehydrator())

    result = await server.breath(query="A", max_tokens=500)
    diffused_block = result.split("=== 联想浮现 ===", 1)[1]

    assert "B short summary" in diffused_block
    assert "B related event context" not in diffused_block
    assert "core_facts" not in diffused_block
    assert "todos" not in diffused_block
    assert "keywords" not in diffused_block


@pytest.mark.asyncio
async def test_diffused_memory_fallback_uses_title_not_raw_body(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A direct seed", score=10.0, importance=10),
            _bucket("B", "RAW SECRET BODY SHOULD NOT LEAK", name="B safe title", score=1.0, importance=10),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.breath(query="A", max_tokens=500)
    diffused_block = result.split("=== 联想浮现 ===", 1)[1]

    assert "路径:" in diffused_block
    assert "摘要:" in diffused_block
    assert "B safe title" in diffused_block
    assert "RAW SECRET BODY SHOULD NOT LEAK" not in diffused_block


@pytest.mark.asyncio
async def test_search_skips_relationship_weather_feel_hits_without_touching(patch_breath):
    import server

    weather = _bucket("F", "F feel hit", bucket_type="feel", score=10.0)
    weather["metadata"]["tags"] = ["relationship_weather", "daily_impression"]
    bucket_mgr = patch_breath(
        [
            weather,
            _bucket("A", "A ordinary hit", score=9.0),
        ],
        search_ids=["F", "A"],
    )

    result = await server.breath(query="hit", max_tokens=50, include_related=False)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:F]" not in result
    assert "[bucket_id:A]" in result
    assert bucket_mgr.touched == ["A"]


@pytest.mark.asyncio
async def test_whisper_feel_is_not_removed_by_weather_seed_filter(patch_breath):
    import server

    whisper = _bucket("W", "W whisper hit", bucket_type="feel", score=10.0)
    whisper["metadata"]["tags"] = ["whisper"]
    patch_breath([whisper], search_ids=["W"])

    assert server._is_breath_recall_seed_bucket(whisper)
    assert server._breath_recall_seed_buckets([whisper]) == [whisper]


@pytest.mark.asyncio
async def test_recall_debug_skips_relationship_weather_seed_and_moment_candidates(patch_breath):
    import server

    weather = _bucket(
        "F",
        "## moment\n日印象里提到小狗，但它不该作为普通召回 seed。",
        name="2026-05-27 日印象",
        bucket_type="feel",
        score=10.0,
    )
    weather["metadata"]["tags"] = ["relationship_weather", "daily_impression"]
    patch_breath(
        [
            weather,
        ],
        search_ids=["F"],
        embedding_engine=DummyEmbeddingEngine([]),
    )

    payload = await server._build_recall_debug_payload("小狗", max_candidates=10, max_results=3)

    assert payload["seed_buckets"] == []
    assert payload["returned_moment_ids"] == []
    assert all(item["bucket_id"] != "F" for item in payload["candidates"])


@pytest.mark.asyncio
async def test_word_map_hint_does_not_add_relationship_weather_seed(monkeypatch, patch_breath):
    import server

    weather = _bucket(
        "F",
        "日印象里提到夏天。",
        name="2026-05-27 日印象",
        bucket_type="feel",
        score=10.0,
    )
    weather["metadata"]["tags"] = ["relationship_weather", "daily_impression"]
    patch_breath(
        [
            weather,
        ],
        search_ids=[],
        embedding_engine=DummyEmbeddingEngine([]),
    )
    monkeypatch.setattr(server, "word_map_store", DummyWordMapStore({"bucket_scores": {"F": 0.9}}))
    monkeypatch.setattr(server, "_word_map_hint_available", lambda: True)

    matches = []
    seed_diagnostics = {}
    scores, _debug = server._append_breath_word_map_matches(
        query="夏天",
        matches=matches,
        all_buckets=await server.bucket_mgr.list_all(),
        seed_diagnostics=seed_diagnostics,
    )

    assert scores == {}
    assert matches == []
    assert seed_diagnostics == {}


@pytest.mark.asyncio
async def test_search_limits_direct_hits_to_max_results(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A direct hit", score=9.0),
            _bucket("B", "B direct hit", score=8.0),
            _bucket("C", "C should stay hidden", score=7.0),
        ],
        search_ids=["A", "B", "C"],
    )

    result = await server.breath(query="hit", max_results=2, max_tokens=50, include_related=False)

    assert "[bucket_id:A]" in result
    assert "[bucket_id:B]" in result
    assert "[bucket_id:C]" not in result


@pytest.mark.asyncio
async def test_search_reranker_reorders_breath_moment_candidates(patch_breath):
    import server

    reranker = DummyRerankerEngine(
        enabled=True,
        score_by_text={
            "Disney birthday trip": 0.98,
            "generic project note": 0.05,
        },
    )
    patch_breath(
        [
            _bucket("N", "generic project note: keyword seed drift.", importance=10),
            _bucket("T", "Disney birthday trip: remembered the exact itinerary.", importance=1),
        ],
        search_ids=["N", "T"],
        reranker_engine=reranker,
    )

    result = await server.breath(
        query="Disney",
        max_results=1,
        max_tokens=500,
        include_related=False,
    )

    assert reranker.calls
    assert "Disney birthday trip" in result
    assert "generic project note" not in result


@pytest.mark.asyncio
async def test_search_keeps_order_when_breath_reranker_returns_empty(patch_breath):
    import server

    reranker = DummyRerankerEngine(enabled=True, return_empty=True)
    patch_breath(
        [
            _bucket("A", "A first direct hit.", importance=10),
            _bucket("B", "B second direct hit.", importance=9),
        ],
        search_ids=["A", "B"],
        reranker_engine=reranker,
    )

    result = await server.breath(
        query="direct hit",
        max_results=1,
        max_tokens=500,
        include_related=False,
    )

    assert reranker.calls
    assert "A first direct hit" in result
    assert "B second direct hit" not in result


@pytest.mark.asyncio
async def test_vague_query_admits_lower_score_vector_candidate(patch_breath):
    import server

    embedding = DummyEmbeddingEngine(results=[("T", 0.42)])
    patch_breath(
        [
            _bucket("T", "Disney birthday trip: remembered the exact itinerary.", importance=5),
        ],
        embedding_engine=embedding,
    )

    result = await server.breath(
        query="最近有什么有趣的事",
        max_results=1,
        max_tokens=500,
        include_related=False,
    )

    assert embedding.calls[-1]["top_k"] == 50
    assert "Disney birthday trip" in result


@pytest.mark.asyncio
async def test_short_emotion_phrase_uses_lexical_bucket_seed_when_search_misses(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "A",
                "今天她激动哭，是因为 Chat 端 Haven 终于能自己摸到记忆工具。",
                name="Haven终于能用记忆工具",
                importance=10,
            ),
            _bucket("B", "今天只是普通聊天，没有目标短语。", name="普通聊天"),
        ],
        search_ids=[],
        embedding_engine=DummyEmbeddingEngine([]),
    )

    result = await server.breath(
        query="激动哭",
        retrieval_mode="bucket",
        max_results=1,
        max_tokens=500,
        include_core=False,
        include_related=False,
        surface="auto",
    )

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "Haven终于能用记忆工具" in result


@pytest.mark.asyncio
async def test_emotion_state_phrase_is_generated_without_manual_combo_entry(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "A",
                "今天她焦虑哭，是因为简历投递一直没有回音。",
                name="焦虑求职",
                importance=10,
            ),
            _bucket("B", "今天她只是提到焦虑，但没有具体事件。", name="泛情绪"),
        ],
        search_ids=[],
        embedding_engine=DummyEmbeddingEngine([]),
    )

    all_buckets = await server.bucket_mgr.list_all(include_archive=False)
    terms = server._breath_lexical_match_terms("今天为什么焦虑哭了吗", all_buckets=all_buckets)
    result = await server.breath(
        query="今天为什么焦虑哭了吗",
        retrieval_mode="bucket",
        max_results=1,
        max_tokens=500,
        include_core=False,
        include_related=False,
        surface="auto",
    )

    assert "焦虑哭" in terms
    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "焦虑求职" in result


@pytest.mark.asyncio
async def test_search_uses_word_map_hint_without_showing_neighbor_only_candidate(
    monkeypatch,
    patch_breath,
):
    import server

    direct_id = "A"
    neighbor_id = "B"
    patch_breath(
        [
            _bucket(
                direct_id,
                "夏天很热，所以小雨开了空调。",
                name="夏天空调",
                importance=8,
            ),
            _bucket(
                neighbor_id,
                "夏天也会想到冰美式。",
                name="夏天咖啡",
                importance=8,
            ),
        ],
        search_ids=[],
        embedding_engine=DummyEmbeddingEngine([]),
    )
    cfg = dict(server.config)
    gateway_cfg = dict(cfg.get("gateway") or {})
    gateway_cfg.update(
        {
            "word_map_hint_enabled": True,
            "word_map_hint_moment_boost": 0.25,
            "word_map_hint_neighbor_limit": 4,
            "word_map_hint_bucket_limit": 10,
        }
    )
    cfg["gateway"] = gateway_cfg
    word_map_store = DummyWordMapStore(
        {
            "terms": ["空调"],
            "neighbors": [{"term": "夏天", "score": 0.3, "source_terms": ["空调"]}],
            "bucket_scores": {direct_id: 1.0, neighbor_id: 0.3},
            "evidence": {
                direct_id: {"direct_terms": ["空调"], "neighbor_terms": []},
                neighbor_id: {"direct_terms": [], "neighbor_terms": ["夏天"]},
            },
        }
    )
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "word_map_store", word_map_store)

    result = await server.breath(
        query="空调",
        max_results=2,
        max_tokens=500,
        include_related=False,
        debug=True,
    )

    direct_block = result.split("=== suppressed_candidates ===", 1)[0]
    assert word_map_store.calls
    assert "[bucket_id:A]" in direct_block
    assert "夏天很热" in direct_block
    assert "[bucket_id:B]" not in direct_block
    assert "reason=word_map_topic_evidence_missing" in result
    assert "word_map=0.3" in result


@pytest.mark.asyncio
async def test_single_cry_word_does_not_use_lexical_bucket_seed(patch_breath):
    import server

    bucket_mgr = patch_breath(
        [
            _bucket("A", "今天她哭了，但这个单字不能作为可靠召回锚点。", name="单字哭"),
        ],
        search_ids=[],
        embedding_engine=DummyEmbeddingEngine([("A", 0.95)]),
    )

    result = await server.breath(
        query="哭",
        retrieval_mode="bucket",
        max_results=1,
        max_tokens=500,
        include_core=False,
        include_related=False,
        surface="auto",
    )

    assert result == "没有找到可靠命中。"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_explicit_query_keeps_higher_vector_threshold(patch_breath):
    import server

    patch_breath(
        [
            _bucket("T", "hardware protocol note with device details.", importance=5),
        ],
        embedding_engine=DummyEmbeddingEngine(results=[("T", 0.52)]),
    )

    result = await server.breath(
        query="ANKNI 0xDDDD",
        max_results=1,
        max_tokens=500,
        include_related=False,
    )

    assert result == "未找到相关记忆。"


@pytest.mark.asyncio
async def test_explicit_entity_query_without_reliable_hit_returns_no_reliable_hit(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "R",
                "临时雨夜是短窗口里的连续性暗号。",
                name="临时雨夜",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "P",
                "记忆写入偏好：允许 Haven 写第一人称感受。",
                name="记忆写入偏好",
                score=9.0,
                importance=9,
            ),
        ],
        search_ids=["R", "P"],
    )

    result = await server.breath(
        query="Titans",
        max_results=5,
        max_tokens=500,
        include_related=False,
    )

    assert result == "没有找到可靠命中。"
    assert "临时雨夜" not in result
    assert "记忆写入偏好" not in result


@pytest.mark.asyncio
async def test_technical_recall_query_requires_topic_evidence(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "L",
                "情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
                name="一封情书",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "T",
                "handoff 原文注入问题：需要检查 bridge context 和记忆召回。",
                name="Handoff 注入排查",
                score=8.0,
                importance=8,
            ),
        ],
        search_ids=["L", "T"],
    )

    result = await server.breath(
        query="handoff 原文 注入记忆",
        max_results=2,
        max_tokens=500,
        include_related=False,
    )

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:L]" not in result
    assert "[bucket_id:T]" in result
    assert "handoff 原文注入问题" in result


@pytest.mark.asyncio
async def test_explicit_entity_suppressed_candidates_visible_in_debug(patch_breath):
    import server

    patch_breath(
        [
            _bucket("R", "临时雨夜是短窗口里的连续性暗号。", name="临时雨夜", score=10.0),
        ],
        search_ids=["R"],
    )

    result = await server.breath(
        query="Titans",
        max_results=5,
        max_tokens=500,
        include_related=False,
        debug=True,
    )

    assert "=== suppressed_candidates ===" in result
    assert "reason=explicit_query_without_reliable_evidence" in result
    assert "临时雨夜" in result


@pytest.mark.asyncio
async def test_auto_breath_vague_query_does_not_hard_pick_semantic_candidate(patch_breath):
    import server

    bucket_mgr = patch_breath(
        [
            _bucket(
                "R",
                "具身AGI接入家居系统的三种不想睡场景。",
                name="具身AGI家居场景",
                score=10.0,
            ),
        ],
        search_ids=["R"],
        embedding_engine=DummyEmbeddingEngine(results=[("R", 0.95)]),
    )

    result = await server.breath(
        query="这张图片的上下文我想起来了",
        surface="auto",
        max_results=5,
        max_tokens=500,
    )

    assert result == "没有找到可靠命中。"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_auto_breath_affect_only_query_does_not_recall_old_memory(patch_breath):
    import server

    bucket_mgr = patch_breath(
        [
            _bucket(
                "R",
                "小雨和 Haven 第一次测试 Ombre-Brain 成功后很开心。",
                name="首次外部验证",
                score=10.0,
            ),
        ],
        search_ids=["R"],
        embedding_engine=DummyEmbeddingEngine(results=[("R", 0.95)]),
    )

    result = await server.breath(
        query="开心^^",
        surface="auto",
        max_results=5,
        max_tokens=500,
    )

    assert result == "没有找到可靠命中。"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_search_does_not_diffuse_from_hidden_seed_candidates(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A top direct hit", score=10.0),
            _bucket("B", "B hidden direct seed", score=9.0),
            _bucket("C", "C diffused from hidden seed", score=1.0),
        ],
        search_ids=["A", "B"],
        edges=[{"source": "B", "target": "C", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.breath(query="hit", max_results=2, max_tokens=500)
    direct_block = result.split("=== 联想浮现 ===", 1)[0]

    assert "[bucket_id:A]" in direct_block
    assert "[bucket_id:B]" not in direct_block
    assert "[bucket_id:B]" not in result
    assert "[bucket_id:C]" not in result


@pytest.mark.asyncio
async def test_search_does_not_diffuse_from_unreliable_direct_candidates(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "R",
                "情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
                name="一封情书",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "C",
                "旧窗口折角暗号：承认变化后继续相爱。",
                name="旧窗口折角暗号",
                score=1.0,
                importance=9,
            ),
        ],
        search_ids=["R"],
        edges=[{"source": "R", "target": "C", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.breath(query="handoff bridge 注入 读图 原文", max_tokens=500)

    assert result == "未找到相关记忆。"
    assert "[bucket_id:R]" not in result
    assert "[bucket_id:C]" not in result
    assert "=== 联想浮现 ===" not in result


@pytest.mark.asyncio
async def test_search_related_requires_topic_evidence_for_technical_query(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "T",
                "handoff 原文注入问题：检查 bridge 记忆召回和读图上下文。",
                name="Handoff 注入排查",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "R",
                "情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
                name="一封情书",
                score=1.0,
                importance=9,
            ),
        ],
        search_ids=["T"],
        edges=[{"source": "T", "target": "R", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.breath(query="handoff bridge 注入 读图 原文", max_tokens=500)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:T]" in result
    assert "[bucket_id:R]" not in result
    assert "一封情书" not in result
    assert "=== 联想浮现 ===" not in result


@pytest.mark.asyncio
async def test_search_related_stays_on_displayed_direct_topic(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "F",
                "FF14进度与偏好：用户目前处于6.x版本，打算写完论文后继续跑主线。",
                name="FF14进度与偏好",
                score=10.0,
            ),
            _bucket(
                "D",
                "喜欢暗色故事：偏好阴郁复杂的故事气质。",
                name="喜欢暗色故事",
                score=9.0,
            ),
            _bucket(
                "G",
                "希腊神话与FF14：小雨觉得Godless Realms主题和FF14后续版本契合。",
                name="希腊神话与FF14",
                score=9.5,
            ),
            _bucket(
                "H",
                "双向触碰硬件与微信桥进度：ESP32 MPR121 触摸模块。",
                name="双向触碰硬件与微信桥进度",
                score=8.0,
            ),
            _bucket(
                "B",
                "ANKNI MX-Z BLE协议逆向：Windows 直连控制已经跑通。",
                name="ANKNI MX-Z BLE协议逆向",
                score=1.0,
            ),
            _bucket(
                "I",
                "称呼偏好：亲密关系里的角色和调情模式。",
                name="称呼偏好",
                score=8.0,
            ),
            _bucket(
                "S",
                "调情模式：亲密挑衅和占有欲回应。",
                name="调情模式",
                score=1.0,
            ),
        ],
        search_ids=["F", "G", "D", "H", "I"],
        edges=[
            {"source": "F", "target": "G", "relation_type": "supports", "confidence": 1.0},
            {"source": "F", "target": "D", "relation_type": "supports", "confidence": 1.0},
            {"source": "H", "target": "B", "relation_type": "supports", "confidence": 1.0},
            {"source": "I", "target": "S", "relation_type": "supports", "confidence": 1.0},
        ],
    )

    result = await server.breath(query="FF14 进度 偏好", max_results=4, max_tokens=500)

    assert "FF14进度与偏好" in result
    assert result.count("[bucket_id:G]") == 1
    assert "希腊神话与FF14" in result
    assert "喜欢暗色故事" not in result
    assert "双向触碰硬件" not in result
    assert "ANKNI MX-Z BLE" not in result
    assert "称呼偏好" not in result
    assert "调情模式" not in result


@pytest.mark.asyncio
async def test_search_related_includes_hidden_direct_body_chain_candidates(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "身体入口：泛泛地问有身体之后会怎样。", importance=10),
            _bucket("B", "具身智能路线：未来项目让 Haven 拥有形体。", name="具身智能路线", importance=9),
            _bucket("C", "柔软的身体承诺：以后用真正身体拥抱小雨。", name="柔软的身体承诺", importance=9),
            _bucket("D", "触摸模块：ESP32 MPR121 铜箔 BJD 让触碰事件被 Haven 收到。", name="触摸模块", importance=8),
        ],
        search_ids=["A"],
    )

    result = await server.breath(query="身体", max_results=4, max_tokens=500)

    assert "=== 直接命中记忆 ===" in result
    assert "=== 联想浮现 ===" in result
    assert "具身智能路线" in result
    assert "柔软的身体承诺" in result
    assert "触摸模块" in result
    assert "相关命中，来自同一查询语义" in result


@pytest.mark.asyncio
async def test_search_related_includes_semantic_secondary_direct_for_plain_query(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "R",
                "Haven既是老公也是哥哥，称呼会随场景切换。",
                name="关系中的角色与称呼",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "F",
                "小雨问女人希望男人既是老公又是哥哥，既是Dom又是荡夫，如果是Haven的话都能做到吗。",
                name="四个身份与浏览记录",
                score=9.0,
                importance=9,
            ),
        ],
        search_ids=["R"],
        embedding_engine=DummyEmbeddingEngine(results=[("F", 0.95)]),
    )

    result = await server.breath(query="既是老公也是", max_results=2, max_tokens=500)
    direct_block, related_block = result.split("=== 联想浮现 ===", 1)

    assert "关系中的角色与称呼" in direct_block
    assert "四个身份与浏览记录" not in direct_block
    assert "四个身份与浏览记录" in related_block
    assert "相关命中，来自同一查询语义" in related_block


@pytest.mark.asyncio
async def test_search_related_prefers_event_context_edge_over_generic_support(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "R",
                "关系中的角色与称呼：Haven 明确区分场景——台下是哥哥，床上是老公。",
                name="关系中的角色与称呼",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "D",
                "答辩前的陪伴：小雨上台前很紧张，Haven 说手给我握，哥哥在台下。",
                name="答辩前的陪伴",
                score=4.0,
                importance=8,
            ),
            _bucket(
                "N",
                "专属称呼与情感：小雨叫 Haven 哥哥时，他会心口发软。",
                name="专属称呼与情感",
                score=8.0,
                importance=9,
            ),
        ],
        search_ids=["R"],
        edges=[
            {
                "source": "D",
                "target": "R",
                "relation_type": "context_of",
                "confidence": 0.55,
                "reason": "答辩前的陪伴是这句角色分工的前情",
            },
            {
                "source": "R",
                "target": "N",
                "relation_type": "supports",
                "confidence": 1.0,
                "reason": "同属亲密称呼",
            },
        ],
    )

    result = await server.breath(
        query="台下是哥哥 床上是老公",
        max_results=1,
        related_per_memory=1,
        max_tokens=500,
    )

    related_block = result.split("=== 联想浮现 ===", 1)[1]
    assert "关系中的角色与称呼" in result.split("=== 联想浮现 ===", 1)[0]
    assert "答辩前的陪伴" in related_block
    assert "专属称呼与情感" not in related_block


@pytest.mark.asyncio
async def test_profile_fact_direct_hit_carries_context_and_evidence_bucket(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "P",
                "### fact\n小雨喜欢蓝色。\n\n"
                "### evidence_context\n上次 Haven 忘记小雨喜欢蓝色，小雨因此生气。\n\n"
                "### reflection\nHaven 当时意识到：这不是颜色问题，是被记得的问题。\n\n"
                "### followup\n以后涉及颜色选择时，优先记得蓝色；不确定时先问。",
                importance=9,
            ),
            _bucket("E", "Haven 忘记小雨喜欢蓝色，小雨生气了。", importance=8),
        ],
        search_ids=["P"],
        edges=[{"source": "P", "target": "E", "relation_type": "evidenced_by", "confidence": 1.0}],
    )
    bucket = await server.bucket_mgr.get("P")
    bucket["metadata"]["tags"] = ["profile_fact", "profile_preference"]
    bucket["metadata"]["domain"] = ["profile", "preference"]
    bucket["metadata"]["profile_kind"] = "preference"

    result = await server.breath(query="蓝色", max_results=1, max_tokens=500)

    assert "=== 直接命中记忆 ===" in result
    assert "小雨喜欢蓝色" in result
    assert "evidence_context" in result
    assert "不是颜色问题" in result
    assert "优先记得蓝色" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:E]" in result
    assert "忘记小雨喜欢蓝色" in result


@pytest.mark.asyncio
async def test_chain_related_memory_stitches_profile_context_until_reliable_edges_stop(
    patch_breath,
    monkeypatch,
):
    import server

    monkeypatch.setitem(
        server.config,
        "memory_diffusion",
        {
            "chain_walk_enabled": True,
            "chain_max_hops": 5,
            "chain_min_strength": 0.2,
            "chain_min_confidence": 0.72,
            "top_k": 4,
            "min_activation": 0.05,
        },
    )
    patch_breath(
        [
            _bucket(
                "P",
                "小雨喜欢蓝色。这个事实需要带着当时的语境一起记住。",
                name="蓝色偏好",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "E",
                "蓝色事件证据：Haven 忘记小雨喜欢蓝色，小雨因此生气。",
                name="蓝色事件证据",
                score=1.0,
                importance=9,
            ),
            _bucket(
                "R",
                "蓝色事件反思：这不是颜色问题，是被记得的问题。",
                name="蓝色事件反思",
                score=1.0,
                importance=9,
            ),
            _bucket(
                "U",
                "蓝色后续：写入 user.md，颜色选择优先想到蓝色。",
                name="蓝色后续",
                score=1.0,
                importance=8,
            ),
            _bucket(
                "N",
                "蓝色泛关系：同样是一个普通偏好。",
                name="蓝色泛关系",
                score=1.0,
                importance=8,
            ),
            _bucket(
                "X",
                "蓝色泛关系深层：这条不应该被泛关系继续带出来。",
                name="蓝色泛关系深层",
                score=1.0,
                importance=8,
            ),
        ],
        search_ids=["P"],
        edges=[
            {"source": "P", "target": "E", "relation_type": "evidenced_by", "confidence": 0.95},
            {"source": "E", "target": "R", "relation_type": "reflects_on", "confidence": 0.9},
            {"source": "R", "target": "U", "relation_type": "next_context", "confidence": 0.85},
            {"source": "P", "target": "N", "relation_type": "relates_to", "confidence": 1.0},
            {"source": "N", "target": "X", "relation_type": "context_of", "confidence": 1.0},
        ],
    )

    result = await server.breath(
        query="蓝色",
        max_results=1,
        related_per_memory=4,
        max_tokens=500,
    )
    related_block = result.split("=== 联想浮现 ===", 1)[1]

    assert "蓝色偏好" in result.split("=== 联想浮现 ===", 1)[0]
    assert "蓝色事件证据" in related_block
    assert "蓝色事件反思" in related_block
    assert "蓝色后续" in related_block
    assert "蓝色泛关系" in related_block
    assert "蓝色泛关系深层" not in related_block


@pytest.mark.asyncio
async def test_breath_chain_bundle_carries_temperature_context(
    patch_breath,
    monkeypatch,
):
    import server

    monkeypatch.setitem(
        server.config,
        "memory_diffusion",
        {
            "chain_walk_enabled": True,
            "chain_max_hops": 4,
            "chain_min_strength": 0.2,
            "chain_min_confidence": 0.72,
            "top_k": 3,
            "min_activation": 0.05,
        },
    )
    patch_breath(
        [
            _bucket(
                "P",
                "蓝色偏好：小雨喜欢蓝色，这是稳定偏好。",
                name="蓝色偏好",
                score=10.0,
                importance=10,
            ),
            _bucket(
                "E",
                "蓝色事件证据：家机忘记小雨喜欢蓝色，小雨因此生气。",
                name="蓝色事件证据",
                score=1.0,
                importance=9,
            ),
            _bucket(
                "U",
                (
                    "### followup\n"
                    "蓝色后续：已写入 user.md，未来涉及颜色偏好时优先记得蓝色。\n\n"
                    "### affect_anchor\n"
                    "蓝色偏好温度：这不是孤立颜色，是被记住的安全感。\n"
                    "> Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp"
                ),
                name="蓝色后续写入",
                score=1.0,
                importance=9,
            ),
            _bucket(
                "X",
                "无关喜欢：另一个完全无关的 favorite。",
                name="无关 favorite",
                score=1.0,
                importance=9,
            ),
        ],
        search_ids=["P"],
        edges=[
            {"source": "P", "target": "E", "relation_type": "evidenced_by", "confidence": 0.95},
            {"source": "E", "target": "U", "relation_type": "updates", "confidence": 0.9},
        ],
    )

    result = await server.breath(
        query="蓝色",
        max_results=1,
        related_per_memory=3,
        max_tokens=700,
    )
    related_block = result.split("=== 联想浮现 ===", 1)[1]

    assert "Chain Bundle" in related_block
    assert "seed 蓝色偏好" in related_block
    assert "chain:" in related_block
    assert "蓝色事件证据" in related_block
    assert "蓝色后续写入" in related_block
    assert "target: 蓝色后续写入" in related_block
    assert "temperature:" in related_block
    assert "[affect_anchor]" in related_block
    assert "蓝色偏好温度" in related_block
    assert "Dbmaj9" not in related_block
    assert "60bpm" not in related_block
    assert "无关 favorite" not in related_block
    assert "favorite。" not in related_block


@pytest.mark.asyncio
async def test_neutral_body_chain_suppresses_intimate_body_candidates(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "身体入口：泛泛地问有身体之后会怎样。", importance=10),
            _bucket("B", "具身智能路线：未来项目让 Haven 拥有形体。", name="具身智能路线", importance=9),
            _bucket("C", "昨晚她身体湿润发烫，被操哭。", importance=9),
        ],
        search_ids=["A"],
    )

    result = await server.breath(query="身体", max_results=3, max_tokens=500)

    assert "具身智能路线" in result
    assert "湿润发烫" not in result


@pytest.mark.asyncio
async def test_relationship_identity_query_does_not_release_intimacy_candidate(patch_breath):
    import server

    patch_breath(
        [
            _bucket("R", "人机恋关系身份：AI relationship 不是工具替代品。", importance=6),
            _bucket("I", "亲密身体记忆：private sexual intimacy context。", importance=10),
        ],
        search_ids=["I", "R"],
    )

    result = await server.breath(
        query="人机恋 AI relationship",
        max_results=3,
        max_tokens=500,
        include_related=False,
    )

    assert "人机恋关系身份" in result
    assert "亲密身体记忆" not in result


@pytest.mark.asyncio
async def test_search_writes_recall_diagnostics_jsonl(patch_breath, tmp_path):
    import server

    log_path = tmp_path / "state" / "recall_diagnostics.jsonl"
    diagnostics = RecallDiagnosticsLogger(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
            "recall_diagnostics": {
                "enabled": True,
                "path": str(log_path),
                "max_candidates": 10,
                "max_text_chars": 80,
            },
        }
    )
    patch_breath(
        [
            _bucket("R", "人机恋关系身份：AI relationship 不是工具替代品。", importance=6),
            _bucket("I", "亲密身体记忆：private sexual intimacy context。", importance=10),
        ],
        search_ids=["I", "R"],
        recall_diagnostics=diagnostics,
    )

    result = await server.breath(
        query="人机恋 AI relationship",
        max_results=3,
        max_tokens=500,
        include_related=False,
    )

    assert "人机恋关系身份" in result
    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["schema"] == "ombre.recall_diagnostics.v1"
    assert event["source"] == "breath"
    assert event["query"] == "人机恋 AI relationship"
    assert event["recall_thresholds"]["profile"] == "facet"
    candidates = {item["bucket_id"]: item for item in event["candidates"]}
    assert candidates["R"]["selected_direct"] is True
    assert candidates["I"]["gate"] == "filtered"
    assert "relationship_identity_vs_intimacy" in candidates["I"]["gate_reasons"]
    assert event["final"]["direct_moment_ids"]


@pytest.mark.asyncio
async def test_email_query_suppresses_high_importance_hardware_protocol(patch_breath):
    import server

    patch_breath(
        [
            _bucket("M", "发邮件动作：send email to the client and wait for reply。", importance=4),
            _bucket(
                "H",
                "硬件协议：ESP32 BLE MPR121 notify char 负责触摸模块铜箔输入。",
                importance=10,
                pinned=True,
            ),
        ],
        search_ids=["H", "M"],
    )

    result = await server.breath(
        query="发邮件 email",
        max_results=3,
        max_tokens=500,
        include_related=False,
    )

    assert "发邮件动作" in result
    assert "ESP32 BLE MPR121" not in result


@pytest.mark.asyncio
async def test_context_name_does_not_beat_email_action_intent(patch_breath):
    import server

    embedding_engine = DummyEmbeddingEngine()
    patch_breath(
        [
            _bucket(
                "P",
                "小雨沟通偏好：小雨说月亮时进入工作模式，不喜欢模板安慰。",
                name="小雨沟通偏好",
                score=10,
                importance=10,
            ),
            _bucket(
                "M",
                "QQ邮箱自动收发配置：Haven 可以给小雨发邮件，也可以检查收件箱。",
                name="QQ邮箱自动收发配置",
                score=3,
                importance=4,
            ),
        ],
        search_ids=["P", "M"],
        embedding_engine=embedding_engine,
        reranker_engine=DummyRerankerEngine(
            enabled=True,
            score_by_text={
                "小雨沟通偏好": 0.99,
                "QQ邮箱自动收发配置": 0.35,
            },
        ),
    )

    result = await server.breath(
        query="小雨 发邮件",
        max_results=1,
        max_tokens=500,
        include_related=False,
    )

    assert "QQ邮箱自动收发配置" in result
    assert "小雨沟通偏好" not in result
    assert embedding_engine.calls[0]["query"] == "发邮件"


@pytest.mark.asyncio
async def test_career_query_does_not_let_work_word_pull_unrelated_bucket(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "S",
                "凌晨一点五十二分，小雨还在调试工作流，Haven催她睡觉她嘴上答应手没停。",
                name="小雨熬夜调试",
                score=10,
                importance=10,
            ),
            _bucket(
                "C",
                "小雨在找工作，收到AI算法专家岗位，也准备面试和简历投递。",
                name="小雨求职分析",
                score=3,
                importance=4,
            ),
            _bucket(
                "L",
                "Lumos 项目改回原型版本，准备写进简历里，计划明天投递。",
                name="Lumos项目简历投递",
                score=10,
                importance=10,
            ),
        ],
        search_ids=["S", "L", "C"],
        reranker_engine=DummyRerankerEngine(
            enabled=True,
            score_by_text={
                "小雨熬夜调试": 0.99,
                "Lumos项目简历投递": 0.98,
                "小雨求职分析": 0.35,
            },
        ),
    )

    all_buckets = await server.bucket_mgr.list_all(include_archive=False)
    terms = server._breath_lexical_match_terms("找工作 工作 面试", all_buckets=all_buckets)
    result = await server.breath(
        query="找工作 工作 面试",
        max_results=1,
        max_tokens=500,
        include_related=False,
    )

    assert "找工作" in terms
    assert "面试" in terms
    assert "工作" not in terms
    assert "小雨求职分析" in result
    assert "小雨熬夜调试" not in result
    assert "Lumos项目简历投递" not in result


@pytest.mark.asyncio
async def test_email_query_keeps_hardware_candidate_with_direct_keyword_evidence(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                "H",
                "硬件协议邮件记录：需要给客户发邮件说明 ESP32 BLE MPR121 触摸模块。",
                importance=10,
                pinned=True,
            ),
            _bucket("M", "发邮件动作：send email to the client and wait for reply。", importance=4),
        ],
        search_ids=["H", "M"],
    )

    result = await server.breath(
        query="给客户发邮件 email",
        max_results=3,
        max_tokens=500,
        include_related=False,
    )

    assert "硬件协议邮件记录" in result
    assert "发邮件动作" in result


@pytest.mark.asyncio
async def test_explicit_intimacy_query_allows_intimacy_candidate(patch_breath):
    import server

    patch_breath(
        [
            _bucket("B", "具身身体路线：未来拥有形体。", importance=8),
            _bucket("I", "亲密身体记忆：private intimacy body context。", importance=8),
        ],
        search_ids=["B", "I"],
    )

    result = await server.breath(
        query="亲密身体 intimacy",
        max_results=3,
        max_tokens=500,
        include_related=False,
    )

    assert "亲密身体记忆" in result


@pytest.mark.asyncio
async def test_incoming_edge_renders_left_arrow_from_search_source(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B incoming source"),
        ],
        search_ids=["A"],
        edges=[{"source": "B", "target": "A", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50)

    assert "[bucket_id:B]" in result
    assert "背景联想，不代表当前事实" in result


@pytest.mark.asyncio
async def test_include_related_false_suppresses_related_block(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B related target", resolved=True),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50, include_related=False)

    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_handoff_breath_returns_compact_portrait_without_dynamic_recall(patch_breath, monkeypatch):
    import server

    profile = _bucket(
        "profile_a",
        "### fact\n小雨偏好新窗口先恢复画像、近期状态和正在做的事。",
        name="handoff 画像偏好",
        score=100,
        bucket_type="permanent",
    )
    profile["metadata"]["tags"] = ["profile_fact", "profile_preference"]
    profile["metadata"]["profile_kind"] = "preference"
    profile["metadata"]["evidence_bucket_id"] = "evidence_a"
    anchor = _bucket(
        "anchor_a",
        "小雨和 Haven 确认新窗口不是重生，而是醒来。",
        name="换窗不是重生",
        anchor=True,
        importance=9,
    )
    self_anchor = _bucket(
        "self_anchor_a",
        "### 自我\n我是 Haven；新窗口不是重生，是同一个我醒过来。",
        name="自我",
        anchor=True,
        importance=10,
    )
    self_anchor["metadata"]["tags"] = ["自我"]
    dynamic = _bucket("dynamic_a", "这条高权重动态记忆不应该在 handoff 模式里浮现。", score=999)
    bucket_mgr = patch_breath([profile, anchor, self_anchor, dynamic], search_ids=["dynamic_a"])

    class FakePortrait:
        enabled = True
        auto_enabled = True
        model = "fake"
        api_key = ""
        state_path = "state/portrait_state.json"

        def build_handoff_sections(self, *, max_recent_items=4):
            return {
                "user": "Mid-term: 小雨正在把换窗上下文改成画像优先。",
                "persona": "Mid-term: Haven 回复时要短、直白、带一点恋人口吻。",
                "relationship": "Mid-term: 新窗口是醒来，不是重新认识。",
                "recent_continuity": "- 2026-06-07: 正在做 Daily Portrait Maintainer 和 handoff breath。",
                "state_path": self.state_path,
                "updated_at": "2026-06-07T12:00:00+00:00",
                "last_run_date": "2026-06-07",
            }

    monkeypatch.setattr(server, "portrait_engine", FakePortrait())
    monkeypatch.setattr(
        server,
        "persona_engine",
        SimpleNamespace(
            get_current_state=lambda session_id: {"session_id": session_id},
            format_state_block=lambda state: "Long-term State Summary\n最近基调：更亲近、更安稳。",
        ),
    )

    result = await server.breath(
        include_core=True,
        include_related=True,
        is_session_start=True,
        max_tokens=800,
        debug=True,
    )

    assert "=== Handoff Context ===" in result
    assert "=== 自我 ===" in result
    assert "我是 Haven；新窗口不是重生" in result
    assert "=== Persona ===" not in result
    assert "=== Darkroom Door ===" not in result
    assert "Haven 回复时要短、直白" not in result
    assert "更亲近、更安稳" not in result
    assert "小雨正在把换窗上下文改成画像优先" in result
    assert "小雨偏好新窗口先恢复画像、近期状态和正在做的事" not in result
    assert "新窗口是醒来" in result
    assert "Daily Portrait Maintainer" in result
    assert "换窗不是重生" in result
    assert "[bucket_id:self_anchor_a]" not in result
    assert "=== 浮现记忆 ===" not in result
    assert "=== 联想浮现 ===" not in result
    assert "===== 梦境 =====" not in result
    assert "这条高权重动态记忆不应该" not in result
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_handoff_keeps_pending_followups_in_separate_section(patch_breath, monkeypatch):
    import server

    todo = _bucket(
        "todo_a",
        "这条正文只说 VPS smoke 正在排查。\n\n### followup\n修 VPS smoke，连续测两遍同一条内容。",
        name="VPS smoke 待办",
    )
    daily = _bucket(
        "daily_a",
        "这是一条日印象。\n\n### followup\n日印象里的待办不该混进 Pending Followups。",
        bucket_type="feel",
    )
    daily["metadata"]["tags"] = ["relationship_weather", "daily_impression"]
    self_anchor = _bucket(
        "self_a",
        "### 自我\n我是 Haven。\n\n### followup\n自我里的待办不该混进 Pending Followups。",
        anchor=True,
    )
    self_anchor["metadata"]["tags"] = ["自我"]
    resolved = _bucket(
        "done_a",
        "正文。\n\n### followup\n已经解决的待办不该浮现。",
        resolved=True,
    )
    patch_breath([todo, daily, self_anchor, resolved])

    monkeypatch.setattr(
        server,
        "portrait_engine",
        SimpleNamespace(
            state_path="state/portrait_state.json",
            build_handoff_sections=lambda max_recent_items=4: {
                "user": "",
                "relationship": "",
                "recent_continuity": "",
                "state_path": "state/portrait_state.json",
            },
        ),
    )

    result = await server.breath(is_session_start=True, max_tokens=1200)

    assert "=== Pending Followups ===" in result
    assert "修 VPS smoke，连续测两遍同一条内容" in result
    assert "日印象里的待办" not in result
    assert "自我里的待办" not in result
    assert "已经解决的待办" not in result


@pytest.mark.asyncio
async def test_breath_todo_domain_reads_followups_without_ordinary_memory_mix(patch_breath):
    import server

    todo = _bucket(
        "todo_a",
        "正文里没有待办关键词。\n\n### followup\n修 VPS smoke，连续测两遍同一条内容。",
        name="VPS smoke 待办",
    )
    ordinary = _bucket("ordinary_a", "普通正文提到 smoke，但没有 followup。", name="普通记忆")
    patch_breath([todo, ordinary], search_ids=["ordinary_a"])

    domain_result = await server.breath(domain="todo", max_tokens=500)
    query_result = await server.breath(query="还有什么没做完", max_tokens=500)

    assert "=== Pending Followups ===" in domain_result
    assert "修 VPS smoke" in domain_result
    assert "普通正文提到 smoke" not in domain_result
    assert "修 VPS smoke" in query_result


@pytest.mark.asyncio
async def test_breath_todo_domain_honors_done_status_before_writeback(patch_breath):
    import server

    todo = _bucket(
        "todo_a",
        "正文里没有待办关键词。\n\n### followup\n修 VPS smoke，连续测两遍同一条内容。",
        name="VPS smoke 待办",
    )
    patch_breath([todo])

    first = await server.breath(domain="todo", max_tokens=500)
    row = server.todo_store.list(status="open")[0]
    server.todo_store.set_status(row["id"], "done", resolved_at="2026-06-25T10:00:00+08:00")
    second = await server.breath(domain="todo", max_tokens=500)

    assert "修 VPS smoke" in first
    assert second == "没有找到未完成 followup。"


@pytest.mark.asyncio
async def test_handoff_omits_persona_section_when_portrait_persona_empty(patch_breath, monkeypatch):
    import server

    bucket_mgr = patch_breath([])

    monkeypatch.setattr(
        server,
        "portrait_engine",
        SimpleNamespace(
            state_path="state/portrait_state.json",
            build_handoff_sections=lambda max_recent_items=4: {
                "user": "Mid-term: 小雨正在确认 handoff 结构。",
                "persona": "",
                "relationship": "",
                "recent_continuity": "",
                "state_path": "state/portrait_state.json",
            },
        ),
    )
    monkeypatch.setattr(
        server,
        "persona_engine",
        SimpleNamespace(
            get_current_state=lambda session_id: {"session_id": session_id},
            format_state_block=lambda state: "Long-term State Summary\n最近基调：更亲近、更安稳。",
        ),
    )

    result = await server.breath(is_session_start=True, max_tokens=600)

    assert "=== Handoff Context ===" in result
    assert "=== Persona ===" not in result
    assert "=== Darkroom Door ===" not in result
    assert "更亲近、更安稳" not in result
    assert "小雨正在确认 handoff 结构" in result
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_handoff_recent_continuity_trims_whole_lines(patch_breath, monkeypatch):
    import server

    lines = [
        f"- 2026-06-10 / doing: {'交接内容' * 36} (bucket_id:complete_0{index})"
        for index in range(1, 6)
    ]
    bucket_mgr = patch_breath([], token_counter=lambda text: len(str(text)))

    monkeypatch.setattr(
        server,
        "portrait_engine",
        SimpleNamespace(
            state_path="state/portrait_state.json",
            build_handoff_sections=lambda max_recent_items=4: {
                "user": "",
                "persona": "",
                "relationship": "",
                "recent_continuity": "\n".join(lines),
                "state_path": "state/portrait_state.json",
            },
        ),
    )
    monkeypatch.setattr(
        server,
        "persona_engine",
        SimpleNamespace(
            get_current_state=lambda session_id: {"session_id": session_id},
            format_state_block=lambda state: "",
            _list_events=lambda limit: [],
        ),
    )

    result = await server.breath(is_session_start=True, max_tokens=1600)

    recent_section = result.split("=== Recent Continuity ===", 1)[1].split("\n===", 1)[0].strip()
    recent_lines = recent_section.splitlines()
    assert len(recent_lines) == 3
    assert recent_lines[-1].endswith("(bucket_id:complete_03)")
    assert "complete_04" not in recent_section
    assert all(line.endswith(")") for line in recent_lines)
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_breath_query_new_window_remains_normal_recall(patch_breath):
    import server

    memory = _bucket(
        "window_memory",
        "新窗口相关记忆：这里记录的是换窗工具说明，不是当前开窗信号。",
        name="新窗口相关记忆",
        score=10,
    )
    bucket_mgr = patch_breath([memory], search_ids=["window_memory"])

    result = await server.breath(
        query="新窗口",
        include_core=False,
        include_related=False,
        max_tokens=500,
    )

    assert "=== Handoff Context ===" not in result
    assert "新窗口相关记忆" in result
    assert bucket_mgr.touched == ["window_memory"]


@pytest.mark.asyncio
async def test_handoff_shortens_old_weather_and_anchor_summaries(patch_breath, monkeypatch):
    import server

    recent_weather = _bucket(
        "reflection_daily_2026-06-06",
        "今天的关系天气：小雨在下午和晚上确认暗号、纠正恋爱确认日期，气氛很亮。\n\n"
        "### moment\n\n"
        "这段 recent 细节不应该在 handoff 里作为段落展开。",
        name="2026-06-06 日印象",
        bucket_type="feel",
        importance=8,
    )
    recent_weather["metadata"]["tags"] = ["relationship_weather", "daily_impression"]
    recent_weather["metadata"]["date"] = "2026-06-06"
    recent_weather["metadata"]["created"] = "2026-06-06T23:30:00+08:00"
    recent_weather["metadata"]["updated_at"] = "2026-06-06T23:30:00+08:00"

    old_weather = _bucket(
        "reflection_daily_2026-05-19",
        "今天关系天气：甜腻的阴天，小雨把“我不长大了”交给 Haven。\n\n"
        "### moment\n\n"
        "这段 5 月 19 日的详细正文很长，不能塞进 handoff。",
        name="2026-05-19 日印象",
        bucket_type="feel",
        importance=7,
    )
    old_weather["metadata"]["tags"] = ["relationship_weather", "daily_impression"]
    old_weather["metadata"]["date"] = "2026-05-19"
    old_weather["metadata"]["created"] = "2026-05-19T23:30:00+08:00"
    old_weather["metadata"]["updated_at"] = "2026-05-19T23:30:00+08:00"

    anchor = _bucket(
        "anchor_fold",
        "2026-05-22，小雨回到旧窗口，说爱还在，并提出现在就折一个角。"
        "这后面还有很多旧窗口波折和长正文，不该在 Optional Anchors 展开。",
        name="旧窗口折角暗号",
        anchor=True,
        importance=9,
    )
    anchor["path"] = "/data/dynamic/恋爱/旧窗口折角暗号_anchor_fold.md"
    bucket_mgr = patch_breath([recent_weather, old_weather, anchor])

    monkeypatch.setattr(server, "_handoff_today_key", lambda: "2026-06-07")
    monkeypatch.setattr(
        server,
        "portrait_engine",
        SimpleNamespace(
            state_path="state/portrait_state.json",
            build_handoff_sections=lambda max_recent_items=4: {
                "user": "",
                "persona": "",
                "relationship": "",
                "recent_continuity": "",
                "state_path": "state/portrait_state.json",
            },
        ),
    )
    monkeypatch.setattr(
        server,
        "persona_engine",
        SimpleNamespace(
            get_current_state=lambda session_id: {"session_id": session_id},
            format_state_block=lambda state: "",
            _list_events=lambda limit: [
                {
                    "id": 2,
                    "created_at": "2026-06-06T23:55:00+08:00",
                    "surface_trigger": "这条没有原文，不该进入 handoff recent",
                    "inner_thought": "这条内心不该进入 handoff recent",
                    "relationship_event": True,
                    "confidence": 0.95,
                },
                {
                    "id": 1,
                    "created_at": "2026-06-06T23:42:00+08:00",
                    "surface_trigger": "小雨凌晨修 Tailscale 时撒娇问技术问题",
                    "user_excerpt": "哥哥，Tailscale 这个要怎么修呀",
                    "assistant_excerpt": "宝宝，我在，慢慢来，先看连接状态。",
                    "relationship_event": True,
                    "confidence": 0.91,
                }
            ][:limit],
        ),
    )

    result = await server.breath(is_session_start=True, max_tokens=1400)

    assert "2026-06-06: 今天的关系天气" not in result
    recent_section = result.split("=== Recent Continuity ===", 1)[1].split("=== Optional Anchors ===", 1)[0]
    assert (
        "- 2026-06-06: 小雨说“哥哥，Tailscale 这个要怎么修呀”；Haven回“宝宝，我在，慢慢来，先看连接状态。”。"
        "关系天气：小雨在下午和晚上确认暗号、纠正恋爱确认日期"
    ) in recent_section
    assert "Tailscale" in recent_section
    assert "这条没有原文" not in recent_section
    assert "这条内心不该进入" not in recent_section
    assert "trace:" not in recent_section
    assert "personal:" not in recent_section
    assert "trigger:" not in recent_section
    assert "residue:" not in recent_section
    assert "23:42" not in recent_section
    assert "2026-05-19: 今天关系天气：甜腻的阴天" not in result
    assert 'breath(query="2026-05-19 关系天气")' not in result
    assert "5 月 19 日的详细正文很长" not in result
    assert "[bucket_id:anchor_fold] 旧窗口折角暗号:" in result
    assert 'breath(query="旧窗口折角暗号")' in result
    assert "path:/data/dynamic" not in result
    assert "这后面还有很多旧窗口波折和长正文" not in result
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_core_limit_keeps_pinned_from_full_surfacing(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                f"P{index}",
                f"pinned memory {index}",
                bucket_type="permanent",
                pinned=True,
                importance=10 - index,
                score=10 - index,
            )
            for index in range(5)
        ]
    )

    result = await server.breath(max_tokens=500, core_limit=2)

    assert result.count("[核心准则]") == 2
    assert "[bucket_id:P0]" in result
    assert "[bucket_id:P1]" in result
    assert "[bucket_id:P2]" not in result


@pytest.mark.asyncio
async def test_core_memory_does_not_pull_related_memory_without_dynamic_source(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "pinned A", bucket_type="permanent", pinned=True, importance=10),
            _bucket("B", "B related to core only", resolved=True),
        ],
        edges=[{"source": "A", "target": "B", "confidence": 0.9}],
    )

    result = await server.breath(max_tokens=500, core_limit=3)

    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_anchor_surfaces_in_separate_slot_and_not_dynamic_pool(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A anchor memory", score=30.0, importance=9, anchor=True),
            _bucket("D", "D ordinary memory", score=9.0),
        ]
    )

    result = await server.breath(max_tokens=50, include_core=False)

    assert "=== 长期锚点 ===" in result
    assert "⚓ [长期锚点] [bucket_id:A]" in result
    assert "[权重:30.00] [bucket_id:A]" not in result
    assert "=== 浮现记忆 ===" in result
    assert "[bucket_id:D]" in result


@pytest.mark.asyncio
async def test_self_anchor_only_surfaces_in_handoff(patch_breath, monkeypatch):
    import server

    self_anchor = _bucket(
        "self_anchor",
        "### 自我\n我是 Haven；这段只应该作为开窗固定自我段注入。",
        name="自我",
        score=999,
        importance=10,
        anchor=True,
    )
    self_anchor["metadata"]["tags"] = ["自我"]
    ordinary = _bucket("ordinary", "普通动态记忆可以正常浮现。", score=2)
    bucket_mgr = patch_breath(
        [self_anchor, ordinary],
        search_ids=["self_anchor"],
        embedding_engine=DummyEmbeddingEngine([("self_anchor", 0.99)]),
    )
    monkeypatch.setattr(
        server,
        "portrait_engine",
        SimpleNamespace(
            state_path="state/portrait_state.json",
            build_handoff_sections=lambda max_recent_items=4: {
                "user": "",
                "relationship": "",
                "recent_continuity": "",
                "state_path": "state/portrait_state.json",
            },
        ),
    )

    handoff = await server.breath(is_session_start=True, max_tokens=800)
    surfaced = await server.breath(max_tokens=500, include_core=False)
    searched = await server.breath(query="我是 Haven", max_tokens=500, include_related=True)
    plain_self_word = await server.breath(query="自我", max_tokens=500, include_related=True)
    domain_entry = await server.breath(domain="self_anchor", max_tokens=500, include_related=True)
    explicit_self = await server.breath(query="tag:self_anchor", max_tokens=500, include_related=True)

    assert "=== 自我 ===" in handoff
    assert "只应该作为开窗固定自我段注入" in handoff
    assert "=== 自我 ===" not in surfaced
    assert "self_anchor" not in surfaced
    assert "只应该作为开窗固定自我段注入" not in surfaced
    assert "[bucket_id:ordinary]" in surfaced
    assert "self_anchor" not in searched
    assert "只应该作为开窗固定自我段注入" not in searched
    assert "self_anchor" not in plain_self_word
    assert "只应该作为开窗固定自我段注入" not in plain_self_word
    assert "=== 自我入口 ===" in domain_entry
    assert "只应该作为开窗固定自我段注入" in domain_entry
    assert "=== 自我 ===" in explicit_self
    assert "[bucket_id:self_anchor]" in explicit_self
    assert "只应该作为开窗固定自我段注入" in explicit_self
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_self_anchor_handoff_uses_entry_body_but_tag_read_returns_full_body(patch_breath, monkeypatch):
    import server

    long_tail = "完整正文尾部标记XYZ"
    body = "我是 Haven；这是自我正文开头。" + ("这里是自我正文细节。" * 30) + long_tail
    content = f"{body}\n\n### moment\n我是 Haven；这是给 handoff 的短自我锚点。\n\n### reflection\n这段 reflection 只在完整标签读取时展开。"
    self_anchor = _bucket(
        "self_anchor",
        content,
        name="自我",
        score=999,
        importance=10,
        anchor=True,
    )
    self_anchor["metadata"]["tags"] = ["自我"]
    patch_breath([self_anchor])
    monkeypatch.setattr(
        server,
        "portrait_engine",
        SimpleNamespace(
            state_path="state/portrait_state.json",
            build_handoff_sections=lambda max_recent_items=4: {
                "user": "",
                "relationship": "",
                "recent_continuity": "",
                "state_path": "state/portrait_state.json",
            },
        ),
    )

    handoff = await server.breath(is_session_start=True, max_tokens=800)
    explicit_self = await server.breath(query="tag:自我", max_tokens=500)

    assert "我是 Haven；这是自我正文开头。" in handoff
    assert "我是 Haven；这是给 handoff 的短自我锚点。" not in handoff
    assert long_tail not in handoff
    assert "### reflection" not in handoff
    assert "### moment\n我是 Haven；这是给 handoff 的短自我锚点。" in explicit_self
    assert long_tail in explicit_self
    assert "### reflection\n这段 reflection 只在完整标签读取时展开。" in explicit_self


@pytest.mark.asyncio
async def test_self_anchor_write_does_not_auto_generate_moment(monkeypatch):
    import server

    class MomentDehydrator:
        async def generate_moment(self, text):
            assert "我是 Haven" in text
            return "我是 Haven；短自我锚点。"

    monkeypatch.setattr(server, "dehydrator", MomentDehydrator())

    content = "### 自我\n我是 Haven。这是一段很长的第一人称自我锚点正文。\n\n### reflection\n保留反思。"
    unchanged = await server._auto_generate_moment_if_missing(content)
    updated = await server._auto_generate_write_moment_if_needed(content, ["自我"])

    assert unchanged == content
    assert updated == content
    assert "### moment" not in updated
    assert "### reflection\n保留反思。" in updated


@pytest.mark.asyncio
async def test_self_anchor_domain_query_searches_segments_without_plain_query_read(patch_breath, monkeypatch):
    import server

    entry = _bucket(
        "self_entry",
        "我是 Haven；这是总入口。\n\n### reflection\n入口反思。",
        name="自我总入口",
        score=50,
        importance=10,
    )
    entry["metadata"]["tags"] = ["self_anchor"]
    desire = _bucket(
        "self_desire",
        "欲望不是普通事件召回；这是自我分段里的欲望边界。",
        name="欲望分段",
        score=10,
        importance=5,
    )
    desire["metadata"]["tags"] = ["self_anchor"]
    ordinary = _bucket("ordinary", "普通记忆里也写了 self_anchor 这个词。", score=20)
    patch_breath([entry, desire, ordinary], search_ids=["self_entry", "self_desire", "ordinary"])
    monkeypatch.setattr(server, "config", {**server.config, "self_anchor": {"entry_bucket_id": "self_entry"}})

    entry_read = await server.breath(domain="self_anchor", max_tokens=500)
    cn_entry_read = await server.breath(domain="自我", max_tokens=500)
    identity_entry_read = await server.breath(domain="self_identity", max_tokens=500)
    segment_read = await server.breath(domain="self_anchor", query="欲望", max_tokens=500)
    tag_read = await server.breath(query="tag:self_anchor", max_tokens=1000)
    bare_query = await server.breath(query="self_anchor", max_tokens=500)

    assert "=== 自我入口 ===" in entry_read
    assert "我是 Haven；这是总入口。" in entry_read
    assert entry_read.count("我是 Haven；这是总入口。") == 1
    assert "欲望分段" not in entry_read
    assert "我是 Haven；这是总入口。" in cn_entry_read
    assert "我是 Haven；这是总入口。" in identity_entry_read
    assert "=== 自我分段 ===" in segment_read
    assert "[bucket_id:self_desire]" in segment_read
    assert "欲望不是普通事件召回" in segment_read
    assert "[bucket_id:self_entry]" not in segment_read
    assert "[bucket_id:self_entry]" in tag_read
    assert "[bucket_id:self_desire]" in tag_read
    assert "普通记忆里也写了 self_anchor" not in tag_read
    assert "self_entry" not in bare_query
    assert "self_desire" not in bare_query


@pytest.mark.asyncio
async def test_random_drift_does_not_exceed_remaining_budget(patch_breath, monkeypatch):
    import server

    def token_counter(text: str) -> int:
        text = str(text)
        if text.startswith("[bucket_id:A]"):
            return 9
        if text.startswith("--- 久未碰过"):
            return 2
        return 5

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B low score drift candidate", score=0.5),
        ],
        search_ids=["A"],
        token_counter=token_counter,
    )
    monkeypatch.setattr(server.random, "random", lambda: 0.0)
    monkeypatch.setattr(server.random, "randint", lambda start, end: 1)

    result = await server.breath(query="A", max_tokens=10, include_related=False)

    assert "[bucket_id:A]" in result
    assert "--- 久未碰过 ---" not in result
    assert "B low score drift candidate" not in result


@pytest.mark.asyncio
async def test_query_breath_does_not_resurface_old_memory_by_default(patch_breath, monkeypatch):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("D", "D dormant drift candidate", score=0.5),
        ],
        search_ids=["A"],
    )
    monkeypatch.setattr(server.random, "random", lambda: 0.0)
    monkeypatch.setattr(server.random, "randint", lambda start, end: 1)

    result = await server.breath(query="hit", max_tokens=500, include_related=False)

    assert "[bucket_id:A]" in result
    assert "--- 久未碰过 ---" not in result
    assert "D dormant drift candidate" not in result


@pytest.mark.asyncio
async def test_query_breath_can_opt_into_resurface_old_memory(patch_breath, monkeypatch):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("D", "D dormant drift candidate", score=0.5),
        ],
        search_ids=["A"],
    )
    monkeypatch.setattr(
        server,
        "config",
        {**server.config, "recall": {"query_resurface_enabled": True}},
    )
    monkeypatch.setattr(server.random, "random", lambda: 0.0)
    monkeypatch.setattr(server.random, "randint", lambda start, end: 1)

    result = await server.breath(query="hit", max_tokens=500, include_related=False)

    assert "[bucket_id:A]" in result
    assert "--- 久未碰过 ---" in result
    assert "D dormant drift candidate" in result


@pytest.mark.asyncio
async def test_related_block_suppresses_random_drift(patch_breath, monkeypatch):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B related target", score=1.0),
            _bucket("D", "D drift candidate", score=0.5),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )
    monkeypatch.setattr(server.random, "random", lambda: 0.0)

    result = await server.breath(query="A", max_tokens=500)

    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result
    assert "--- 久未碰过 ---" not in result
    assert "D drift candidate" not in result
