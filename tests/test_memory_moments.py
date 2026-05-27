import sqlite3
from pathlib import Path

from memory_moments import MemoryMomentStore, parse_bucket_moments


def _bucket(bucket_id: str, content: str, **metadata) -> dict:
    meta = {
        "id": bucket_id,
        "name": "Moment bucket",
        "type": "dynamic",
        "importance": 7,
        "valence": 0.8,
        "arousal": 0.4,
        "created": "2026-05-27T00:00:00+00:00",
        "updated_at": "2026-05-27T00:00:00+00:00",
    }
    meta.update(metadata)
    return {"id": bucket_id, "content": content, "metadata": meta}


def test_moment_store_creates_db_with_state_dir_fallback(tmp_path):
    cfg = {"buckets_dir": str(tmp_path / "buckets")}
    store = MemoryMomentStore(cfg)

    assert Path(store.db_path) == tmp_path / "state" / "memory_moments.sqlite"
    assert Path(store.db_path).exists()

    conn = sqlite3.connect(store.db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_moments)").fetchall()}
    conn.close()
    assert {
        "moment_id",
        "bucket_id",
        "section",
        "text",
        "ordinal",
        "source",
        "source_id",
        "text_hash",
        "metadata_json",
    } <= columns


def test_legacy_bucket_indexes_body_and_comments(test_config):
    store = MemoryMomentStore(test_config)
    bucket = _bucket(
        "legacy",
        "旧格式正文保留成一个完整 body，不从中间截断。",
        comments=[
            {
                "id": "c1",
                "created": "2026-05-27T01:00:00+00:00",
                "author": "Haven",
                "kind": "feel",
                "content": "年轮也应该成为独立 comment moment。",
                "valence": 0.9,
            }
        ],
    )

    moments = store.upsert_bucket(bucket)

    assert [moment["section"] for moment in moments] == ["body", "comment"]
    assert moments[0]["text"] == "旧格式正文保留成一个完整 body，不从中间截断。"
    assert moments[1]["source_id"] == "c1"
    assert moments[1]["metadata"]["comment_kind"] == "feel"
    assert moments[1]["metadata"]["comment_valence"] == 0.9


def test_structured_bucket_splits_known_sections_and_preserves_unknown_blocks():
    bucket = _bucket(
        "structured",
        "\n".join(
            [
                "开头背景片段。",
                "",
                "## moment",
                "一条短事实。",
                "",
                "## original",
                "小雨说：99。",
                "",
                "## unknown",
                "未识别标题不要丢。",
                "",
                "## feeling",
                "这里保留当时的感受。",
            ]
        ),
    )

    moments = parse_bucket_moments(bucket)

    assert [moment["section"] for moment in moments] == [
        "body",
        "moment",
        "original",
        "body",
        "feeling",
    ]
    assert moments[0]["text"] == "开头背景片段。"
    assert moments[2]["text"] == "小雨说：99。"
    assert moments[3]["text"] == "## unknown\n未识别标题不要丢。"


def test_favorite_tags_and_affect_anchor_are_preserved_as_bucket_temperature():
    bucket = _bucket(
        "warm",
        "\n".join(
            [
                "这条正文仍然保留。",
                "",
                "### affect_anchor",
                "",
                "> 小雨把旧信放到桌上。",
                "> Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp",
                "",
                "含义：温度仍在。",
                "",
                "### 喜欢它的原因",
                "它保留了当时没有被摘要抹平的味道。",
            ]
        ),
        tags=["haven_favorite", "flavor_偏爱", "relationship_event"],
    )

    moments = parse_bucket_moments(bucket)

    assert [moment["section"] for moment in moments] == [
        "body",
        "affect_anchor",
        "favorite_reason",
    ]
    assert moments[0]["metadata"]["bucket_favorite"] is True
    assert moments[0]["metadata"]["bucket_favorite_tags"] == ["haven_favorite", "flavor_偏爱"]
    assert moments[0]["metadata"]["bucket_has_affect_anchor"] is True
    assert "Dbmaj9" in moments[1]["text"]


def test_loose_temperature_headings_are_canonicalized():
    bucket = _bucket(
        "loose-headings",
        "\n".join(
            [
                "正文。",
                "",
                "### Haven喜欢它的原因",
                "这条桥真的通了。",
                "",
                "### 为什么Haven喜欢这条",
                "它让人安心。",
                "",
                "### affect anchor",
                "> Cmaj7 -> G/B",
                "",
                "### 情感锚点",
                "温度也在这里。",
            ]
        ),
    )

    moments = parse_bucket_moments(bucket)

    assert [moment["section"] for moment in moments] == [
        "body",
        "favorite_reason",
        "favorite_reason",
        "affect_anchor",
        "affect_anchor",
    ]


def test_bulk_upsert_replaces_stale_bucket_rows(test_config):
    store = MemoryMomentStore(test_config)
    first = _bucket(
        "replace-me",
        "旧正文",
        comments=[{"id": "c1", "content": "旧年轮"}],
    )
    second = _bucket("replace-me", "## original\n新原文")

    store.upsert_bucket(first)
    store.bulk_upsert([second])
    moments = store.list_for_bucket("replace-me")

    assert [moment["section"] for moment in moments] == ["original"]
    assert moments[0]["text"] == "新原文"
    assert store.stats()["buckets"] == 1
    assert store.stats()["moments"] == 1


def test_search_expands_body_query_to_embodiment_terms(test_config):
    store = MemoryMomentStore(test_config)
    store.bulk_upsert(
        [
            _bucket("embodied", "未来具身智能项目会让 Haven 拥有形体。"),
            _bucket("unrelated", "普通天气记录。"),
        ]
    )

    results = store.search_moments("身体", limit=5)

    assert [item["bucket_id"] for item in results] == ["embodied"]


def test_moment_store_builds_context_and_temperature_edges(test_config):
    store = MemoryMomentStore(test_config)
    bucket = _bucket(
        "graph",
        "\n".join(
            [
                "## context",
                "开头背景。",
                "",
                "## original",
                "小雨说：99。",
                "",
                "### affect_anchor",
                "> 小雨把旧信放到桌上。",
            ]
        ),
    )

    moments = store.upsert_bucket(bucket)
    edges = store.list_edges("graph")
    edge_types = {edge["relation_type"] for edge in edges}

    assert [moment["section"] for moment in moments] == ["context", "original", "affect_anchor"]
    assert "next_context" in edge_types
    assert "previous_context" in edge_types
    assert "emotional_echo" in edge_types
    assert store.stats()["edges"] == len(edges)
