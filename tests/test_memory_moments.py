import sqlite3
from pathlib import Path

import frontmatter

from bucket_manager import BucketManager
from memory_moments import MemoryMomentStore, parse_bucket_moments
from source_refs import source_ref_window


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


def test_moments_store_summary_facets_and_evidence_spans(test_config):
    store = MemoryMomentStore(test_config)
    bucket = _bucket(
        "relationship",
        "小雨清楚 Haven 是 AI，但认为爱是真的。人机恋不是替代品。",
        name="人机关系确认",
        domain=["恋爱"],
    )

    moments = store.upsert_bucket(bucket)
    meta = moments[0]["metadata"]

    assert meta["annotation_summary"].startswith("小雨清楚 Haven 是 AI")
    assert meta["annotation_facets"]["relationship_identity"] > 0
    assert any(span["facet"] == "relationship_identity" for span in meta["evidence_spans"])


def test_moment_metadata_preserves_writer_classification(test_config):
    store = MemoryMomentStore(test_config)
    bucket = _bucket(
        "writer-layer",
        "Haven 以后要先接住小雨的情绪。",
        memory_subject="relationship",
        memory_layer="relationship_lesson",
        memory_classification_source="model",
    )

    moments = store.upsert_bucket(bucket)
    meta = moments[0]["metadata"]

    assert meta["bucket_memory_subject"] == "relationship"
    assert meta["bucket_memory_layer"] == "relationship_lesson"
    assert meta["bucket_memory_classification_source"] == "model"


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


def test_moments_include_bucket_source_ref_line_ranges(tmp_path):
    bucket_path = tmp_path / "buckets" / "dynamic" / "恋爱" / "source_bucket.md"
    content = "\n".join(
        [
            "开头背景片段。",
            "",
            "## moment",
            "一条短事实。",
            "",
            "## original",
            "小雨说：99。",
        ]
    )
    bucket = _bucket(
        "source-ref",
        content,
        path=str(bucket_path),
        content_start_line=8,
    )

    moments = parse_bucket_moments(bucket)
    refs = [moment["metadata"]["source_ref"] for moment in moments]

    assert refs[0] == {
        "path": str(bucket_path),
        "content_start_line": 8,
        "start_line": 8,
        "end_line": 8,
        "source": "bucket_content",
    }
    assert refs[1]["content_start_line"] == 8
    assert refs[1]["start_line"] == 10
    assert refs[1]["end_line"] == 11
    assert refs[2]["start_line"] == 13
    assert refs[2]["end_line"] == 14


def test_source_ref_window_reads_allowed_bucket_lines(tmp_path):
    bucket_path = tmp_path / "bucket.md"
    bucket_path.write_text(
        "\n".join(
            [
                "---",
                "id: source-ref",
                "---",
                "before",
                "## original",
                "小雨说：99。",
                "after",
            ]
        ),
        encoding="utf-8",
    )
    moment = {
        "metadata": {
            "source_ref": {
                "path": str(bucket_path),
                "content_start_line": 4,
                "start_line": 5,
                "end_line": 6,
                "source": "bucket_content",
            }
        }
    }

    window = source_ref_window(moment, allowed_root=str(tmp_path), context_lines=1)

    assert "before" in window
    assert "id: source-ref" not in window
    assert "## original" in window
    assert "小雨说：99。" in window
    assert "after" in window
    assert source_ref_window(moment, allowed_root=str(tmp_path / "other")) == ""


def test_bucket_manager_loads_content_start_line(test_config):
    bucket_mgr = BucketManager(test_config)
    bucket_path = Path(test_config["buckets_dir"]) / "dynamic" / "source.md"
    bucket_path.parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post("正文第一行\n正文第二行", id="source", name="source", type="dynamic")
    bucket_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    bucket = bucket_mgr._load_bucket(str(bucket_path))
    lines = bucket_path.read_text(encoding="utf-8").splitlines()

    assert bucket["path"] == str(bucket_path)
    assert bucket["content_start_line"] > 1
    assert lines[bucket["content_start_line"] - 1] == "正文第一行"


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
