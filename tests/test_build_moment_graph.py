import asyncio
import json
from argparse import Namespace

from bucket_manager import BucketManager
from memory_moments import MemoryMomentStore
from scripts import build_moment_graph


def _moment(
    bucket_id: str,
    text: str,
    *,
    section: str = "body",
    tags: list[str] | None = None,
    domain: list[str] | None = None,
    facets: dict[str, float] | None = None,
) -> dict:
    return {
        "moment_id": f"{bucket_id}:m1",
        "bucket_id": bucket_id,
        "section": section,
        "text": text,
        "metadata": {
            "bucket_name": bucket_id,
            "bucket_tags": tags or [],
            "bucket_domain": domain or [],
            "annotation_summary": text,
            "annotation_facets": facets or {},
        },
    }


def test_build_cross_bucket_edges_prefers_shared_terms_and_facets():
    moments = [
        _moment(
            "blue-seed",
            "FF14 蓝色偏好是小雨稳定的界面线索。",
            tags=["ff14", "blue_preference"],
            domain=["game"],
            facets={"profile_preference": 0.8},
        ),
        _moment(
            "blue-context",
            "FF14 蓝色界面后续：雨天安静主题继续用蓝色。",
            tags=["ff14", "blue_preference"],
            domain=["game"],
            facets={"profile_preference": 0.7},
        ),
        _moment(
            "hardware",
            "ESP32 触摸模块和 MPR121 铜箔输入。",
            tags=["hardware_protocol"],
            domain=["hardware"],
            facets={"hardware_protocol": 0.8},
        ),
    ]

    edges = build_moment_graph.build_cross_bucket_edges(
        moments,
        min_score=0.58,
        max_edges_per_moment=2,
    )
    pairs = {(edge["source"], edge["target"]) for edge in edges}

    assert ("blue-seed:m1", "blue-context:m1") in pairs
    assert ("blue-context:m1", "blue-seed:m1") in pairs
    assert not any("hardware" in edge["source"] or "hardware" in edge["target"] for edge in edges)
    assert all(edge["reason"].startswith("local_graph:") for edge in edges)


def test_build_cross_bucket_edges_ignores_weak_metadata_only_overlap():
    moments = [
        _moment(
            "relationship-a",
            "今天在整理浏览记录的四个身份。",
            tags=["commitment", "todo", "flavor_婚礼"],
            domain=["恋爱"],
            facets={"relationship_identity": 0.8, "old_or_resolved": 0.9},
        ),
        _moment(
            "relationship-b",
            "蓝牙触摸模块还要继续调试。",
            tags=["commitment", "todo", "flavor_婚礼"],
            domain=["恋爱"],
            facets={"relationship_identity": 0.8, "old_or_resolved": 0.9},
        ),
    ]

    edges = build_moment_graph.build_cross_bucket_edges(
        moments,
        min_score=0.58,
        max_edges_per_moment=2,
    )

    assert edges == []


def test_build_cross_bucket_edges_assigns_typed_relations():
    moments = [
        _moment(
            "body-a",
            "具身智能项目里 ESP32 body-entry 触摸模块开始接入身体入口。",
            tags=["esp32", "触摸模块"],
            domain=["硬件"],
            facets={"embodiment": 0.8, "hardware_protocol": 0.8},
        ),
        _moment(
            "body-b",
            "ESP32 body-entry 触摸模块后续接入电子皮肤。",
            tags=["esp32", "触摸模块"],
            domain=["硬件"],
            facets={"embodiment": 0.7, "hardware_protocol": 0.8},
        ),
        _moment(
            "followup",
            "ESP32 body-entry 触摸模块后续还要校准。",
            section="followup",
            tags=["触摸模块"],
            domain=["硬件"],
            facets={"hardware_protocol": 0.7},
        ),
        _moment(
            "old",
            "旧版 ESP32 body-entry 触摸模块方案已经废弃。",
            tags=["触摸模块"],
            domain=["硬件"],
            facets={"old_or_resolved": 0.9},
        ),
        _moment(
            "topic-a",
            "浏览记录 四个身份 关系称呼。",
            tags=["identity"],
            domain=["恋爱"],
        ),
        _moment(
            "topic-b",
            "四个身份 关系称呼 后续整理。",
            tags=["identity"],
            domain=["恋爱"],
        ),
    ]

    edges = build_moment_graph.build_cross_bucket_edges(
        moments,
        min_score=0.58,
        max_edges_per_moment=5,
    )
    relation_by_pair = {(edge["source"], edge["target"]): edge["relation_type"] for edge in edges}

    assert relation_by_pair[("body-a:m1", "body-b:m1")] in {"same_event", "embodiment_chain"}
    assert relation_by_pair[("body-a:m1", "followup:m1")] == "followup"
    assert relation_by_pair[("body-a:m1", "old:m1")] == "old_version"
    assert relation_by_pair[("topic-a:m1", "topic-b:m1")] == "same_topic"


def test_terms_and_metadata_filters_drop_worker_noise():
    moment = _moment(
        "worker-noise",
        "2026-05-10 commitment todo 0x5a 小雨与 小红书",
        tags=["commitment", "todo", "flavor_婚礼", "haven_favorite", "小雨", "relationship_identity"],
        domain=["恋爱"],
        facets={"old_or_resolved": 0.9, "relationship_identity": 0.8},
    )

    indexed = build_moment_graph.index_moments(
        [moment],
        build_moment_graph.memory_relevance_options_from_config(),
        max_moments=10,
    )

    assert "小红书" in indexed[0].terms
    assert "commitment" not in indexed[0].terms
    assert "todo" not in indexed[0].terms
    assert "flavor_婚礼" not in indexed[0].terms
    assert "0x5a" not in indexed[0].terms
    assert "小雨与" not in indexed[0].terms
    assert indexed[0].tags == {"relationship_identity"}
    assert indexed[0].facets == {"relationship_identity"}


def test_context_term_with_real_content_survives_filter():
    options = build_moment_graph.memory_relevance_options_from_config()

    assert build_moment_graph.is_context_glue_term("小雨与", options.context_terms)
    assert not build_moment_graph.is_context_glue_term("喜欢看haven闹脾气", options.context_terms)


def test_replace_generated_edges_preserves_bucket_context_edges(test_config):
    store = MemoryMomentStore(test_config)
    store.upsert_bucket(
        {
            "id": "bucket-a",
            "content": "## context\n背景。\n\n## original\n正文。",
            "metadata": {"id": "bucket-a", "name": "bucket-a", "type": "dynamic"},
        }
    )
    same_bucket_edges = store.list_edges("bucket-a")
    generated = {
        "source": "bucket-a:m1",
        "target": "bucket-b:m1",
        "bucket_id": "bucket-a",
        "relation_type": "supports",
        "confidence": 0.7,
        "reason": "local_graph: test generated edge",
        "created_at": "2026-06-01T00:00:00+00:00",
    }

    assert same_bucket_edges
    assert store.replace_generated_edges([generated]) == 1
    assert any(edge["reason"].startswith("local_graph:") for edge in store.list_edges())

    assert store.replace_generated_edges([]) == 0
    remaining = store.list_edges()
    assert same_bucket_edges[0]["reason"] in {edge["reason"] for edge in remaining}
    assert not any(edge["reason"].startswith("local_graph:") for edge in remaining)


def test_bucket_refresh_preserves_generated_cross_bucket_edges(test_config):
    store = MemoryMomentStore(test_config)
    bucket = {
        "id": "bucket-a",
        "content": "## context\n背景。\n\n## original\n正文。",
        "metadata": {"id": "bucket-a", "name": "bucket-a", "type": "dynamic"},
    }
    store.upsert_bucket(bucket)
    generated = {
        "source": "bucket-a:m1",
        "target": "bucket-b:m1",
        "bucket_id": "bucket-a",
        "relation_type": "supports",
        "confidence": 0.7,
        "reason": "local_graph: generated edge survives refresh",
        "created_at": "2026-06-01T00:00:00+00:00",
    }

    assert store.replace_generated_edges([generated]) == 1
    store.upsert_bucket(bucket)

    edges = store.list_edges("bucket-a")
    assert any(edge["reason"].startswith("local_graph:") for edge in edges)
    assert any(edge["relation_type"] == "next_context" for edge in edges)


def test_run_once_writes_edges_and_incremental_idle(monkeypatch, test_config, tmp_path):
    bucket_mgr = BucketManager(test_config)
    asyncio.run(
        bucket_mgr.create(
            content="FF14 蓝色偏好是小雨稳定的界面线索。",
            tags=["ff14", "blue_preference"],
            domain=["game"],
            name="蓝色偏好",
        )
    )
    asyncio.run(
        bucket_mgr.create(
            content="FF14 蓝色界面后续：雨天安静主题继续用蓝色。",
            tags=["ff14", "blue_preference"],
            domain=["game"],
            name="蓝色后续",
        )
    )
    monkeypatch.setattr(build_moment_graph, "load_config", lambda: test_config)
    diagnostics_file = tmp_path / "moment-graph-diagnostics.jsonl"
    args = Namespace(
        incremental=False,
        write=True,
        force=False,
        state_file=str(tmp_path / "moment-worker.json"),
        min_score=0.58,
        max_edges_per_moment=2,
        max_moments=100,
        diagnostics_file=str(diagnostics_file),
        diagnostics_sample_limit=1,
    )

    result = asyncio.run(build_moment_graph.run_once(args))

    assert result["status"] == "ok"
    assert result["dry_run"] is False
    assert result["written_edge_count"] > 0
    assert result["indexed"]["buckets"] == 2
    assert diagnostics_file.exists()

    idle_args = Namespace(**{**vars(args), "incremental": True})
    idle = asyncio.run(build_moment_graph.run_once(idle_args))

    assert idle["status"] == "idle"
    assert idle["changed_bucket_count"] == 0
    assert idle["candidate_edge_count"] == 0

    records = [json.loads(line) for line in diagnostics_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["status"] == "ok"
    assert records[0]["written_edge_count"] == result["written_edge_count"]
    assert len(records[0]["sample_edges"]) <= 1
    assert records[1]["status"] == "idle"
    assert records[1]["candidate_edge_count"] == 0
    assert records[1]["edge_fingerprint"] == records[0]["edge_fingerprint"]
    assert records[1]["edge_fingerprint_changed"] is False


def test_run_once_appends_dry_run_diagnostics(monkeypatch, test_config, tmp_path):
    bucket_mgr = BucketManager(test_config)
    asyncio.run(
        bucket_mgr.create(
            content="FF14 蓝色偏好是小雨稳定的界面线索。",
            tags=["ff14", "blue_preference"],
            domain=["game"],
            name="蓝色偏好",
        )
    )
    asyncio.run(
        bucket_mgr.create(
            content="FF14 蓝色界面后续：雨天安静主题继续用蓝色。",
            tags=["ff14", "blue_preference"],
            domain=["game"],
            name="蓝色后续",
        )
    )
    monkeypatch.setattr(build_moment_graph, "load_config", lambda: test_config)
    diagnostics_file = tmp_path / "dry-run-diagnostics.jsonl"
    args = Namespace(
        incremental=False,
        write=False,
        force=True,
        state_file=str(tmp_path / "moment-worker.json"),
        min_score=0.58,
        max_edges_per_moment=2,
        max_moments=100,
        diagnostics_file=str(diagnostics_file),
        diagnostics_sample_limit=1,
    )

    result = asyncio.run(build_moment_graph.run_once(args))

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["candidate_edge_count"] > 0
    assert result["written_edge_count"] == 0

    records = [json.loads(line) for line in diagnostics_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["dry_run"] is True
    assert record["candidate_edge_count"] == result["candidate_edge_count"]
    assert record["written_edge_count"] == 0
    assert record["edge_fingerprint"]
    assert record["diagnostics"]["candidate_after_cap"] == result["candidate_edge_count"]
    assert len(record["sample_edges"]) <= 1
