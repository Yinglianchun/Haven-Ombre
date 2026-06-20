from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from bucket_manager import BucketManager
from gateway import GatewayService
from gateway_state import GatewayStateStore
from memory_edges import MemoryEdgeStore


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_memory_recall.yaml"


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        compact = " ".join(str(content or "").split())
        return f"{name}: {compact[:80]}"


class JsonSummaryDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        return json.dumps(
            {
                "core_facts": [f"{name} fact one", f"{name} fact two"],
                "todos": ["do not inject"],
                "keywords": ["json", "noise"],
                "summary": f"{name} compact summary",
            },
            ensure_ascii=False,
        )


class DummyEmbeddingEngine:
    enabled = True

    def __init__(self, results: list[tuple[str, float]] | None = None):
        self.results = results or []

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.results[:top_k]


class DummyPersonaEngine:
    enabled = True
    profile_id = "haven_xiaoyu"
    mode = "test"
    model = "dummy"
    api_key = ""

    async def build_pre_reply_guidance(self, session_id: str, latest_user_message: str = "") -> dict:
        return self.get_current_state(session_id)

    async def update_from_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        recalled_memory_ids: list[str] | None = None,
        tool_summary: str = "",
        recent_conversation_turns: list[dict] | None = None,
    ) -> dict:
        return self.get_current_state(session_id)

    def get_current_state(self, session_id: str) -> dict:
        return {"personality": {}, "affect": {}, "relationship": {}, "reply_guidance": ""}

    def format_state_block(self, state: dict) -> str:
        return "Long-term State Summary"


def _load_cases() -> list[dict[str, Any]]:
    data = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    return list(data.get("cases", []))


def _case_params() -> list[pytest.ParameterSet]:
    params = []
    for case in _load_cases():
        marks = []
        if case.get("xfail"):
            marks.append(pytest.mark.xfail(reason=str(case["xfail"]), strict=False))
        params.append(pytest.param(case, id=case["id"], marks=marks))
    return params


def _run(coro):
    return asyncio.run(coro)


def _case_config(test_config: dict, case: dict[str, Any]) -> dict:
    cfg = deepcopy(test_config)
    cfg["gateway"] = {**cfg.get("gateway", {}), **case.get("gateway", {})}
    if case.get("memory_diffusion"):
        cfg["memory_diffusion"] = dict(case["memory_diffusion"])
    cfg.setdefault("dehydration", {})["api_key"] = ""
    cfg.setdefault("persona", {})["api_key"] = ""
    return cfg


def _create_buckets(bucket_mgr: BucketManager, case: dict[str, Any]) -> dict[str, str]:
    ids = {}
    for bucket in case.get("buckets", []):
        timestamp = (
            datetime.now() - timedelta(hours=float(bucket.get("hours_ago", 24)))
        ).isoformat(timespec="seconds")
        bucket_id = _run(
            bucket_mgr.create(
                content=str(bucket["content"]),
                tags=list(bucket.get("tags", [])),
                importance=int(bucket.get("importance", 5)),
                domain=list(bucket.get("domain", ["未分类"])),
                valence=float(bucket.get("valence", 0.7)),
                arousal=float(bucket.get("arousal", 0.4)),
                bucket_type=str(bucket.get("bucket_type", "dynamic")),
                name=str(bucket.get("name") or bucket["key"]),
                pinned=bool(bucket.get("pinned", False)),
                protected=bool(bucket.get("protected", False)),
                created=timestamp,
                last_active=timestamp,
                updated_at=timestamp,
                resolved=bool(bucket.get("resolved", False)),
                digested=bool(bucket.get("digested", False)),
            )
        )
        ids[str(bucket["key"])] = bucket_id
    return ids


def _embedding_results(case: dict[str, Any], bucket_ids: dict[str, str]) -> list[tuple[str, float]]:
    results = []
    for key, score in case.get("embedding_results", []):
        results.append((bucket_ids[str(key)], float(score)))
    return results


def _create_edges(cfg: dict, case: dict[str, Any], bucket_ids: dict[str, str]) -> None:
    edge_store = MemoryEdgeStore(cfg)
    for edge in case.get("edges", []):
        edge_store.add_edge(
            bucket_ids[str(edge["source"])],
            bucket_ids[str(edge["target"])],
            str(edge.get("relation_type", "relates_to")),
            float(edge.get("confidence", 0.5)),
            str(edge.get("reason", "")),
        )


def _dehydrator(case: dict[str, Any]):
    if case.get("dehydrator") == "json_summary":
        return JsonSummaryDehydrator()
    return DummyDehydrator()


def _build_service(cfg: dict, bucket_mgr: BucketManager, case: dict[str, Any], bucket_ids: dict[str, str]) -> GatewayService:
    return GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=_dehydrator(case),
        embedding_engine=DummyEmbeddingEngine(_embedding_results(case, bucket_ids)),
        state_store=GatewayStateStore(str(Path(cfg["state_dir"]) / "gateway_state.db")),
        persona_engine=DummyPersonaEngine(),
    )


def _run_gateway_recall_case(cfg: dict, bucket_mgr: BucketManager, case: dict[str, Any], bucket_ids: dict[str, str]) -> dict[str, str]:
    service = _build_service(cfg, bucket_mgr, case, bucket_ids)
    session_id = str(case.get("session_id") or case["id"])
    query = str(case["query"])
    all_buckets = _run(bucket_mgr.list_all(include_archive=False))
    all_moments, grouped_moments, moment_edges = service._refresh_moment_graph(all_buckets)
    recalled_moments, moment_candidates, _suppressed_moments, _suppressed_buckets = _run(
        service._select_dynamic_moments(query, session_id, all_buckets, grouped_moments)
    )
    recalled_memory = _run(
        service._format_recalled_moments(
            recalled_moments,
            grouped_moments,
            all_buckets,
            service.recalled_budget,
            query_text=query,
        )
    )
    diffused_memory = service._build_moment_diffused_memory_block(
        recalled_moments,
        moment_candidates,
        all_moments,
        moment_edges,
        query,
    )
    _, dynamic_context = service._build_injected_context_messages(
        persona_block="",
        core_memory="",
        portrait_memory="",
        recent_context="",
        recalled_memory=recalled_memory,
        relationship_weather="",
        favorite_memory="",
        related_memory=diffused_memory,
        dream_context="",
    )
    return {
        "dynamic_context": dynamic_context,
        "recalled_memory": recalled_memory,
        "diffused_memory": diffused_memory,
    }


def _run_diffused_block_case(cfg: dict, bucket_mgr: BucketManager, case: dict[str, Any], bucket_ids: dict[str, str]) -> dict[str, str]:
    service = _build_service(cfg, bucket_mgr, case, bucket_ids)
    all_buckets = _run(bucket_mgr.list_all(include_archive=False))
    recalled = [
        _run(bucket_mgr.get(bucket_ids[str(key)]))
        for key in case.get("seed_bucket_keys", [])
    ]
    block = _run(service._build_diffused_memory_block(recalled, all_buckets, str(case.get("query") or "")))
    _, dynamic_context = service._build_injected_context_messages(
        persona_block="",
        core_memory="",
        portrait_memory="",
        recent_context="",
        recalled_memory="",
        relationship_weather="",
        favorite_memory="",
        related_memory=block,
        dream_context="",
    )
    return {"dynamic_context": dynamic_context, "recalled_memory": "", "diffused_memory": block}


def _assert_expected(case: dict[str, Any], outputs: dict[str, str]) -> None:
    expected = case.get("expected", {})
    section_title = expected.get("section_title")
    if section_title:
        assert section_title in outputs["dynamic_context"], outputs["dynamic_context"]

    target_name = str(expected.get("target") or "dynamic_context")
    target = outputs[target_name]
    assert target.strip(), f"{target_name} was empty for {case['id']}"

    for needle in expected.get("must_include", []):
        assert str(needle) in target, f"{needle!r} missing from {target_name}:\n{target}"
    for needle in expected.get("must_not_include", []):
        assert str(needle) not in target, f"{needle!r} leaked into {target_name}:\n{target}"


@pytest.mark.parametrize("case", _case_params())
def test_memory_recall_golden(case, test_config):
    cfg = _case_config(test_config, case)
    bucket_mgr = BucketManager(cfg)
    bucket_ids = _create_buckets(bucket_mgr, case)
    _create_edges(cfg, case, bucket_ids)

    if case["mode"] == "gateway_recall":
        outputs = _run_gateway_recall_case(cfg, bucket_mgr, case, bucket_ids)
    elif case["mode"] == "gateway_diffused_block":
        outputs = _run_diffused_block_case(cfg, bucket_mgr, case, bucket_ids)
    else:
        raise AssertionError(f"unknown golden mode: {case['mode']}")

    _assert_expected(case, outputs)
