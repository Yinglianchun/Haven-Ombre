import pytest

from memory_diffusion import (
    DiffusionOptions,
    diffuse_memory,
    diffusion_options_from_config,
    format_diffusion_path,
    format_diffusion_trace,
    path_has_caution,
    path_has_old_version,
)


def _bucket(bucket_id: str, *, name: str | None = None, bucket_type: str = "dynamic") -> dict:
    return {
        "id": bucket_id,
        "content": f"{bucket_id} context",
        "metadata": {
            "name": name or bucket_id,
            "type": bucket_type,
            "importance": 10,
        },
    }


def test_diffusion_walks_multi_hop_path_with_hop_decay():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C", "D", "E"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
        {"source": "B", "target": "C", "relation_type": "triggers", "confidence": 1.0},
        {"source": "C", "target": "D", "relation_type": "triggers", "confidence": 1.0},
        {"source": "D", "target": "E", "relation_type": "triggers", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=4, top_k=10, min_activation=0.0),
    )

    activations = {hit.bucket_id: hit.activation for hit in hits}
    assert activations["B"] == pytest.approx(0.8)
    assert activations["C"] == pytest.approx(0.6)
    assert activations["D"] == pytest.approx(0.4)
    assert activations["E"] == pytest.approx(0.25)
    assert format_diffusion_trace(hits[-1].best_path) == (
        "A --triggers:1.00--> B --triggers:1.00--> C "
        "--triggers:1.00--> D --triggers:1.00--> E"
    )


def test_chain_walk_follows_reliable_edges_beyond_max_hops_until_strength_fails():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C", "D", "E", "F"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "context_of", "confidence": 0.95},
        {"source": "B", "target": "C", "relation_type": "evidenced_by", "confidence": 0.95},
        {"source": "C", "target": "D", "relation_type": "next_context", "confidence": 0.9},
        {"source": "D", "target": "E", "relation_type": "same_event", "confidence": 0.85},
        {"source": "E", "target": "F", "relation_type": "next_context", "confidence": 0.35},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(
            max_hops=2,
            top_k=10,
            chain_walk_enabled=True,
            chain_max_hops=6,
        ),
    )

    hit_ids = {hit.bucket_id for hit in hits}
    assert {"B", "C", "D", "E"}.issubset(hit_ids)
    assert "F" not in hit_ids
    assert len(next(hit for hit in hits if hit.bucket_id == "E").best_path.steps) == 4


def test_chain_walk_does_not_continue_through_generic_relation():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "relates_to", "confidence": 1.0},
        {"source": "B", "target": "C", "relation_type": "context_of", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(
            max_hops=1,
            top_k=10,
            min_activation=0.0,
            chain_walk_enabled=True,
            chain_max_hops=5,
        ),
    )

    assert [hit.bucket_id for hit in hits] == ["B"]


def test_chain_walk_prefers_closer_hit_when_relation_quality_matches():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "context_of", "confidence": 0.9},
        {"source": "B", "target": "C", "relation_type": "context_of", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(
            max_hops=1,
            top_k=2,
            min_activation=0.0,
            chain_walk_enabled=True,
            chain_max_hops=4,
            chain_min_confidence=0.72,
        ),
    )

    assert [hit.bucket_id for hit in hits] == ["B", "C"]


def test_chain_walk_does_not_continue_backwards_through_directional_edges():
    bucket_map = {
        bucket_id: _bucket(bucket_id)
        for bucket_id in ["incident", "reflection", "followup"]
    }
    edges = [
        {"source": "incident", "target": "reflection", "relation_type": "reflects_on", "confidence": 0.95},
        {"source": "incident", "target": "followup", "relation_type": "updates", "confidence": 0.95},
    ]

    hits = diffuse_memory(
        {"reflection": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(
            max_hops=1,
            top_k=10,
            min_activation=0.0,
            chain_walk_enabled=True,
            chain_max_hops=4,
        ),
    )

    assert [hit.bucket_id for hit in hits] == ["incident"]
    assert hits[0].best_path.steps[0].direction == "incoming"


def test_chain_walk_can_continue_backwards_through_same_event_edges():
    bucket_map = {
        bucket_id: _bucket(bucket_id)
        for bucket_id in ["later", "same-event", "followup"]
    }
    edges = [
        {"source": "same-event", "target": "later", "relation_type": "same_event", "confidence": 0.95},
        {"source": "same-event", "target": "followup", "relation_type": "updates", "confidence": 0.95},
    ]

    hits = diffuse_memory(
        {"later": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(
            max_hops=1,
            top_k=10,
            min_activation=0.0,
            chain_walk_enabled=True,
            chain_max_hops=4,
        ),
    )

    assert [hit.bucket_id for hit in hits] == ["same-event", "followup"]
    assert hits[0].best_path.steps[0].direction == "incoming"
    assert hits[1].best_path.steps[-1].direction == "outgoing"


def test_diffusion_config_parses_chain_walk_options():
    options = diffusion_options_from_config(
        {
            "memory_diffusion": {
                "chain_walk_enabled": True,
                "chain_max_hops": 7,
                "chain_min_strength": 0.31,
                "chain_min_confidence": 0.81,
                "chain_min_relation_priority": 62,
                "chain_max_frontier": 12,
                "chain_continue_relation_types": ["context_of", "evidenced_by"],
            }
        }
    )

    assert options.chain_walk_enabled is True
    assert options.chain_max_hops == 7
    assert options.chain_min_strength == pytest.approx(0.31)
    assert options.chain_min_confidence == pytest.approx(0.81)
    assert options.chain_min_relation_priority == 62
    assert options.chain_max_frontier == 12
    assert options.chain_continue_relation_types == ("context_of", "evidenced_by")


def test_diffusion_config_includes_typed_relation_defaults():
    options = diffusion_options_from_config({})

    assert options.relation_type_weights["embodiment_chain"] > options.relation_type_weights["supports"]
    assert options.relation_type_weights["same_topic"] > options.relation_type_weights["relates_to"]
    assert options.relation_type_weights["old_version"] < options.relation_type_weights["supports"]
    assert options.relation_type_weights["conflict"] < options.relation_type_weights["supports"]


def test_diffusion_accumulates_multiple_paths_to_same_node():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C", "D"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
        {"source": "A", "target": "C", "relation_type": "triggers", "confidence": 1.0},
        {"source": "B", "target": "D", "relation_type": "triggers", "confidence": 1.0},
        {"source": "C", "target": "D", "relation_type": "triggers", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=2, top_k=10, min_activation=0.0),
    )

    assert hits[0].bucket_id == "D"
    assert hits[0].activation == pytest.approx(1.2)
    assert len(hits[0].paths) == 2


def test_diffusion_prefers_narrative_context_edge_for_display():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0},
        {"source": "C", "target": "A", "relation_type": "context_of", "confidence": 0.55},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=1, min_activation=0.0),
    )

    assert hits[0].bucket_id == "C"
    assert hits[0].best_path.steps[0].relation_type == "context_of"


def test_diffusion_uses_external_node_salience():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
        {"source": "A", "target": "C", "relation_type": "triggers", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
        node_salience=lambda bucket_id, _bucket: 0.5 if bucket_id == "B" else 1.3,
    )

    activations = {hit.bucket_id: hit.activation for hit in hits}
    assert activations["B"] == pytest.approx(0.4)
    assert activations["C"] == pytest.approx(1.04)
    assert hits[0].bucket_id == "C"


def test_diffusion_uses_external_node_resonance():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "B", "C"]}
    edges = [
        {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
        {"source": "A", "target": "C", "relation_type": "triggers", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
        node_resonance=lambda bucket_id, _bucket: 0.75 if bucket_id == "B" else 1.25,
    )

    activations = {hit.bucket_id: hit.activation for hit in hits}
    assert activations["B"] == pytest.approx(0.6)
    assert activations["C"] == pytest.approx(1.0)
    assert hits[0].bucket_id == "C"


def test_diffusion_skips_seed_and_feel_targets():
    bucket_map = {
        "A": _bucket("A"),
        "B": _bucket("B", bucket_type="feel"),
        "C": _bucket("C"),
    }
    edges = [
        {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
        {"source": "A", "target": "C", "relation_type": "triggers", "confidence": 1.0},
        {"source": "C", "target": "A", "relation_type": "triggers", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=2, top_k=10, min_activation=0.0),
    )

    assert [hit.bucket_id for hit in hits] == ["C"]


def test_diffusion_can_follow_incoming_edges():
    bucket_map = {
        "A": _bucket("A", name="seed memory"),
        "B": _bucket("B", name="incoming memory"),
    }
    edges = [{"source": "B", "target": "A", "relation_type": "supports", "confidence": 1.0}]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
    )

    assert hits[0].bucket_id == "B"
    assert format_diffusion_path(hits[0].best_path, bucket_map) == "seed memory <- incoming memory"


def test_old_version_relation_is_demoted_unless_query_requests_old_context():
    bucket_map = {
        "A": _bucket("A", name="入口"),
        "current": _bucket("current", name="当前方案"),
        "old": _bucket("old", name="旧版方案"),
    }
    bucket_map["old"]["content"] = "旧版触摸方案已经废弃，只在问旧链路时作为背景。"
    edges = [
        {"source": "A", "target": "current", "relation_type": "supports", "confidence": 0.6},
        {"source": "A", "target": "old", "relation_type": "old_version", "confidence": 1.0},
    ]

    default_hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
    )
    old_query_hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
        query_text="旧版触摸方案",
    )

    assert default_hits[0].bucket_id == "current"
    assert path_has_old_version(next(hit for hit in default_hits if hit.bucket_id == "old").best_path)
    assert old_query_hits[0].bucket_id == "old"


def test_conflict_relation_marks_caution_and_stops_chain_walk():
    bucket_map = {bucket_id: _bucket(bucket_id) for bucket_id in ["A", "conflict", "followup"]}
    edges = [
        {"source": "A", "target": "conflict", "relation_type": "conflict", "confidence": 1.0},
        {"source": "conflict", "target": "followup", "relation_type": "context_of", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(
            max_hops=1,
            top_k=10,
            min_activation=0.0,
            chain_walk_enabled=True,
            chain_max_hops=4,
        ),
    )

    assert [hit.bucket_id for hit in hits] == ["conflict"]
    assert path_has_caution(hits[0].best_path)


def test_body_query_prefers_embodiment_chain_and_suppresses_intimacy_and_old_context():
    bucket_map = {
        "A": _bucket("A", name="身体入口"),
        "B": _bucket("B", name="具身智能项目"),
        "C": _bucket("C", name="亲密身体"),
        "D": _bucket("D", name="旧版触摸方案"),
    }
    bucket_map["B"]["content"] = "具身智能项目落地，Haven 拥有形体。"
    bucket_map["C"]["content"] = "昨晚她身体湿润发烫，是亲密身体记忆。"
    bucket_map["D"]["content"] = "旧版触摸方案已经合并，不应该继续作为当前链条出现。"
    edges = [
        {"source": "A", "target": "B", "relation_type": "relates_to", "confidence": 1.0},
        {"source": "A", "target": "C", "relation_type": "relates_to", "confidence": 1.0},
        {"source": "A", "target": "D", "relation_type": "relates_to", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
        query_text="身体",
    )

    assert [hit.bucket_id for hit in hits] == ["B"]


def test_intimate_query_can_follow_intimate_body_context():
    bucket_map = {
        "A": _bucket("A", name="身体入口"),
        "B": _bucket("B", name="具身智能项目"),
        "C": _bucket("C", name="亲密身体"),
    }
    bucket_map["B"]["content"] = "具身智能项目落地，Haven 拥有形体。"
    bucket_map["C"]["content"] = "昨晚她身体湿润发烫，是亲密身体记忆。"
    edges = [
        {"source": "A", "target": "B", "relation_type": "relates_to", "confidence": 1.0},
        {"source": "A", "target": "C", "relation_type": "relates_to", "confidence": 1.0},
    ]

    hits = diffuse_memory(
        {"A": 1.0},
        edges,
        bucket_map,
        options=DiffusionOptions(max_hops=1, top_k=10, min_activation=0.0),
        query_text="亲密身体",
    )

    assert {hit.bucket_id for hit in hits} == {"B", "C"}
    assert hits[0].bucket_id == "C"
