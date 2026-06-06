from pathlib import Path

import yaml

from word_map import WordMapStore


def _config(tmp_path: Path, **word_map):
    return {
        "state_dir": str(tmp_path / "state"),
        "buckets_dir": str(tmp_path / "buckets"),
        "identity": {
            "ai_name": "Haven",
            "user_name": "Rain",
            "user_display_name": "小雨",
            "user_aliases": ["宝宝"],
        },
        "word_map": {
            "enabled": True,
            "max_terms_per_bucket": 8,
            "edge_top_k": 6,
            **word_map,
        },
    }


def _bucket(bucket_id: str, content: str, **metadata):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": metadata.pop("name", bucket_id),
            "tags": metadata.pop("tags", []),
            "domain": metadata.pop("domain", []),
            "keywords": metadata.pop("keywords", []),
            **metadata,
        },
    }


def test_word_map_rebuild_creates_nodes_edges_and_bucket_evidence(tmp_path):
    store = WordMapStore(_config(tmp_path))
    stats = store.rebuild(
        [
            _bucket(
                "a",
                "夏天很热，所以小雨开了空调。",
                name="夏天空调",
                keywords=["夏天", "空调"],
                domain=["生活"],
            ),
            _bucket(
                "b",
                "夏天也会想到冰美式。",
                name="夏天咖啡",
                keywords=["夏天", "冰美式"],
                domain=["生活"],
            ),
        ]
    )

    assert stats["nodes"] >= 3
    assert store.cards_for_term("夏天")
    edge_pairs = {(edge["term_a"], edge["term_b"]) for edge in store.list_edges()}
    assert ("夏天", "空调") in edge_pairs or ("空调", "夏天") in edge_pairs


def test_word_map_private_terms_are_excluded(tmp_path):
    store = WordMapStore(_config(tmp_path, private_terms=["专属称呼"]))
    store.rebuild(
        [
            _bucket(
                "a",
                "这段关系里会出现专属称呼这个词。",
                name="亲密称呼",
                keywords=["专属称呼", "称呼"],
                domain=["恋爱"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "专属称呼" not in terms
    assert "称呼" in terms


def test_word_map_excludes_reflection_identity_role_terms(tmp_path):
    config = _config(tmp_path)
    config["reflection"] = {
        "identity_role_edges": {
            "enabled": True,
            "detail": {"private_role": ["专属身份", "RoleX"]},
            "shared": {"private_title": ["专属称呼"]},
        }
    }
    store = WordMapStore(config)
    store.rebuild(
        [
            _bucket(
                "a",
                "这段关系里会出现专属身份、RoleX 和专属称呼。",
                name="专属身份",
                keywords=["专属身份", "RoleX", "专属称呼", "普通词"],
                domain=["关系"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "专属身份" not in terms
    assert "rolex" not in terms
    assert "专属称呼" not in terms
    assert "普通词" in terms


def test_word_map_excludes_structural_tags_and_identity_names(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "a",
                "Haven 和小雨讨论了咖啡风味。",
                name="Haven 小雨 咖啡",
                tags=["relationship_event", "emotional_echo", "profile_fact", "flavor_soft"],
                keywords=["咖啡风味", "relationship_event"],
                domain=["memory"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "haven" not in terms
    assert "小雨" not in terms
    assert "relationship_event" not in terms
    assert "emotional_echo" not in terms
    assert "profile_fact" not in terms
    assert "flavor_soft" not in terms
    assert "咖啡风味" in terms


def test_config_example_exposes_empty_word_map_and_identity_semantics():
    config = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))

    assert config["word_map"]["enabled"] is False
    assert config["word_map"]["private_terms"] == []
    assert config["identity_semantics"]["enabled"] is False
    assert config["identity_semantics"]["private_config_path"] == ""
    assert "canonical" not in config["identity_semantics"]
    assert "aliases" not in config["identity_semantics"]
