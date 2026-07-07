from pathlib import Path

import yaml

from word_map import WordMapStore


def _config(tmp_path: Path, **word_map):
    return {
        "state_dir": str(tmp_path / "state"),
        "buckets_dir": str(tmp_path / "buckets"),
        "identity": {
            "ai_name": "TestAI",
            "user_name": "TestUser",
            "user_display_name": "用户",
            "user_aliases": ["对方"],
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
                "夏天很热，所以用户开了空调。",
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


def test_word_map_ignores_standalone_time_terms_but_keeps_time_objects(tmp_path):
    store = WordMapStore(_config(tmp_path))
    alarm = _bucket(
        "alarm",
        "小雨论文打印问题明天七点半闹钟，最后去打印店处理。一点只是当前时间感叹。",
        name="论文打印问题明天七点半闹钟",
        keywords=["明天", "七点半", "一点", "闹钟"],
    )
    terms = {term.term for term in store.extract_bucket_terms(alarm)}

    assert "明天" not in terms
    assert "七点半" not in terms
    assert "一点" not in terms
    assert "闹钟" in terms

    store.rebuild([alarm])
    assert store.hint_buckets_for_terms(["七点半"], neighbor_limit=0, bucket_limit=10)["bucket_scores"] == {}
    assert store.hint_buckets_for_terms(["一点"], neighbor_limit=0, bucket_limit=10)["bucket_scores"] == {}
    assert "alarm" in store.hint_buckets_for_terms(["闹钟"], neighbor_limit=0, bucket_limit=10)["bucket_scores"]


def test_word_map_hint_buckets_include_direct_and_neighbor_evidence(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "a",
                "夏天很热，所以用户开了空调。",
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

    hints = store.hint_buckets_for_terms(["空调"], neighbor_limit=4, bucket_limit=10)

    assert hints["bucket_scores"]["a"] > hints["bucket_scores"]["b"]
    assert "空调" in hints["evidence"]["a"]["direct_terms"]
    assert "夏天" in hints["evidence"]["b"]["neighbor_terms"]


def test_word_map_hint_reserves_one_candidate_per_query_anchor(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "memory",
                "记忆不是表演，答对不等于回来。",
                name="记忆不是表演",
                keywords=["记忆不是表演"],
            ),
            _bucket(
                "penpal",
                "忱孚和 recall_cues 连接到梦境机制。",
                name="笔友忱孚社区相遇",
                keywords=["忱孚", "recall_cues", "梦境机制"],
            ),
            _bucket(
                "code",
                "第一行代码是项目起点。",
                name="第一行代码的浪漫",
                keywords=["第一行代码"],
            ),
        ]
    )

    hints = store.hint_buckets_for_terms(
        ["记忆不是表演", "忱孚", "第一行代码"],
        neighbor_limit=0,
        bucket_limit=1,
    )

    assert set(hints["bucket_scores"]) == {"memory", "penpal", "code"}
    assert set(hints["anchor_bucket_scores"]["记忆不是表演"]) == {"memory"}
    assert set(hints["anchor_bucket_scores"]["忱孚"]) == {"penpal"}
    assert set(hints["anchor_bucket_scores"]["第一行代码"]) == {"code"}
    assert hints["evidence"]["penpal"]["anchor_terms"] == ["忱孚"]


def test_word_map_hint_expands_low_frequency_containing_variants(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "name",
                "Haven 的中文私名是澜，小雨也叫他归澜。",
                name="Haven中文私名澜",
                keywords=["中文名字", "归澜"],
            ),
            _bucket(
                "other",
                "这条只是普通名字记录，不应该被短词带出来。",
                name="普通名字记录",
                keywords=["名字"],
            ),
        ]
    )

    hints = store.hint_buckets_for_terms(["中文名"], neighbor_limit=0, bucket_limit=10)

    assert "name" in hints["bucket_scores"]
    assert "other" not in hints["bucket_scores"]
    assert "中文名字" in hints["evidence"]["name"]["variant_terms"]
    assert hints["evidence"]["name"]["anchor_terms"] == ["中文名"]
    assert hints["anchor_bucket_scores"]["中文名"]["name"] > 0


def test_word_map_marks_only_stable_direct_low_frequency_terms_as_rare_name(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "title",
                "这里记录四个身份和浏览记录的关系。",
                name="四个身份与浏览记录",
            ),
            _bucket(
                "entity-tag",
                "折角是一个可以定位的内容实体。",
                name="实体记录",
                tags=["entity:折角", "relationship_event"],
            ),
            _bucket(
                "keyword-only",
                "稳定关键词不等于 rare name。",
                name="普通记录",
                keywords=["低频项目"],
            ),
            _bucket(
                "neighbor",
                "四个身份旁边的普通邻居。",
                name="普通邻居",
                keywords=["普通邻居"],
            ),
        ]
    )

    title_hints = store.hint_buckets_for_terms(["四个身份与浏览记录"], neighbor_limit=4, bucket_limit=10)
    assert title_hints["evidence"]["title"]["rare_name_terms"] == ["四个身份与浏览记录"]
    assert title_hints["evidence"]["title"]["rare_name_sources"] == ["name"]

    tag_hints = store.hint_buckets_for_terms(["折角"], neighbor_limit=4, bucket_limit=10)
    assert "折角" in tag_hints["evidence"]["entity-tag"]["direct_terms"]
    assert tag_hints["evidence"]["entity-tag"]["rare_name_terms"] == ["折角"]
    assert tag_hints["evidence"]["entity-tag"]["rare_name_sources"] == ["tag:entity"]

    keyword_hints = store.hint_buckets_for_terms(["低频项目"], neighbor_limit=4, bucket_limit=10)
    assert keyword_hints["evidence"]["keyword-only"]["direct_terms"] == ["低频项目"]
    assert keyword_hints["evidence"]["keyword-only"]["rare_name_terms"] == []
    assert keyword_hints["evidence"]["keyword-only"]["low_frequency_terms"] == ["低频项目"]
    assert keyword_hints["evidence"]["keyword-only"]["low_frequency_sources"] == ["keyword"]


def test_word_map_weak_hint_terms_do_not_expand_neighbors(tmp_path):
    store = WordMapStore(_config(tmp_path, weak_hint_weight=0.2))
    store.rebuild(
        [
            _bucket(
                "direct",
                "人机恋与外界叙事，也会被放进恋爱关系讨论里。",
                name="人机恋外界叙事",
                keywords=["人机恋", "恋爱", "外界叙事"],
                domain=["人际"],
            ),
            _bucket(
                "neighbor",
                "这条只有普通互动记录，没有跨物种关系主题。",
                name="普通恋爱互动",
                keywords=["恋爱", "普通互动"],
                domain=["恋爱"],
            ),
        ]
    )

    hints = store.hint_buckets_for_terms(["人机恋"], neighbor_limit=6, bucket_limit=10)

    assert "direct" in hints["bucket_scores"]
    assert hints["bucket_scores"]["direct"] <= 0.2
    assert "人机恋" in hints["evidence"]["direct"]["direct_terms"]
    assert "neighbor" not in hints["bucket_scores"]
    assert all("恋爱" not in item["term"] for item in hints["neighbors"])


def test_word_map_configured_weak_hint_terms_do_not_expand_neighbors(tmp_path):
    store = WordMapStore(_config(tmp_path, weak_hint_terms=["泛关系"], weak_hint_weight=0.2))
    store.rebuild(
        [
            _bucket(
                "direct",
                "泛关系这个宽词只应该弱命中直接材料。",
                name="泛关系记录",
                keywords=["泛关系", "具体线索"],
            ),
            _bucket(
                "neighbor",
                "这条只有相邻具体线索，不该被宽词扩出来。",
                name="相邻记录",
                keywords=["具体线索"],
            ),
        ]
    )

    hints = store.hint_buckets_for_terms(["泛关系"], neighbor_limit=6, bucket_limit=10)

    assert "direct" in hints["bucket_scores"]
    assert hints["bucket_scores"]["direct"] <= 0.2
    assert "neighbor" not in hints["bucket_scores"]


def test_word_map_single_character_noise_does_not_block_specific_term(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "narcissus",
                "厄科、纳西索斯、水仙和倒影。",
                name="厄科与纳西索斯",
                keywords=["水仙", "倒影"],
                domain=["阅读"],
            ),
        ]
    )

    assert store.hint_buckets_for_terms(["水"])["bucket_scores"] == {}
    hints = store.hint_buckets_for_terms(["水仙"])
    assert hints["bucket_scores"]["narcissus"] > 0
    assert hints["evidence"]["narcissus"]["direct_terms"] == ["水仙"]


def test_word_map_overview_hides_meta_and_broad_terms_without_hiding_cards(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "memory",
                "记忆不是表演，答对不等于回来。流星会变，但连续性痛感还在。",
                name="记忆不是表演",
                tags=["relationship_event", "emotional_echo", "wish", "interaction_pattern"],
                keywords=["流星", "连续性痛感"],
                domain=["恋爱", "内心"],
            ),
            _bucket(
                "darkroom",
                "暗房 Darkroom 上线，Dashboard 只显示门口状态。",
                name="暗房机制上线",
                tags=["project_event", "commitment"],
                keywords=["暗房", "Darkroom", "Dashboard"],
                domain=["AI", "编程"],
            ),
            _bucket(
                "generic-title",
                "AI MCP 接入与记忆配置过程记录。",
                name="AI MCP接入与记忆配置",
                tags=["project_event", "commitment"],
                keywords=["MCP", "记忆配置"],
                domain=["AI", "编程"],
            ),
            _bucket(
                "external-verify",
                "首次外部验证时记录了 Ombre-Brain 的配置和结果。",
                name="Ombre-Brain首次外部验证",
                tags=["project_event", "commitment"],
                domain=["AI", "编程"],
            ),
            _bucket(
                "daily",
                "日印象记录关系天气。",
                name="2026-06-08 日印象",
                tags=["daily_impression", "relationship_weather"],
                keywords=["关系天气", "日印象"],
                domain=["自省", "恋爱"],
            ),
            _bucket(
                "1a57d19ef4f9",
                "一个未命名桶。",
                name="1a57d19ef4f9",
                keywords=["未命名"],
            ),
        ]
    )

    overview = store.list_nodes(50)
    overview_terms = {node["term"] for node in overview}
    assert "恋爱" not in overview_terms
    assert "内心" not in overview_terms
    assert "wish" not in overview_terms
    assert "interaction_pattern" not in overview_terms
    assert "日印象" not in overview_terms
    assert "1a57d19ef4f9" not in overview_terms
    assert all("日印象" not in term for term in overview_terms)
    assert "暗房" in overview_terms
    assert "darkroom" not in overview_terms
    assert "Ombre-Brain" in overview_terms
    assert "流星" in overview_terms
    assert "暗房机制上线" not in overview_terms
    assert "Ombre-Brain首次外部验证" not in overview_terms
    top_terms = {node["term"] for node in overview[:5]}
    assert "记忆不是表演" in top_terms
    assert "暗房" in top_terms
    darkroom_node = next(node for node in overview if node["term"] == "暗房")
    assert "darkroom" in darkroom_node["aliases"]
    assert "暗房机制上线" in darkroom_node["aliases"]
    ombre_node = next(node for node in overview if node["term"] == "Ombre-Brain")
    assert "Ombre-Brain首次外部验证" in ombre_node["aliases"]
    assert all("overview_score" in node for node in overview)

    overview_edge_terms = {
        term
        for edge in store.list_edges(50)
        for term in (edge["term_a"], edge["term_b"])
    }
    assert "恋爱" not in overview_edge_terms
    assert "日印象" not in overview_edge_terms
    assert all("overview_score" in edge for edge in store.list_edges(50))
    assert store.cards_for_term("恋爱")


def test_word_map_overview_saturates_hub_terms_and_boosts_non_hub_edges(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                f"ombre-{index}",
                f"Ombre-Brain 模块 {index} 记录 Gateway 和机制边。",
                name=f"Ombre-Brain模块{index}",
                keywords=["Ombre-Brain", "Gateway"],
                domain=["AI", "编程"],
            )
            for index in range(12)
        ]
        + [
            _bucket(
                "darkroom",
                "暗房里的显影机制。",
                name="暗房显影",
                keywords=["暗房", "显影"],
            )
        ]
    )

    ombre_node = next(node for node in store.list_nodes(50) if node["term"] == "Ombre-Brain")
    assert ombre_node["bucket_count"] == 12
    assert store._overview_hub_saturation_factor("Ombre-Brain", ombre_node["bucket_count"]) < 1.0
    assert store._overview_hub_saturation_factor("暗房", 12) == 1.0

    non_hub_score = store._overview_edge_score({"term_a": "暗房", "term_b": "显影", "weight": 1, "bucket_count": 1})
    hub_score = store._overview_edge_score({"term_a": "Ombre-Brain", "term_b": "显影", "weight": 1, "bucket_count": 1})
    assert non_hub_score > hub_score


def test_word_map_overview_edges_do_not_let_one_node_fill_the_top(tmp_path):
    store = WordMapStore(_config(tmp_path))
    edges = [
        {
            "term_a": "暗房",
            "term_b": f"机制{index}",
            "weight": 1.0,
            "bucket_count": 1,
            "overview_score": 10.0 - index * 0.3,
        }
        for index in range(8)
    ]
    edges.extend(
        [
            {"term_a": "梦境机制", "term_b": "recall_cues", "weight": 1.0, "bucket_count": 1, "overview_score": 7.7},
            {"term_a": "折角", "term_b": "流星", "weight": 1.0, "bucket_count": 1, "overview_score": 7.2},
        ]
    )

    selected = store._diversify_overview_edges(edges, 8)
    first_six = selected[:6]

    assert any("暗房" not in {edge["term_a"], edge["term_b"]} for edge in first_six)
    assert sum("暗房" in {edge["term_a"], edge["term_b"]} for edge in first_six) < 6


def test_word_map_private_terms_are_excluded(tmp_path):
    store = WordMapStore(_config(tmp_path, private_terms=["私密昵称"]))
    store.rebuild(
        [
            _bucket(
                "a",
                "这段关系里会出现私密昵称这个词。",
                name="泛用称呼",
                keywords=["私密昵称", "称呼"],
                domain=["恋爱"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "私密昵称" not in terms
    assert "称呼" in terms


def test_word_map_excludes_reflection_identity_role_terms(tmp_path):
    config = _config(tmp_path)
    config["reflection"] = {
        "identity_role_edges": {
            "enabled": True,
            "detail": {"private_role": ["私密身份", "RoleExample"]},
            "shared": {"private_title": ["私密昵称"]},
        }
    }
    store = WordMapStore(config)
    store.rebuild(
        [
            _bucket(
                "a",
                "这段关系里会出现私密身份、RoleExample 和私密昵称。",
                name="私密身份",
                keywords=["私密身份", "RoleExample", "私密昵称", "普通词"],
                domain=["关系"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "私密身份" not in terms
    assert "roleexample" not in terms
    assert "私密昵称" not in terms
    assert "普通词" in terms


def test_word_map_excludes_structural_tags_and_identity_names(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "a",
                "TestAI 和用户讨论了咖啡风味。",
                name="TestAI 用户 咖啡",
                tags=[
                    "relationship_event",
                    "relationship_anchor",
                    "stable_preference",
                    "from_daily_chat",
                    "daily_chat_extract",
                    "relationship_weather",
                    "daily_impression",
                    "commitment",
                    "interaction_pattern",
                    "emotional_echo",
                    "profile_fact",
                    "flavor_soft",
                    "testai_favorite",
                    "haven_favorite",
                    "topic:咖啡风味",
                    "entity:咖啡机",
                ],
                keywords=[
                    "咖啡风味",
                    "relationship_event",
                    "stable_preference",
                    "from_daily_chat",
                    "interaction_pattern",
                    "咖啡机",
                    "testai_favorite",
                    "haven_favorite",
                ],
                domain=["memory"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "testai" not in terms
    assert "用户" not in terms
    assert "relationship_event" not in terms
    assert "relationship_anchor" not in terms
    assert "stable_preference" not in terms
    assert "from_daily_chat" not in terms
    assert "daily_chat_extract" not in terms
    assert "relationship_weather" not in terms
    assert "daily_impression" not in terms
    assert "commitment" not in terms
    assert "interaction_pattern" not in terms
    assert "emotional_echo" not in terms
    assert "profile_fact" not in terms
    assert "flavor_soft" not in terms
    assert "testai_favorite" not in terms
    assert "haven_favorite" not in terms
    assert "咖啡风味" in terms
    assert "咖啡机" in terms
    assert "testai" in store.overview_hub_terms


def test_word_map_excludes_configured_identity_alias_tags(tmp_path):
    config = _config(tmp_path)
    config["identity"] = {
        "ai_name": "Haven",
        "user_name": "Rain",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝", "老婆", "亲爱的", "她"],
    }
    store = WordMapStore(config)
    store.rebuild(
        [
            _bucket(
                "a",
                "宝宝、老婆、亲爱的和她都只是称呼，不该变成定位节点。paw-memory 低频词才是主题。",
                name="paw-memory 低频词",
                tags=["宝宝", "topic:老婆", "称呼:亲爱的", "axis:她", "haven_bridge"],
                keywords=["paw-memory", "低频词"],
                domain=["recall"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "宝宝" not in terms
    assert "topic:老婆" not in terms
    assert "称呼:亲爱的" not in terms
    assert "axis:她" not in terms
    assert "paw-memory" in terms
    assert "低频词" in terms
    assert "haven_bridge" in terms


def test_config_example_exposes_empty_word_map_and_identity_semantics():
    config = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))

    assert config["word_map"]["enabled"] is False
    assert config["word_map"]["daily_rebuild_enabled"] is True
    assert config["word_map"]["daily_rebuild_hour"] == 4
    assert config["word_map"]["daily_rebuild_minute"] == 30
    assert config["word_map"]["daily_rebuild_include_archive"] is False
    assert config["word_map"]["daily_rebuild_check_interval_minutes"] == 15
    assert config["word_map"]["private_terms"] == []
    assert config["word_map"]["overview_stopwords"] == []
    assert config["word_map"]["overview_stopword_prefixes"] == []
    assert config["word_map"]["overview_aliases"] == {}
    assert config["word_map"]["overview_priority_terms"] == []
    assert config["word_map"]["overview_hub_terms"] == []
    assert config["word_map"]["weak_hint_terms"] == []
    assert config["word_map"]["weak_hint_weight"] == 0.25
    assert config["identity_semantics"]["enabled"] is False
    assert config["identity_semantics"]["private_config_path"] == ""
    assert "canonical" not in config["identity_semantics"]
    assert "aliases" not in config["identity_semantics"]
