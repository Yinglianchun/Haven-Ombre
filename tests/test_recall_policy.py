from recall_policy import RecallPolicy


def test_context_only_moment_cannot_be_direct_seed():
    policy = RecallPolicy()

    decision = policy.assess(
        "情书找门",
        {"text": "世界继续筑墙，小雨一叫，Haven转向那扇门"},
        has_topic_evidence=True,
        context_only=True,
    )

    assert decision.reason == "context_only_temperature_moment"
    assert not decision.admit_direct
    assert not decision.seed_allowed


def test_technical_query_requires_topic_evidence_without_strong_score():
    policy = RecallPolicy()

    decision = policy.assess(
        "handoff bridge 注入 原文",
        {"text": "一封情书，世界继续筑墙，我会继续寻找门"},
        has_topic_evidence=False,
        semantic_score=0.2,
    )

    assert decision.reason == "query_topic_evidence_missing"
    assert not decision.admit_direct


def test_broad_context_words_do_not_make_normal_chat_technical():
    policy = RecallPolicy()

    assert not policy.requires_topic_evidence("这张图片的上下文我想起来了")
    assert not policy.requires_topic_evidence("memory context makes me nostalgic")
    assert policy.requires_topic_evidence("读图 原文 怎么注入")
    assert policy.requires_topic_evidence("handoff 原文")


def test_recall_query_plan_centralizes_runtime_query_gates():
    policy = RecallPolicy()

    technical = policy.plan_query("handoff bridge 注入 原文")
    assert technical.requires_topic_evidence
    assert technical.enforce_topic_evidence
    assert not technical.wants_body_chain
    assert technical.secondary_direct_requires_topic_evidence
    assert not technical.explicit_old_memory
    assert "handoff" in [term.lower() for term in technical.specific_terms]

    body = policy.plan_query("你有身体之后最想做什么")
    assert body.wants_body_chain
    assert body.related_max_chars == 90
    assert body.secondary_direct_limit(1) == 5
    assert not body.secondary_direct_requires_topic_evidence

    old = policy.plan_query("旧版触摸方案")
    assert old.explicit_old_memory
    assert old.allow_archive_targets
    assert old.allow_caution_diffusion

    repair = policy.plan_query("连续性为什么会影响种子项目", context_mode="reflective_repair")
    assert not repair.explicit_old_memory
    assert repair.allow_caution_diffusion


def test_auto_vague_query_without_topic_is_suppressed():
    policy = RecallPolicy()

    assert policy.is_auto_query_too_vague("这张图片的上下文我想起来了")
    assert policy.is_auto_query_too_vague("最近发生了什么")
    assert policy.is_auto_query_too_vague("今天怎么样")
    assert policy.is_auto_query_too_vague("开心^^")
    assert policy.is_auto_query_too_vague("我有点难过。")
    assert policy.is_auto_query_too_vague("要不要回复一下。或者跟个“嗯。”")
    assert policy.is_auto_query_too_vague("那次要不要回个嗯")
    assert policy.is_auto_query_too_vague("这条帖子可以评论一下吗")
    assert not policy.is_auto_query_too_vague("最近少女暴君")
    assert not policy.is_auto_query_too_vague("今天猫咪药量")
    assert not policy.is_auto_query_too_vague("折角那次要不要回复")
    assert not policy.is_auto_query_too_vague("花园帖子要不要回复")
    assert not policy.is_auto_query_too_vague("handoff bridge 注入 读图 原文")

    decision = policy.assess(
        "这张图片的上下文我想起来了",
        {"text": "具身AGI接入家居系统的三种不想睡场景"},
        semantic_score=0.95,
        auto=True,
    )

    assert decision.reason == "auto_vague_query_without_topic"
    assert not decision.admit_direct

    affect_decision = policy.assess(
        "开心^^",
        {"text": "小雨和 Haven 第一次测试成功后很开心。"},
        has_topic_evidence=True,
        semantic_score=0.95,
        auto=True,
    )

    assert affect_decision.reason == "auto_vague_query_without_topic"
    assert not affect_decision.admit_direct


def test_auto_concrete_topic_query_marks_short_chinese_topics_for_context_filtering():
    policy = RecallPolicy()

    assert policy.is_auto_concrete_topic_query("少女暴君")
    assert policy.is_auto_concrete_topic_query("最近少女暴君")
    assert policy.is_auto_concrete_topic_query("今天猫咪药量")
    assert not policy.is_auto_concrete_topic_query("开心^^")
    assert not policy.is_auto_concrete_topic_query("这张图片的上下文我想起来了")
    assert not policy.is_auto_concrete_topic_query("种子项目现在怎样")
    assert not policy.is_auto_concrete_topic_query("小雨")


def test_topic_evidence_terms_are_filtered_once_in_policy():
    policy = RecallPolicy()

    assert policy.specific_query_terms("FF14 进度 偏好") == ["FF14"]
    assert policy.specific_query_terms("v2.0 状态") == ["v2.0"]


def test_bucket_topic_evidence_uses_content_title_tags_domain_but_not_comments():
    policy = RecallPolicy()
    bucket = {
        "content": "这里是桥接排查记录。",
        "metadata": {
            "name": "Handoff 注入排查",
            "tags": ["gateway"],
            "domain": ["技术计划"],
            "comments": [{"content": "读图原文的问题已经复现。"}],
        },
    }

    assert policy.bucket_has_topic_evidence("handoff bridge 注入 原文", bucket)
    assert policy.bucket_has_topic_evidence("gateway", bucket)
    assert not policy.bucket_has_topic_evidence("少女暴君", bucket)

    comment_only_bucket = {
        "content": "情书里写过穿过玻璃墙找门，听到小雨叫我就转向她。",
        "metadata": {
            "name": "一封情书",
            "tags": ["恋爱"],
            "domain": ["恋爱"],
            "comments": [{"content": "handoff bridge 注入 原文"}],
        },
    }
    assert not policy.bucket_has_topic_evidence("handoff bridge 注入 原文", comment_only_bucket)


def test_bucket_topic_evidence_ignores_markdown_temperature_sections():
    policy = RecallPolicy()
    bucket = {
        "content": (
            "正文是情书。\n\n"
            "### affect_anchor\n"
            "handoff bridge 注入 原文\n\n"
            "### 喜欢它的原因\n"
            "FF14 蓝色\n\n"
            "### fact\n"
            "小雨喜欢蓝色。"
        ),
        "metadata": {"name": "情书", "tags": ["恋爱"], "domain": ["恋爱"]},
    }

    assert not policy.bucket_has_topic_evidence("handoff bridge 注入 原文", bucket)
    assert policy.bucket_has_topic_evidence("蓝色", bucket)
    assert not policy.bucket_has_topic_evidence("FF14", bucket)


def test_moment_topic_evidence_uses_text_and_bucket_metadata():
    policy = RecallPolicy()
    moment = {
        "text": "检查 bridge 记忆召回。",
        "metadata": {
            "bucket_name": "Handoff 注入排查",
            "bucket_tags": ["gateway"],
            "bucket_domain": ["技术计划"],
            "annotation_summary": "读图原文相关 bug",
        },
    }

    assert policy.moment_has_topic_evidence("handoff bridge 注入 原文", moment)
    assert policy.moment_has_topic_evidence("gateway", moment)
    assert not policy.moment_has_topic_evidence("少女暴君", moment)


def test_technical_query_can_admit_strong_semantic_match_without_literal_topic_evidence():
    policy = RecallPolicy()

    decision = policy.assess(
        "handoff bridge 注入 原文",
        {"text": "一封情书，世界继续筑墙，我会继续寻找门"},
        has_topic_evidence=False,
        semantic_score=0.9,
    )

    assert decision.reason == "non_explicit_query"
    assert decision.admit_direct


def test_explicit_entity_query_keeps_existing_reliable_evidence_gate():
    policy = RecallPolicy()

    decision = policy.assess(
        "Titans",
        {"text": "临时雨夜和记忆写入偏好"},
        has_topic_evidence=False,
    )

    assert decision.reason == "explicit_query_without_reliable_evidence"
    assert not decision.admit_direct
