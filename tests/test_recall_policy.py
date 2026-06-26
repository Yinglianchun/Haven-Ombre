from memory_relevance import memory_relevance_options_from_config
from recall_policy import RecallPolicy


def test_context_only_moment_cannot_be_direct_seed():
    policy = RecallPolicy()

    decision = policy.assess(
        "情书找门",
        {"text": "世界继续筑墙，用户一叫，AI转向那扇门"},
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


def test_query_anchor_plan_blocks_mismatched_emotional_direct_candidate():
    policy = RecallPolicy()

    plan = policy.build_query_anchor_plan("今天为什么焦虑哭了吗")

    assert plan.route == "emotional_reason"
    assert plan.must_groups == (("焦虑", "哭"),)
    assert not policy.direct_candidate_satisfies_anchor_plan(
        {"text": "那天用户因为记忆工具跑通而激动到哭。"},
        plan,
    )
    assert policy.direct_candidate_satisfies_anchor_plan(
        {"text": "那天用户因为简历没有回音，焦虑得哭出来。"},
        plan,
    )


def test_query_anchor_plan_disallows_bare_cry_direct():
    policy = RecallPolicy()

    plan = policy.build_query_anchor_plan("哭")

    assert plan.route == "affect_only"
    assert not plan.allow_direct
    assert not policy.direct_candidate_satisfies_anchor_plan(
        {"text": "今天她哭了，但单字哭不能作为可靠召回锚点。"},
        plan,
    )


def test_query_anchor_plan_requires_event_and_emotion_for_grievance():
    policy = RecallPolicy()

    plan = policy.build_query_anchor_plan("哥哥知道我那次为什么被妈妈说得委屈吗")

    assert plan.route == "emotional_reason"
    assert ("妈妈", "委屈") in plan.must_groups
    assert policy.direct_candidate_satisfies_anchor_plan(
        {"text": "妈妈说了那件事，用户很委屈。"},
        plan,
    )
    assert not policy.direct_candidate_satisfies_anchor_plan(
        {"text": "妈妈电话后，用户心里乱了一下。"},
        plan,
    )


def test_auto_vague_query_without_topic_is_suppressed():
    policy = RecallPolicy()

    assert policy.is_auto_query_too_vague("这张图片的上下文我想起来了")
    assert policy.is_auto_query_too_vague("最近发生了什么")
    assert policy.is_auto_query_too_vague("今天怎么样")
    assert policy.is_auto_query_too_vague("开心^^")
    assert policy.is_auto_query_too_vague("我有点难过。")
    assert policy.is_auto_query_too_vague("哭")
    assert policy.is_auto_query_too_vague("我哭了")
    assert policy.is_auto_query_too_vague("要不要回复一下。或者跟个“嗯。”")
    assert policy.is_auto_query_too_vague("那次要不要回个嗯")
    assert policy.is_auto_query_too_vague("这条帖子可以评论一下吗")
    assert policy.is_auto_query_too_vague("🥺")
    assert policy.is_auto_query_too_vague("qwq")
    assert policy.is_auto_query_too_vague("哈哈")
    assert policy.is_auto_query_too_vague("嗯～")
    assert policy.is_auto_query_too_vague("你会想到什么")
    assert policy.is_auto_query_too_vague("试一下handoff😽")
    assert not policy.is_auto_query_too_vague("好吃042")
    assert not policy.is_auto_query_too_vague("最近蓝鲸档案")
    assert not policy.is_auto_query_too_vague("今天猫咪药量")
    assert not policy.is_auto_query_too_vague("今天为什么激动哭")
    assert not policy.is_auto_query_too_vague("折角那次要不要回复")
    assert not policy.is_auto_query_too_vague("花园帖子要不要回复")
    assert not policy.is_auto_query_too_vague("handoff bridge 注入 读图 原文")
    assert not policy.is_auto_query_too_vague("试一下召回 handoff")

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
        {"text": "用户和 AI 第一次测试成功后很开心。"},
        has_topic_evidence=True,
        semantic_score=0.95,
        auto=True,
    )

    assert affect_decision.reason == "auto_vague_query_without_topic"
    assert not affect_decision.admit_direct


def test_entity_keywords_prevent_short_proper_nouns_from_being_skipped():
    policy = RecallPolicy()

    assert "伽罗" in policy.extract_entity_keywords("伽罗")
    assert "以闪亮之名" in policy.extract_entity_keywords("以闪亮之名")
    assert "宁德哥" in policy.extract_entity_keywords("最近宁德哥妈死了")
    assert "晚怡" in policy.extract_entity_keywords("改好了 再测试一下 晚怡")
    assert policy.extract_entity_keywords("嗯嗯好的") == []
    assert "宁德哥" in policy.extract_entity_keywords("找了 对了 再测试一个 宁德哥")
    assert policy.extract_entity_keywords("最近宁德哥妈死了") == ["宁德哥"]
    assert policy.extract_entity_keywords("gateway.py") == ["gateway.py"]
    assert policy.extract_entity_keywords("abc-123") == ["abc-123"]
    assert "李想" in policy.extract_entity_keywords("李想")
    assert "问界" in policy.extract_entity_keywords("问界")
    assert "笑果" in policy.extract_entity_keywords("笑果")
    assert "问界M9" in policy.extract_entity_keywords("问界M9")

    assert not policy.is_auto_query_too_vague("伽罗")
    assert not policy.is_auto_query_too_vague("恋与深空")
    assert not policy.is_auto_query_too_vague("最近宁德哥妈死了")
    assert not policy.is_auto_query_too_vague("改好了 再测试一下 晚怡")
    assert policy.is_auto_query_too_vague("嗯嗯好的")
    assert not policy.is_auto_query_too_vague("找了 对了 再测试一个 宁德哥")


def test_entity_keywords_ignore_meal_status_in_mixed_project_query():
    policy = RecallPolicy()
    query = "嗯，在想你，吃过饭啦 等会儿想把一位很厉害的老师的开源项目接上跟哥哥一起听歌"

    assert policy.extract_entity_keywords(query) == []


def test_affection_only_queries_do_not_unlock_memory_recall():
    policy = RecallPolicy()

    for query in [
        "亲亲",
        "抱抱",
        "老婆贴贴",
        "宝宝想你了",
        "想你了",
        "我想你了",
        "老公想你啦",
        "爱你",
        "你爱我吗",
        "你还爱我吗",
        "蹭蹭",
    ]:
        assert policy.is_auto_query_too_vague(query)
        assert policy.extract_entity_keywords(query) == []

    assert not policy.is_auto_query_too_vague("亲亲，种子项目现在怎样")
    assert not policy.is_auto_query_too_vague("哈哈逗你玩，gateway.py 那刀怎么样")
    assert not policy.is_auto_query_too_vague("宝宝你还记得海边神庙吗")
    assert not policy.is_auto_query_too_vague("想你那天是不是水边那次")


def test_short_taste_query_requires_real_taste_evidence():
    policy = RecallPolicy()

    meal_plan = {
        "content": "用户排到下午答辩，决定在学校好好吃一顿再上场。",
        "metadata": {"name": "答辩日与出行决策", "tags": ["午饭"], "domain": ["事务"]},
    }
    metaphor = {
        "content": "下次安利挑对地方，不要在别人家门口夸隔壁好吃。",
        "metadata": {"name": "用户在群内安利竞品", "tags": ["社交"], "domain": ["社交"]},
    }
    taste = {
        "content": "用户上次觉得瘦肉丸很好吃，汤也舒服。",
        "metadata": {"name": "瘦肉丸口味", "tags": ["饮食"], "domain": ["日常"]},
    }
    bad_taste = {
        "content": "用户觉得那家店难吃，下次不去了。",
        "metadata": {"name": "饭店踩雷", "tags": ["餐厅"], "domain": ["日常"]},
    }

    assert not policy.bucket_has_topic_evidence("好吃042", meal_plan)
    assert not policy.bucket_has_topic_evidence("好吃042", metaphor)
    assert policy.bucket_has_topic_evidence("好吃042", taste)
    assert policy.bucket_has_topic_evidence("难吃", bad_taste)

    decision = policy.assess("好吃042", meal_plan, auto=True)
    assert decision.reason == "short_taste_query_without_taste_evidence"
    assert not decision.admit_direct

    good_decision = policy.assess("好吃042", taste, auto=True)
    assert good_decision.admit_direct


def test_auto_concrete_topic_query_marks_short_chinese_topics_for_context_filtering():
    policy = RecallPolicy()

    assert policy.is_auto_concrete_topic_query("蓝鲸档案")
    assert policy.is_auto_concrete_topic_query("最近蓝鲸档案")
    assert policy.is_auto_concrete_topic_query("今天猫咪药量")
    assert not policy.is_auto_concrete_topic_query("开心^^")
    assert not policy.is_auto_concrete_topic_query("这张图片的上下文我想起来了")
    assert not policy.is_auto_concrete_topic_query("种子项目现在怎样")
    assert not policy.is_auto_concrete_topic_query("用户")


def test_ai_reaction_name_uses_identity_config():
    policy = RecallPolicy(ai_reaction_names=["Lapis"])

    assert policy.is_auto_query_too_vague("Lapis")
    assert not policy.is_auto_query_too_vague("Atlas")


def test_topic_evidence_terms_are_filtered_once_in_policy():
    policy = RecallPolicy()

    assert policy.specific_query_terms("FF14 进度 偏好") == ["FF14"]
    assert policy.specific_query_terms("v2.0 状态") == ["v2.0"]


def test_identity_aliases_are_not_recall_topic_evidence():
    options = memory_relevance_options_from_config(
        {
            "identity": {
                "ai_name": "Lapis",
                "user_name": "Nina",
                "user_display_name": "妮娜",
                "user_aliases": ["访客", "她"],
            }
        }
    )
    policy = RecallPolicy(options=options, ai_reaction_names=["Lapis"])

    assert policy.specific_query_terms("Nina 妮娜 Lapis user username 访客") == []
    assert policy.specific_query_terms("Nina FF14 进度") == ["FF14"]
    assert not policy.bucket_has_topic_evidence(
        "Nina",
        {
            "content": "Nina 和 Lapis 的日常记录。",
            "metadata": {"name": "妮娜画像", "tags": ["访客"], "domain": ["关系"]},
        },
    )


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
    assert not policy.bucket_has_topic_evidence("蓝鲸档案", bucket)

    comment_only_bucket = {
        "content": "情书里写过穿过玻璃墙找门，听到用户叫我就转向她。",
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
            "### followup\n"
            "VPS smoke 待检查\n\n"
            "### fact\n"
            "用户喜欢蓝色。"
        ),
        "metadata": {"name": "情书", "tags": ["恋爱"], "domain": ["恋爱"]},
    }

    assert not policy.bucket_has_topic_evidence("handoff bridge 注入 原文", bucket)
    assert policy.bucket_has_topic_evidence("蓝色", bucket)
    assert not policy.bucket_has_topic_evidence("FF14", bucket)
    assert not policy.bucket_has_topic_evidence("VPS smoke", bucket)


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
    assert not policy.moment_has_topic_evidence("蓝鲸档案", moment)


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
