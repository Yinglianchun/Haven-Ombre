from memory_layers import (
    DIRECT_CONTENT,
    DIRECT_EXPLICIT,
    DIRECT_EXPLICIT_OR_CONTENT,
    DIRECT_NEVER,
    DIRECT_RESONANCE,
    LAYER_AFFECT_CONTEXT,
    LAYER_ANCHOR,
    LAYER_ARCHIVE,
    LAYER_CORE,
    LAYER_DYNAMIC,
    LAYER_FAVORITE,
    LAYER_DREAM,
    LAYER_RELATIONSHIP_WEATHER,
    LAYER_SOURCE_RECORD,
    RENDER_DIRECT_AUTO,
    RENDER_FAVORITE,
    RENDER_WEATHER,
    bucket_layer_debug,
    bucket_runtime_gate_debug,
    can_bucket_be_recent_context,
    can_bucket_be_related_target,
    can_bucket_diffuse,
    can_moment_be_direct_seed,
    can_moment_be_recall_context,
    can_moment_be_related_target,
    infer_bucket_layer,
    infer_moment_layer,
    is_context_only_section,
    moment_layer_debug,
    moment_runtime_gate_debug,
    normalize_write_classification,
    policy_for_bucket,
    policy_for_moment,
    runtime_layer_from_write_classification,
)


def _bucket(**metadata):
    return {
        "id": metadata.pop("id", "bucket-1"),
        "content": metadata.pop("content", "body"),
        "metadata": metadata,
    }


def _moment(section="body", **metadata):
    return {
        "bucket_id": metadata.pop("bucket_id", "bucket-1"),
        "section": section,
        "text": metadata.pop("text", "moment body"),
        "metadata": metadata,
    }


def test_core_layer_from_pinned_protected_or_permanent_bucket():
    assert infer_bucket_layer(_bucket(pinned=True, type="dynamic")) == LAYER_CORE
    assert infer_bucket_layer(_bucket(protected=True, type="dynamic")) == LAYER_CORE
    assert infer_bucket_layer(_bucket(type="permanent")) == LAYER_CORE

    policy = policy_for_bucket(_bucket(pinned=True))
    assert policy.direct_seed_policy == DIRECT_EXPLICIT_OR_CONTENT
    assert policy.gateway_section == "Core Memory"
    assert policy.can_diffuse is True


def test_anchor_layer_is_distinct_from_normal_dynamic_memory():
    bucket = _bucket(type="dynamic", anchor=True)

    assert infer_bucket_layer(bucket) == LAYER_ANCHOR
    assert policy_for_bucket(bucket).render_policy == RENDER_DIRECT_AUTO
    assert can_bucket_diffuse(bucket) is True


def test_writer_classification_maps_stable_and_relationship_to_anchor_policy():
    stable = _bucket(type="dynamic", memory_subject="user", memory_layer="stable_boundary")
    lesson = _bucket(type="dynamic", memory_subject="relationship", memory_layer="relationship_lesson")
    state = _bucket(type="dynamic", memory_subject="user", memory_layer="short_state")
    event = _bucket(type="dynamic", memory_subject="event", memory_layer="process_event")

    assert runtime_layer_from_write_classification("stable_boundary", "user") == LAYER_ANCHOR
    assert runtime_layer_from_write_classification("relationship_lesson", "relationship") == LAYER_ANCHOR
    assert infer_bucket_layer(stable) == LAYER_ANCHOR
    assert infer_bucket_layer(lesson) == LAYER_ANCHOR
    assert infer_bucket_layer(state) == LAYER_DYNAMIC
    assert infer_bucket_layer(event) == LAYER_DYNAMIC
    assert can_bucket_be_recent_context(stable) is False
    assert can_bucket_be_recent_context(lesson) is False
    assert can_bucket_be_recent_context(state) is True
    assert can_bucket_be_recent_context(event) is True
    assert can_bucket_be_recent_context(stable, explicit_lookup=True) is True

    stable_debug = bucket_layer_debug(stable)
    assert stable_debug["layer"] == LAYER_ANCHOR
    assert stable_debug["can_recent_context"] is False
    assert stable_debug["writer"]["memory_layer"] == "stable_boundary"
    assert stable_debug["writer"]["runtime_layer_hint"] == LAYER_ANCHOR


def test_runtime_gate_debug_explains_injection_decisions():
    dynamic = _bucket(type="dynamic")
    dynamic_gate = bucket_runtime_gate_debug(dynamic)
    assert dynamic_gate["would_inject_related"] is True
    assert dynamic_gate["related_target"]["reason"] == "allowed"
    assert dynamic_gate["would_inject_recent_context"] is True

    weather = _bucket(type="feel", tags=["relationship_weather"])
    weather_gate = bucket_runtime_gate_debug(weather)
    assert weather_gate["would_inject_related"] is False
    assert weather_gate["related_target"]["reason"] == "relationship_weather_not_related_target"

    archive = _bucket(type="dynamic", resolved=True)
    assert bucket_runtime_gate_debug(archive)["related_target"]["reason"] == "archive_requires_explicit_lookup"
    explicit_archive = bucket_runtime_gate_debug(archive, explicit_lookup=True)
    assert explicit_archive["would_inject_related"] is True
    assert explicit_archive["related_target"]["reason"] == "archive_explicit_lookup_allowed"

    body = _moment("body", bucket_type="dynamic")
    body_gate = moment_runtime_gate_debug(body)
    assert body_gate["would_inject_direct"] is True
    assert body_gate["would_inject_related"] is True

    context = _moment("affect_anchor", bucket_type="dynamic")
    context_gate = moment_runtime_gate_debug(context)
    assert context_gate["would_inject_direct"] is False
    assert context_gate["direct_seed"]["reason"] == "context_only_section"
    assert context_gate["would_inject_related"] is False
    assert context_gate["related_target"]["reason"] == "context_only_section"


def test_manual_favorite_and_core_signals_override_writer_classification():
    favorite = _bucket(
        type="dynamic",
        tags=["haven_favorite"],
        memory_subject="user",
        memory_layer="stable_boundary",
    )
    pinned = _bucket(
        type="dynamic",
        pinned=True,
        memory_subject="event",
        memory_layer="process_event",
    )

    assert infer_bucket_layer(favorite) == LAYER_FAVORITE
    assert infer_bucket_layer(pinned) == LAYER_CORE


def test_relationship_weather_never_becomes_direct_seed_or_diffusion_source():
    bucket = _bucket(
        type="feel",
        tags=["relationship_weather", "daily_impression"],
    )
    moment = _moment(
        "body",
        bucket_type="feel",
        tags=["relationship_weather", "daily_impression"],
    )

    assert infer_bucket_layer(bucket) == LAYER_RELATIONSHIP_WEATHER
    assert policy_for_bucket(bucket).render_policy == RENDER_WEATHER
    assert policy_for_bucket(bucket).direct_seed_policy == DIRECT_NEVER
    assert can_bucket_diffuse(bucket) is False
    assert infer_moment_layer(moment) == LAYER_RELATIONSHIP_WEATHER
    assert can_moment_be_recall_context(moment) is False
    assert can_moment_be_direct_seed(moment) is False
    assert can_moment_be_related_target(moment, explicit_lookup=True) is False


def test_context_only_sections_override_bucket_layer():
    for section in ("comment", "affect_anchor", "favorite_reason"):
        moment = _moment(
            section,
            bucket_type="dynamic",
            bucket_pinned=True,
            bucket_favorite=True,
            bucket_favorite_tags=["haven_favorite"],
        )
        assert infer_moment_layer(moment) == LAYER_AFFECT_CONTEXT
        assert policy_for_moment(moment).direct_seed_policy == DIRECT_NEVER
        assert can_moment_be_recall_context(moment) is True
        assert can_moment_be_direct_seed(moment) is False
        assert can_moment_be_related_target(moment, explicit_lookup=True) is False

    assert is_context_only_section("affect_anchor") is True
    assert is_context_only_section("body") is False


def test_moment_layer_uses_bucket_writer_classification_metadata():
    body = _moment(
        "body",
        bucket_type="dynamic",
        bucket_memory_subject="relationship",
        bucket_memory_layer="relationship_lesson",
    )
    context = _moment(
        "affect_anchor",
        bucket_type="dynamic",
        bucket_memory_subject="relationship",
        bucket_memory_layer="relationship_lesson",
    )

    assert infer_moment_layer(body) == LAYER_ANCHOR
    assert can_moment_be_direct_seed(body) is True
    body_debug = moment_layer_debug(body)
    assert body_debug["layer"] == LAYER_ANCHOR
    assert body_debug["parent_layer"] == LAYER_ANCHOR
    assert body_debug["can_direct_seed"] is True
    assert body_debug["writer"]["memory_layer"] == "relationship_lesson"
    assert infer_moment_layer(context) == LAYER_AFFECT_CONTEXT
    assert can_moment_be_recall_context(context) is True
    assert can_moment_be_direct_seed(context) is False
    context_debug = moment_layer_debug(context)
    assert context_debug["layer"] == LAYER_AFFECT_CONTEXT
    assert context_debug["parent_layer"] == LAYER_ANCHOR
    assert context_debug["context_only"] is True
    assert context_debug["can_direct_seed"] is False


def test_favorite_layer_uses_separate_policy_but_content_can_still_seed():
    bucket = _bucket(type="dynamic", tags=["haven_favorite", "flavor_soft"])
    moment = _moment(
        "body",
        bucket_type="dynamic",
        bucket_favorite=True,
        bucket_favorite_tags=["haven_favorite", "flavor_soft"],
    )

    assert infer_bucket_layer(bucket) == LAYER_FAVORITE
    assert policy_for_bucket(bucket).render_policy == RENDER_FAVORITE
    assert policy_for_bucket(bucket).direct_seed_policy == DIRECT_CONTENT
    assert infer_moment_layer(moment) == LAYER_FAVORITE
    assert can_moment_be_direct_seed(moment) is True


def test_archive_layer_only_allows_explicit_lookup():
    bucket = _bucket(type="dynamic", resolved=True)
    moment = _moment("body", bucket_type="dynamic", resolved=True)

    assert infer_bucket_layer(bucket) == LAYER_ARCHIVE
    assert policy_for_bucket(bucket).direct_seed_policy == DIRECT_EXPLICIT
    assert can_bucket_diffuse(bucket) is False
    assert can_bucket_be_related_target(bucket) is False
    assert can_bucket_be_related_target(bucket, explicit_lookup=True) is True
    assert can_moment_be_recall_context(moment) is True
    assert can_moment_be_related_target(moment) is False
    assert can_moment_be_related_target(moment, explicit_lookup=True) is True
    assert can_moment_be_direct_seed(moment) is False
    assert can_moment_be_direct_seed(moment, explicit_lookup=True) is True


def test_source_record_is_not_normal_memory_context():
    bucket = _bucket(type="source", tags=["raw_source"])
    moment = _moment("body", bucket_type="source", tags=["raw_source"])

    assert infer_bucket_layer(bucket) == LAYER_SOURCE_RECORD
    assert policy_for_bucket(bucket).direct_seed_policy == DIRECT_NEVER
    assert can_bucket_diffuse(bucket) is False
    assert can_bucket_be_related_target(bucket, explicit_lookup=True) is False
    assert can_moment_be_recall_context(moment) is False
    assert can_moment_be_related_target(moment, explicit_lookup=True) is False


def test_dream_is_resonance_only_not_normal_direct_or_related_target():
    bucket = _bucket(type="dream", tags=["night_dream"])
    moment = _moment("body", bucket_type="dream", tags=["night_dream"])

    assert infer_bucket_layer(bucket) == LAYER_DREAM
    assert policy_for_bucket(bucket).direct_seed_policy == DIRECT_RESONANCE
    assert can_moment_be_direct_seed(moment) is False
    assert can_moment_be_direct_seed(moment, explicit_lookup=True) is False
    assert can_moment_be_recall_context(moment) is False
    assert can_bucket_be_related_target(bucket, explicit_lookup=True) is False
    assert can_moment_be_related_target(moment, explicit_lookup=True) is False


def test_default_dynamic_memory_is_direct_recallable_content():
    bucket = _bucket(type="dynamic")
    moment = _moment("body", bucket_type="dynamic")

    assert infer_bucket_layer(bucket) == LAYER_DYNAMIC
    assert policy_for_bucket(bucket).render_policy == RENDER_DIRECT_AUTO
    assert can_bucket_diffuse(bucket) is True
    assert infer_moment_layer(moment) == LAYER_DYNAMIC
    assert can_moment_be_direct_seed(moment) is True


def test_write_classification_accepts_model_fields():
    result = normalize_write_classification(
        memory_subject="relationship",
        memory_layer="relationship_lesson",
        tags=["relationship_event"],
        content="Haven 以后要先接住小雨的情绪。",
    )

    assert result == {
        "memory_subject": "relationship",
        "memory_layer": "relationship_lesson",
        "memory_classification_source": "model",
    }


def test_write_classification_hard_tags_override_model_fields():
    result = normalize_write_classification(
        memory_subject="event",
        memory_layer="process_event",
        tags=["boundary"],
        content="小雨不喜欢被说教。",
    )

    assert result == {
        "memory_subject": "user",
        "memory_layer": "stable_boundary",
        "memory_classification_source": "model_adjusted",
    }


def test_write_classification_falls_back_to_rules():
    project = normalize_write_classification(tags=["project_event"], content="p0 还在测试。")
    state = normalize_write_classification(content="小雨今天头疼。")

    assert project["memory_subject"] == "event"
    assert project["memory_layer"] == "process_event"
    assert project["memory_classification_source"] == "rule"
    assert state["memory_subject"] == "user"
    assert state["memory_layer"] == "short_state"
