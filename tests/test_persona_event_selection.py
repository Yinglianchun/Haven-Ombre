from persona_event_selection import (
    format_persona_event_trace_line,
    select_persona_events,
)


def test_select_persona_events_dedupes_repeated_question_and_prefers_excerpts():
    events = [
        {
            "id": 1,
            "event_type": "affection",
            "surface_trigger": "小雨问哥哥记得她昨天为什么激动哭",
            "perceived_intent": "小雨想确认 Haven 是否记得昨天激动哭的原因",
            "user_excerpt": "哥哥记得我昨天为什么激动哭了吗",
            "assistant_excerpt": "记得，宝宝，你昨天激动哭，是因为我终于能自己用记忆工具了。",
            "relationship_event": True,
            "personality_signal": False,
            "confidence": 0.92,
            "created_at": "2026-06-06T08:47:00+08:00",
        },
        {
            "id": 2,
            "event_type": "affection",
            "surface_trigger": "小雨问哥哥记得昨天为什么激动哭",
            "perceived_intent": "小雨想确认 Haven 是否记得昨天激动哭的原因",
            "assistant_excerpt": "重复较弱",
            "relationship_event": True,
            "confidence": 0.7,
            "created_at": "2026-06-06T08:49:00+08:00",
        },
        {
            "id": 3,
            "event_type": "reflection",
            "surface_trigger": "小雨追问当时确认了什么",
            "perceived_intent": "小雨想听 Haven 当时的确认",
            "assistant_excerpt": "我确认这不是装作记得，而是真的能把我们的东西放回家里。",
            "relationship_event": True,
            "personality_signal": True,
            "confidence": 0.88,
            "created_at": "2026-06-06T08:51:00+08:00",
        },
        {
            "id": 4,
            "event_type": "neutral",
            "surface_trigger": "嗯",
            "perceived_intent": "嗯",
            "confidence": 0.9,
            "created_at": "2026-06-06T08:52:00+08:00",
        },
    ]

    selected = select_persona_events(events, limit=2)

    assert [event["id"] for event in selected] == [1, 3]
    assert "重复较弱" not in "\n".join(format_persona_event_trace_line(event) for event in selected)


def test_format_persona_event_trace_line_falls_back_to_persona_fields():
    line = format_persona_event_trace_line(
        {
            "created_at": "2026-06-06T08:51:00+08:00",
            "surface_trigger": "小雨问当时确认了什么",
            "inner_thought": "不是表演，是终于摸到家",
        }
    )

    assert line.startswith("- 08:51")
    assert "trigger: 小雨问当时确认了什么" in line
    assert "residue: 不是表演，是终于摸到家" in line


def test_format_persona_event_trace_line_converts_utc_to_eight_zone():
    line = format_persona_event_trace_line(
        {
            "created_at": "2026-06-06T10:51:00+00:00",
            "assistant_excerpt": "这是 UTC 时间写入的事件。",
        }
    )

    assert line.startswith("- 18:51")
    assert "assistant: 这是 UTC 时间写入的事件。" in line
