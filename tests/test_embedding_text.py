from utils import (
    bucket_text_for_embedding,
    strip_affect_anchor,
    strip_display_temperature_sections,
    strip_temperature_meaning_lines,
)


def test_bucket_text_for_embedding_includes_title_and_content_only():
    text = bucket_text_for_embedding(
        {
            "content": (
                "正文里有 [[双链]]。\n\n"
                "### affect_anchor\n\n"
                "> 小雨把旧信放到桌上。\n"
                "> Dbmaj9 -> Ab/C -> Bbm9\n\n"
                "含义：温度仍在。"
            ),
            "metadata": {
                "name": "标题 [[记忆]]",
                "comments": [{"content": "一圈 [[年轮]]"}],
            },
        }
    )

    assert "Title: 标题 记忆" in text
    assert "Content: 正文里有 双链。" in text
    assert "affect_anchor" not in text
    assert "Dbmaj9" not in text
    assert "年轮" not in text


def test_bucket_text_for_embedding_keeps_content_only_shape_without_title():
    assert bucket_text_for_embedding({"content": "只有正文", "metadata": {}}) == "只有正文"


def test_strip_affect_anchor_preserves_following_sections():
    text = (
        "正文。\n\n"
        "### affect_anchor\n\n"
        "> 场景\n"
        "> Cmaj7 -> G/B\n\n"
        "含义：移动。\n\n"
        "### 喜欢它的原因\n"
        "这里仍然要保留。"
    )

    cleaned = strip_affect_anchor(text)

    assert "正文。" in cleaned
    assert "affect_anchor" not in cleaned
    assert "Cmaj7" not in cleaned
    assert "### 喜欢它的原因" in cleaned
    assert "这里仍然要保留。" in cleaned


def test_strip_temperature_meaning_lines_only_removes_standalone_lines():
    text = (
        "> 小雨把旧信放到桌上。\n"
        "> Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp\n"
        "含义：这只是模板解释。\n"
        "正文里的含义：应该保留。"
    )

    cleaned = strip_temperature_meaning_lines(text)

    assert "模板解释" not in cleaned
    assert "Dbmaj9" not in cleaned
    assert "60bpm" not in cleaned
    assert "> 小雨把旧信放到桌上。" in cleaned
    assert "正文里的含义：应该保留。" in cleaned


def test_strip_temperature_meaning_lines_removes_inline_chord_tail():
    text = "> 小雨把旧信放到桌上。 > Dbmaj9 -> Ab/C -> Bbm9 · 60bpm · mp"

    cleaned = strip_temperature_meaning_lines(text)

    assert cleaned == "> 小雨把旧信放到桌上。"
    assert "Dbmaj9" not in cleaned
    assert "60bpm" not in cleaned


def test_strip_display_temperature_sections_removes_render_only_blocks():
    text = (
        "正文。\n\n"
        "### 喜欢它的原因\n"
        "这里是偏爱注释。\n\n"
        "### affect_anchor\n"
        "> Cmaj7 -> G/B\n\n"
        "### 普通段落\n"
        "这里要保留。"
    )

    cleaned = strip_display_temperature_sections(text)

    assert "正文。" in cleaned
    assert "喜欢它的原因" not in cleaned
    assert "affect_anchor" not in cleaned
    assert "普通段落" in cleaned
    assert "这里要保留。" in cleaned
