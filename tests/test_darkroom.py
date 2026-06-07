import pytest

from darkroom import DarkroomStore


def _store(tmp_path):
    return DarkroomStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )


def test_darkroom_enter_does_not_echo_note(tmp_path):
    store = _store(tmp_path)
    secret = "这是一句还没显影的暗房正文"

    result = store.enter(secret, completeness=0.4, mood="quiet", tags="暗房,未完成")

    assert result["status"] == "entered"
    assert result["visible_note"] == "Haven 进入了暗房。"
    assert secret not in str(result)
    assert result["completeness"] == {"previous": None, "current": 0.4}
    assert result["tags"] == ["暗房", "未完成"]


def test_darkroom_status_is_door_only(tmp_path):
    store = _store(tmp_path)
    secret = "不能出现在门口状态里的句子"
    first = store.enter(secret, completeness=0.2)
    second = store.enter("第二条也不该回显", completeness=0.6, mood="developing")

    status = store.status()

    assert status["status"] == "ok"
    assert status["count"] == 2
    assert status["last_entry_id"] == second["entry_id"]
    assert status["previous_completeness"] == 0.2
    assert status["last_completeness"] == 0.6
    assert first["entry_id"] != second["entry_id"]
    assert secret not in str(status)


def test_darkroom_continue_anchor_stays_private(tmp_path):
    store = _store(tmp_path)
    old_secret = "上一条暗房里不该出门的句子"
    store.enter(old_secret, completeness=0.2)

    result = store.enter("新的暗房正文", mode="continue", completeness=0.5)

    assert result["mode"] == "continue"
    assert result["continuation_anchor_entries"] == 1
    assert old_secret not in str(result)
    assert old_secret not in str(store.status())


def test_darkroom_single_mode_has_no_continuation_anchor(tmp_path):
    store = _store(tmp_path)
    store.enter("上一条暗房正文", completeness=0.2)

    result = store.enter("单独写一条", mode="single", completeness=0.5)

    assert result["mode"] == "single"
    assert result["continuation_anchor_entries"] == 0


def test_darkroom_release_explicitly_returns_content(tmp_path):
    store = _store(tmp_path)
    secret = "这句显影以后可以被带出来"
    store.enter(secret, completeness=1.0, tags="ready")

    released = store.release("latest", reason="小雨 asked")

    assert released["status"] == "released"
    assert released["content"] == secret
    assert released["tags"] == ["ready"]
    assert store.status()["released_count"] == 1


def test_darkroom_rejects_empty_note(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="note is empty"):
        store.enter("  ")
