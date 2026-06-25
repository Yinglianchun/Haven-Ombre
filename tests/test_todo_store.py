from todo_store import TodoStore


def test_todo_store_sync_lists_open_items(tmp_path):
    store = TodoStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})

    store.sync_from_entries(
        [
            {
                "id": "todo_a",
                "bucket_id": "bucket_a",
                "moment_id": "moment_a",
                "section": "followup",
                "source_hash": "hash_a",
                "title": "待办桶",
                "date": "2026-06-25",
                "updated": "2026-06-25T10:00:00+08:00",
                "text": "修 VPS smoke",
            }
        ]
    )

    rows = store.list(status="open")

    assert [row["id"] for row in rows] == ["todo_a"]
    assert rows[0]["source_bucket_id"] == "bucket_a"
    assert rows[0]["text"] == "修 VPS smoke"


def test_todo_store_done_item_is_not_reopened_by_sync(tmp_path):
    store = TodoStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    entry = {
        "id": "todo_a",
        "bucket_id": "bucket_a",
        "moment_id": "moment_a",
        "section": "followup",
        "source_hash": "hash_a",
        "title": "待办桶",
        "date": "2026-06-25",
        "updated": "2026-06-25T10:00:00+08:00",
        "text": "修 VPS smoke",
    }

    store.sync_from_entries([entry])
    done = store.set_status("todo_a", "done", resolved_at="2026-06-25T11:00:00+08:00")
    store.sync_from_entries([entry])

    assert done["status"] == "done"
    assert done["resolved_at"] == "2026-06-25T11:00:00+08:00"
    assert store.list(status="open") == []
    assert store.get("todo_a")["status"] == "done"


def test_todo_store_missing_open_items_become_inactive(tmp_path):
    store = TodoStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    store.sync_from_entries(
        [
            {
                "id": "todo_a",
                "bucket_id": "bucket_a",
                "moment_id": "moment_a",
                "section": "followup",
                "source_hash": "hash_a",
                "title": "待办桶",
                "date": "2026-06-25",
                "updated": "2026-06-25T10:00:00+08:00",
                "text": "修 VPS smoke",
            }
        ]
    )

    store.sync_from_entries([])

    assert store.list(status="open") == []
    assert store.get("todo_a")["active"] is False
