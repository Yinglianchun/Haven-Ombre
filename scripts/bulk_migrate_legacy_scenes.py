#!/usr/bin/env python3
"""Deterministically migrate curated legacy Ombre buckets into canonical Scenes.

The migration rule is intentionally mechanical and model-free:

* if unheaded body text exists, Scene content is body + reflection sections;
* otherwise Scene content is moment sections + reflection sections;
* original and every other legacy section are discarded from the Scene body.

Legacy source files are preserved.  A successful active Scene suppresses its
source from ordinary recall through ``migration_source_bucket_id``.  Re-running
the script is idempotent, and a newer deterministic Scene supersedes an older
active migration Scene only after the new embedding has been stored.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any

import frontmatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from memory_layers import (
    LAYER_AFFECT_CONTEXT,
    LAYER_ARCHIVE,
    LAYER_DREAM,
    LAYER_PROFILE_INDEX,
    LAYER_RELATIONSHIP_WEATHER,
    LAYER_SOURCE_RECORD,
    infer_bucket_layer,
)
from memory_moments import HEADING_RE, _canonical_section
from memory_moments import MemoryMomentStore
from utils import bucket_text_for_embedding, load_config, normalize_scene_cues, now_iso


CONFIRMATION = "APPLY_DETERMINISTIC_SCENES"
IMPORT_VERSION = "scene-migration-import-v2"
RULE_VERSION = "unheaded-body-or-moment-plus-reflection-v1"
EXCLUDED_LAYERS = {
    LAYER_ARCHIVE,
    LAYER_DREAM,
    LAYER_SOURCE_RECORD,
    LAYER_PROFILE_INDEX,
    LAYER_RELATIONSHIP_WEATHER,
    LAYER_AFFECT_CONTEXT,
}
SYSTEM_CUE_MARKERS = {
    "commitment",
    "emotional_echo",
    "haven_favorite",
    "project_event",
    "relationship_event",
    "reflection",
    "todo",
    "trust",
    "wish",
}
PROVENANCE_FIELDS = (
    "author_actor_id",
    "source_type",
    "source_id",
    "source_chunk_id",
    "source_range",
    "diary_id",
    "fidelity",
    "event_date",
    "written_at",
    "source_hash",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(str(value).encode("utf-8"))


def metadata_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date_type):
        return value.isoformat()
    return str(value or "").strip()


def migration_event_date(metadata: dict) -> tuple[str, str]:
    explicit = metadata_text(metadata.get("date") or metadata.get("event_date"))
    if explicit:
        match = re.match(r"^\d{4}-\d{2}-\d{2}", explicit)
        return (match.group(0) if match else explicit), "source_date"
    created = metadata_text(metadata.get("created"))
    match = re.match(r"^\d{4}-\d{2}-\d{2}", created)
    if match:
        return match.group(0), "source_created"
    return "", ""


def resolve_buckets_dir(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if (root / "buckets").is_dir():
        root = root / "buckets"
    if not root.is_dir():
        raise FileNotFoundError(f"buckets directory does not exist: {root}")
    return root


def iter_bucket_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".tombstones" not in path.relative_to(root).parts
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def exact_content_sections(content: str) -> list[dict[str, Any]]:
    """Return exact body/legacy section spans without their Markdown headings."""
    text = str(content or "")
    headings: list[dict[str, Any]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        match = HEADING_RE.match(line)
        if match:
            headings.append(
                {
                    "start": offset,
                    "content_start": offset + len(raw_line),
                    "heading": match.group(2),
                    "section": _canonical_section(match.group(2)),
                }
            )
        offset += len(raw_line)

    sections: list[dict[str, Any]] = []
    first_heading = headings[0]["start"] if headings else len(text)
    body_start, body_end = _trim_span(text, 0, first_heading)
    if body_start < body_end:
        body = text[body_start:body_end]
        sections.append(
            {
                "section": "body",
                "index": 0,
                "start": body_start,
                "end": body_end,
                "text": body,
                "sha256": sha256_text(body),
            }
        )

    section_indexes: Counter[str] = Counter()
    for index, heading in enumerate(headings):
        section = str(heading.get("section") or "")
        if not section:
            continue
        end = headings[index + 1]["start"] if index + 1 < len(headings) else len(text)
        start, end = _trim_span(text, int(heading["content_start"]), int(end))
        if start >= end:
            continue
        section_index = section_indexes[section]
        section_indexes[section] += 1
        value = text[start:end]
        sections.append(
            {
                "section": section,
                "index": section_index,
                "start": start,
                "end": end,
                "text": value,
                "sha256": sha256_text(value),
            }
        )
    return sections


def selected_scene_parts(sections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    bodies = [item for item in sections if item["section"] == "body" and item["text"].strip()]
    moments = [item for item in sections if item["section"] == "moment" and item["text"].strip()]
    reflections = [item for item in sections if item["section"] == "reflection" and item["text"].strip()]
    if bodies:
        return [*bodies, *reflections], "body_plus_reflection" if reflections else "body_only"
    if moments:
        return [*moments, *reflections], "moment_plus_reflection" if reflections else "moment_only"
    return [], "no_body_or_moment"


def _bucket_from_post(path: Path, post: frontmatter.Post) -> dict[str, Any]:
    return {
        "id": str(post.get("id") or path.stem),
        "content": str(post.content or ""),
        "metadata": dict(post.metadata),
        "path": str(path),
    }


def exclusion_reason(path: Path, root: Path, post: frontmatter.Post, sections: list[dict]) -> str:
    metadata = dict(post.metadata)
    if path.relative_to(root).parts[0].lower() == "archive":
        return "archive"
    if str(metadata.get("memory_value_source") or "").strip() == "authored_scene":
        return "canonical_scene"
    if str(metadata.get("write_contract") or "").strip().lower().startswith("close-window-shadow"):
        return "window_shadow"
    if any(item["section"] == "scene" for item in sections):
        return "canonical_scene"
    layer = infer_bucket_layer(_bucket_from_post(path, post))
    if layer in EXCLUDED_LAYERS:
        return layer
    return ""


def _domain(metadata: dict) -> list[str]:
    raw = metadata.get("domain")
    if isinstance(raw, str):
        result = [part.strip() for part in raw.replace("|", ",").split(",") if part.strip()]
    elif isinstance(raw, (list, tuple, set)):
        result = [str(part).strip() for part in raw if str(part).strip()]
    else:
        result = []
    return result or ["未分类"]


def _source_tags(metadata: dict) -> list[str]:
    raw = metadata.get("tags")
    if isinstance(raw, str):
        values = re.split(r"[,|｜\r\n]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _scene_cues(metadata: dict) -> list[str]:
    candidates: list[str] = []
    raw_cues = metadata.get("scene_cues")
    if isinstance(raw_cues, str):
        candidates.extend(re.split(r"[\r\n|｜]+", raw_cues))
    elif isinstance(raw_cues, (list, tuple, set)):
        candidates.extend(str(item) for item in raw_cues)
    for tag in _source_tags(metadata):
        key = tag.strip().lower()
        if key in SYSTEM_CUE_MARKERS or key.startswith(("profile_", "flavor_")):
            continue
        candidates.append(tag)
    return normalize_scene_cues(candidates, limit=8, max_chars=80)


def _bucket_type(metadata: dict) -> str:
    return "permanent" if str(metadata.get("type") or "").strip().lower() == "permanent" else "dynamic"


def _activation_count(metadata: dict) -> float:
    try:
        return max(0.0, float(metadata.get("activation_count") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def scene_id_for(source_bucket_id: str, content_sha256: str) -> str:
    identity = f"{source_bucket_id}|{RULE_VERSION}|{content_sha256}"
    return "scene_mig2_" + sha256_text(identity)[:20]


def _public_span(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": item["section"],
        "index": item["index"],
        "start": item["start"],
        "end": item["end"],
        "chars": len(item["text"]),
        "content_sha256": item["sha256"],
    }


def build_plan(root: Path) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    parse_errors: list[dict[str, str]] = []
    all_files = iter_bucket_files(root)
    for path in all_files:
        try:
            raw_bytes = path.read_bytes()
            post = frontmatter.loads(raw_bytes.decode("utf-8-sig"))
            metadata = dict(post.metadata)
            sections = exact_content_sections(str(post.content or ""))
            reason = exclusion_reason(path, root, post, sections)
            if reason:
                excluded[reason] += 1
                continue
            selected, selection_rule = selected_scene_parts(sections)
            if not selected:
                excluded[selection_rule] += 1
                continue
            content = "\n\n".join(str(item["text"]) for item in selected).strip()
            if not content:
                excluded["empty_composite"] += 1
                continue
            source_id = str(metadata.get("id") or path.stem).strip()
            content_hash = sha256_text(content)
            source_created = metadata_text(metadata.get("created"))
            source_last_active = metadata_text(metadata.get("last_active")) or source_created
            event_date, event_date_source = migration_event_date(metadata)
            provenance = {
                key: metadata[key]
                for key in PROVENANCE_FIELDS
                if key in metadata and metadata[key] not in (None, "", [], {})
            }
            actions.append(
                {
                    "scene_id": scene_id_for(source_id, content_hash),
                    "source_bucket_id": source_id,
                    "source_rel_path": path.relative_to(root).as_posix(),
                    "source_file_sha256": sha256_bytes(raw_bytes),
                    "source_content_sha256": sha256_text(str(post.content or "")),
                    "content": content,
                    "content_sha256": content_hash,
                    "selection_rule": selection_rule,
                    "source_spans": [_public_span(item) for item in selected],
                    "discarded_sections": sorted(
                        {
                            item["section"]
                            for item in sections
                            if item not in selected and item["section"] != "body"
                        }
                    ),
                    "title": str(metadata.get("name") or "").strip(),
                    "scene_cues": _scene_cues(metadata),
                    "source_tags": _source_tags(metadata),
                    "domain": _domain(metadata),
                    "bucket_type": _bucket_type(metadata),
                    "pinned": bool(metadata.get("pinned")),
                    "protected": bool(metadata.get("protected")),
                    "anchor": bool(metadata.get("anchor") or metadata.get("bucket_anchor")),
                    "importance": max(1, min(10, int(metadata.get("importance") or 5))),
                    "date": event_date,
                    "event_date_source": event_date_source,
                    "source_created": source_created,
                    "source_last_active": source_last_active,
                    "source_activation_count": _activation_count(metadata),
                    "provenance": provenance,
                }
            )
        except Exception as exc:
            parse_errors.append(
                {
                    "source_rel_path": path.relative_to(root).as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    actions.sort(key=lambda item: (item["source_rel_path"], item["scene_id"]))
    digest_rows = [
        {
            "scene_id": item["scene_id"],
            "source_bucket_id": item["source_bucket_id"],
            "source_rel_path": item["source_rel_path"],
            "source_file_sha256": item["source_file_sha256"],
            "content_sha256": item["content_sha256"],
            "selection_rule": item["selection_rule"],
            "source_spans": item["source_spans"],
            "source_activation_count": item["source_activation_count"],
        }
        for item in actions
    ]
    plan_sha256 = sha256_text(json.dumps(digest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return {
        "mode": "deterministic_scene_migration_plan",
        "rule_version": RULE_VERSION,
        "source_buckets_dir": str(root),
        "source_files_modified": False,
        "plan_sha256": plan_sha256,
        "summary": {
            "markdown_files": len(all_files),
            "actions": len(actions),
            "parse_errors": len(parse_errors),
            "selection_rules": dict(sorted(Counter(item["selection_rule"] for item in actions).items())),
            "excluded": dict(sorted(excluded.items())),
            "discarded_section_buckets": dict(
                sorted(
                    Counter(
                        section
                        for item in actions
                        for section in set(item["discarded_sections"])
                    ).items()
                )
            ),
        },
        "parse_errors": parse_errors,
        "actions": actions,
    }


def validate_action(root: Path, action: dict[str, Any]) -> tuple[Path, frontmatter.Post]:
    rel_path = Path(str(action["source_rel_path"]))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("invalid source path")
    path = (root / rel_path).resolve()
    path.relative_to(root.resolve())
    raw = path.read_bytes()
    if sha256_bytes(raw) != action["source_file_sha256"]:
        raise ValueError(f"source file changed: {rel_path.as_posix()}")
    post = frontmatter.loads(raw.decode("utf-8-sig"))
    if str(post.get("id") or path.stem).strip() != action["source_bucket_id"]:
        raise ValueError(f"source bucket id changed: {rel_path.as_posix()}")
    return path, post


def _scene_extra_metadata(action: dict[str, Any], imported_at: str, reviewed_by: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "memory_value_source": "authored_scene",
        "write_contract": "scene-migration-v2",
        "scene_cues": action["scene_cues"] or None,
        "migration_import_version": IMPORT_VERSION,
        "migration_rule_version": RULE_VERSION,
        "migration_reviewed_by": reviewed_by,
        "migration_source_bucket_id": action["source_bucket_id"],
        "migration_source_rel_path": action["source_rel_path"],
        "migration_source_file_sha256": action["source_file_sha256"],
        "migration_source_content_sha256": action["source_content_sha256"],
        "migration_content_sha256": action["content_sha256"],
        "migration_selection_rule": action["selection_rule"],
        "migration_source_spans": action["source_spans"],
        "migration_discarded_sections": action["discarded_sections"] or None,
        "migration_source_tags": action["source_tags"] or None,
        "migration_source_created": action["source_created"] or None,
        "migration_source_last_active": action["source_last_active"] or None,
        "migration_source_activation_count": action["source_activation_count"],
        "migration_event_date_source": action["event_date_source"] or None,
        "migration_imported_at": imported_at,
    }
    metadata.update(action.get("provenance") or {})
    return metadata


def _write_metadata(path: Path, updates: dict[str, Any]) -> None:
    post = frontmatter.load(path)
    for key, value in updates.items():
        if value is None:
            post.metadata.pop(key, None)
        else:
            post[key] = value
    with path.open("w", encoding="utf-8") as handle:
        handle.write(frontmatter.dumps(post))


def _active_migration_predecessor_map(buckets: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for bucket in buckets:
        metadata = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        source_id = str(metadata.get("migration_source_bucket_id") or "").strip()
        if not source_id:
            continue
        if (
            str(metadata.get("memory_value_source") or "") != "authored_scene"
            or str(metadata.get("type") or "") == "archived"
            or metadata.get("resolved")
            or metadata.get("digested")
        ):
            continue
        output.setdefault(source_id, []).append(bucket)
    return output


async def apply_plan(
    config: dict,
    plan: dict[str, Any],
    *,
    reviewed_by: str,
    generate_embeddings: bool,
) -> list[dict[str, Any]]:
    root = Path(config["buckets_dir"]).resolve()
    bucket_mgr = BucketManager(config)
    embedding_engine = EmbeddingEngine(config) if generate_embeddings else None
    moment_store = MemoryMomentStore(config)
    existing_buckets = await bucket_mgr.list_all(include_archive=True)
    existing_by_id = {
        str(bucket.get("id") or ""): bucket
        for bucket in existing_buckets
        if str(bucket.get("id") or "")
    }
    predecessor_map = _active_migration_predecessor_map(existing_buckets)
    results: list[dict[str, Any]] = []
    for action in plan["actions"]:
        try:
            validate_action(root, action)
            existing = existing_by_id.get(action["scene_id"])
            if existing:
                existing_meta = existing.get("metadata", {}) if isinstance(existing.get("metadata"), dict) else {}
                if (
                    str(existing.get("content") or "") != action["content"]
                    or str(existing_meta.get("migration_source_bucket_id") or "") != action["source_bucket_id"]
                ):
                    raise RuntimeError(f"existing Scene conflicts with import: {action['scene_id']}")
                auto_decay_repair = bool(
                    str(existing_meta.get("type") or "") == "archived"
                    and str(existing_meta.get("migration_import_version") or "") == IMPORT_VERSION
                    and _activation_count(existing_meta) == 0.0
                    and float(action["source_activation_count"]) > 0.0
                    and not existing_meta.get("resolved")
                    and not existing_meta.get("digested")
                    and not existing_meta.get("deprecated")
                )
                if auto_decay_repair:
                    if not await bucket_mgr.activate(action["scene_id"]):
                        raise RuntimeError(f"failed to restore prematurely decayed Scene: {action['scene_id']}")
                    existing = await bucket_mgr.get(action["scene_id"])
                    if not existing:
                        raise RuntimeError(f"restored Scene is not readable: {action['scene_id']}")
                    existing_meta = (
                        existing.get("metadata", {}) if isinstance(existing.get("metadata"), dict) else {}
                    )
                    existing_by_id[action["scene_id"]] = existing
                intentionally_retired = bool(
                    str(existing_meta.get("type") or "") == "archived"
                    or existing_meta.get("resolved")
                    or existing_meta.get("digested")
                    or existing_meta.get("deprecated")
                    or existing_meta.get("active") is False
                ) and not bool(existing_meta.get("migration_embedding_failed"))
                if intentionally_retired:
                    results.append(
                        {
                            "scene_id": action["scene_id"],
                            "source_bucket_id": action["source_bucket_id"],
                            "status": "retired_existing",
                            "embedding_refreshed": False,
                            "superseded": [],
                        }
                    )
                    continue
                status = "auto_decay_repaired" if auto_decay_repair else "already_present"
            else:
                imported_at = now_iso()
                await bucket_mgr.create(
                    content=action["content"],
                    tags=[],
                    importance=action["importance"],
                    domain=action["domain"],
                    bucket_type=action["bucket_type"],
                    name=action["title"] or None,
                    pinned=action["pinned"],
                    protected=action["protected"],
                    anchor=action["anchor"],
                    bucket_id=action["scene_id"],
                    source="scene_migration",
                    date=action["date"] or None,
                    created=action["source_created"] or None,
                    last_active=action["source_last_active"] or action["source_created"] or None,
                    updated_at=imported_at,
                    extra_metadata=_scene_extra_metadata(action, imported_at, reviewed_by),
                )
                existing = await bucket_mgr.get(action["scene_id"])
                if existing:
                    existing_by_id[action["scene_id"]] = existing
                status = "created"
            if not existing:
                raise RuntimeError(f"Scene was not readable after create: {action['scene_id']}")

            embedding_refreshed: bool | None = None
            if embedding_engine is not None:
                embedding_ready = bool(await embedding_engine.get_embedding(action["scene_id"]))
                if not embedding_ready:
                    embedding_refreshed = await embedding_engine.generate_and_store(
                        action["scene_id"],
                        bucket_text_for_embedding(existing),
                    )
                    embedding_ready = bool(embedding_refreshed)
                else:
                    embedding_refreshed = False
                if not embedding_ready:
                    _write_metadata(
                        Path(existing["path"]),
                        {"active": False, "resolved": True, "migration_embedding_failed": True},
                    )
                    results.append(
                        {
                            "scene_id": action["scene_id"],
                            "source_bucket_id": action["source_bucket_id"],
                            "status": "embedding_failed_inactive",
                            "embedding_refreshed": False,
                            "superseded": [],
                        }
                    )
                    continue

            existing = await bucket_mgr.get(action["scene_id"])
            if existing:
                _write_metadata(
                    Path(existing["path"]),
                    {
                        "active": True,
                        "deprecated": False,
                        "resolved": False,
                        "activation_count": action["source_activation_count"],
                        "migration_embedding_failed": None,
                    },
                )
                existing = await bucket_mgr.get(action["scene_id"])
                if existing:
                    moment_store.upsert_bucket(existing)

            predecessors = [
                bucket
                for bucket in predecessor_map.get(action["source_bucket_id"], [])
                if str(bucket.get("id") or "") != action["scene_id"]
                and (
                    (
                        (bucket.get("metadata") or {}).get("active") is not False
                        and not (bucket.get("metadata") or {}).get("deprecated")
                    )
                    or str((bucket.get("metadata") or {}).get("superseded_by") or "")
                    == action["scene_id"]
                )
            ]
            superseded: list[str] = []
            for predecessor in predecessors:
                predecessor_id = str(predecessor.get("id") or "")
                _write_metadata(
                    Path(predecessor["path"]),
                    {
                        "active": False,
                        "deprecated": True,
                        "resolved": True,
                        "superseded_by": action["scene_id"],
                        "migration_superseded_at": now_iso(),
                    },
                )
                superseded.append(predecessor_id)
            results.append(
                {
                    "scene_id": action["scene_id"],
                    "source_bucket_id": action["source_bucket_id"],
                    "status": status,
                    "embedding_refreshed": embedding_refreshed,
                    "superseded": superseded,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "scene_id": action["scene_id"],
                    "source_bucket_id": action["source_bucket_id"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "embedding_refreshed": False,
                    "superseded": [],
                }
            )
    return results


def public_plan(plan: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
    if include_content:
        return plan
    return {
        **plan,
        "actions": [
            {key: value for key, value in action.items() if key != "content"}
            for action in plan["actions"]
        ],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_buckets_dir(args.source)
    plan = build_plan(root)
    applying = args.confirm == CONFIRMATION
    payload = public_plan(plan, include_content=bool(args.include_content))
    payload["mode"] = "apply" if applying else "dry_run"
    if not applying:
        payload["confirmation_required"] = CONFIRMATION
        return payload
    if plan["parse_errors"]:
        raise RuntimeError("refusing apply because the plan has parse errors")
    expected = str(args.expected_plan_sha256 or "").strip()
    if not expected or expected != plan["plan_sha256"]:
        raise RuntimeError("expected plan SHA256 is missing or does not match the current source tree")
    config = load_config()
    config["buckets_dir"] = str(root)
    config["state_dir"] = str(root.parent / "state")
    results = await apply_plan(
        config,
        plan,
        reviewed_by=str(args.reviewed_by or "xiaoyu").strip()[:80],
        generate_embeddings=not args.skip_embeddings,
    )
    payload["results"] = results
    payload["result_summary"] = dict(sorted(Counter(item["status"] for item in results).items()))
    payload["superseded_count"] = sum(len(item.get("superseded") or []) for item in results)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Snapshot root or buckets directory")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--expected-plan-sha256", default="")
    parser.add_argument("--reviewed-by", default="xiaoyu-deterministic-rule")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--include-content", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = asyncio.run(run(args))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 1 if payload.get("result_summary", {}).get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
