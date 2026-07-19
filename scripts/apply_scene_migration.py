#!/usr/bin/env python3
"""Apply explicitly selected exact-span Scene migration candidates.

This is an incremental, idempotent importer. It validates the current source
bucket file hash and the selected content hash before creating a new canonical
Scene. Existing legacy buckets are never edited or deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import frontmatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from utils import bucket_text_for_embedding, load_config


CONFIRMATION = "APPLY_VERIFIED_SCENES"
IMPORT_VERSION = "scene-migration-import-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(str(value).encode("utf-8"))


def load_preview(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "dry_run_exact_source_slice":
        raise ValueError("preview is not an exact-source-slice migration report")
    if not isinstance(payload.get("results"), list):
        raise ValueError("preview results are missing")
    return payload


def scene_id_for(result: dict) -> str:
    identity = "|".join(
        (
            str(result.get("source_bucket_id") or ""),
            str(result.get("source_section") or ""),
            str(result.get("content_sha256") or ""),
        )
    )
    return "scene_mig_" + sha256_text(identity)[:20]


def selected_results(preview: dict, candidate_ids: list[str]) -> list[dict]:
    wanted = {str(item or "").strip() for item in candidate_ids if str(item or "").strip()}
    by_id = {str(item.get("candidate_id") or ""): item for item in preview["results"]}
    if not wanted:
        raise ValueError("at least one --candidate-id is required")
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError("candidate id not found: " + ", ".join(missing))
    output = []
    for candidate_id in sorted(wanted):
        result = by_id[candidate_id]
        if str(result.get("verdict") or "") != "accept":
            raise ValueError(f"candidate is not accepted: {candidate_id}")
        content = str(result.get("content") or "")
        if not content.strip() or sha256_text(content) != str(result.get("content_sha256") or ""):
            raise ValueError(f"candidate content hash mismatch: {candidate_id}")
        if not bool(result.get("content_is_exact_source_slice")):
            raise ValueError(f"candidate is not marked as exact source slice: {candidate_id}")
        output.append(result)
    return output


def validate_source_file(target_root: Path, result: dict) -> tuple[Path, dict]:
    rel_path = Path(str(result.get("source_rel_path") or ""))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("invalid source_rel_path")
    path = (target_root / rel_path).resolve()
    try:
        path.relative_to(target_root.resolve())
    except ValueError as exc:
        raise ValueError("source path escapes target buckets directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"source bucket missing: {rel_path.as_posix()}")
    if sha256_bytes(path.read_bytes()) != str(result.get("source_file_sha256") or ""):
        raise ValueError(f"source bucket changed: {rel_path.as_posix()}")
    post = frontmatter.load(path)
    metadata = dict(post.metadata)
    if str(metadata.get("id") or "") != str(result.get("source_bucket_id") or ""):
        raise ValueError(f"source bucket id mismatch: {rel_path.as_posix()}")
    return path, metadata


def migration_action(target_root: Path, result: dict) -> dict:
    _, metadata = validate_source_file(target_root, result)
    raw_domain = metadata.get("domain")
    if isinstance(raw_domain, str):
        domain = [part.strip() for part in raw_domain.replace("|", ",").split(",") if part.strip()]
    elif isinstance(raw_domain, list):
        domain = [str(part).strip() for part in raw_domain if str(part).strip()]
    else:
        domain = []
    return {
        "candidate_id": str(result.get("candidate_id") or ""),
        "scene_id": scene_id_for(result),
        "content": str(result.get("content") or ""),
        "title": str(result.get("title") or metadata.get("name") or "").strip(),
        "scene_cues": [str(cue).strip() for cue in result.get("scene_cues", []) if str(cue).strip()][:4],
        "domain": domain or ["未分类"],
        "date": str(metadata.get("date") or "").strip(),
        "importance": max(1, min(10, int(metadata.get("importance") or 5))),
        "source_bucket_id": str(result.get("source_bucket_id") or ""),
        "source_rel_path": str(result.get("source_rel_path") or ""),
        "source_section": str(result.get("source_section") or ""),
        "source_moment_id": str(result.get("source_moment_id") or ""),
        "source_file_sha256": str(result.get("source_file_sha256") or ""),
        "source_text_sha256": str(result.get("source_text_sha256") or ""),
        "content_sha256": str(result.get("content_sha256") or ""),
        "selector_model": str(result.get("model") or ""),
    }


async def apply_actions(
    config: dict,
    actions: list[dict],
    *,
    reviewed_by: str,
    generate_embeddings: bool,
) -> list[dict]:
    bucket_mgr = BucketManager(config)
    embedding_engine = EmbeddingEngine(config) if generate_embeddings else None
    results: list[dict] = []
    for action in actions:
        existing = await bucket_mgr.get(action["scene_id"])
        if existing:
            existing_meta = existing.get("metadata", {}) if isinstance(existing.get("metadata"), dict) else {}
            same = (
                str(existing.get("content") or "") == action["content"]
                and str(existing_meta.get("migration_source_file_sha256") or "")
                == action["source_file_sha256"]
            )
            if not same:
                raise RuntimeError(f"existing Scene conflicts with import: {action['scene_id']}")
            results.append(
                {
                    "candidate_id": action["candidate_id"],
                    "scene_id": action["scene_id"],
                    "status": "already_present",
                    "embedding_refreshed": False,
                }
            )
            continue
        await bucket_mgr.create(
            content=action["content"],
            tags=[],
            importance=action["importance"],
            domain=action["domain"],
            name=action["title"] or None,
            bucket_id=action["scene_id"],
            source="scene_migration",
            date=action["date"] or None,
            extra_metadata={
                "memory_value_source": "authored_scene",
                "write_contract": "scene-migration-v1",
                "scene_cues": action["scene_cues"] or None,
                "migration_import_version": IMPORT_VERSION,
                "migration_reviewed_by": reviewed_by,
                "migration_source_bucket_id": action["source_bucket_id"],
                "migration_source_rel_path": action["source_rel_path"],
                "migration_source_section": action["source_section"],
                "migration_source_moment_id": action["source_moment_id"] or None,
                "migration_source_file_sha256": action["source_file_sha256"],
                "migration_source_text_sha256": action["source_text_sha256"],
                "migration_content_sha256": action["content_sha256"],
                "migration_selector_model": action["selector_model"],
            },
        )
        embedding_refreshed = False
        if embedding_engine is not None:
            created = await bucket_mgr.get(action["scene_id"])
            if created:
                embedding_refreshed = await embedding_engine.generate_and_store(
                    action["scene_id"],
                    bucket_text_for_embedding(created),
                )
        results.append(
            {
                "candidate_id": action["candidate_id"],
                "scene_id": action["scene_id"],
                "status": "created",
                "embedding_refreshed": embedding_refreshed,
            }
        )
    return results


async def run(args: argparse.Namespace) -> dict:
    preview_path = Path(args.preview).expanduser().resolve()
    target_root = Path(args.target_buckets_dir).expanduser().resolve()
    if not target_root.is_dir():
        raise FileNotFoundError(f"target buckets directory does not exist: {target_root}")
    preview = load_preview(preview_path)
    chosen = selected_results(preview, args.candidate_id)
    actions = [migration_action(target_root, result) for result in chosen]
    payload: dict[str, Any] = {
        "mode": "dry_run" if args.confirm != CONFIRMATION else "apply",
        "target_buckets_dir": str(target_root),
        "source_preview": str(preview_path),
        "source_files_modified": False,
        "actions": [
            {key: value for key, value in action.items() if key != "content"}
            for action in actions
        ],
    }
    if args.confirm != CONFIRMATION:
        payload["confirmation_required"] = CONFIRMATION
        payload["results"] = []
        return payload
    config = load_config()
    config["buckets_dir"] = str(target_root)
    config["state_dir"] = str(target_root.parent / "state")
    payload["results"] = await apply_actions(
        config,
        actions,
        reviewed_by=str(args.reviewed_by or "xiaoyu").strip()[:80],
        generate_embeddings=not args.skip_embeddings,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--target-buckets-dir", required=True)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--confirm", default="")
    parser.add_argument("--reviewed-by", default="xiaoyu")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = asyncio.run(run(args))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
