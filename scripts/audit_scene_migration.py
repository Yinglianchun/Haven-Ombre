#!/usr/bin/env python3
"""Read-only inventory for migrating legacy Ombre buckets to Scene semantics.

The report deliberately does not copy bucket prose. It classifies structure and
flags manual-review groups; it never edits the source tree or promotes a legacy
``### moment`` to a Scene automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import frontmatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_moments import parse_bucket_moments


PROFILE_FACT_MARKERS = {"profile_fact", "profile-fact", "profile index", "profile_index"}
DAILY_MARKERS = {
    "daily_impression",
    "daily-impression",
    "relationship_weather",
    "relationship-weather",
    "日印象",
    "关系天气",
}
WHISPER_MARKERS = {"whisper", "私语", "碎碎念"}


def resolve_buckets_dir(value: str) -> Path:
    source = Path(value).expanduser().resolve()
    nested = source / "buckets"
    if nested.is_dir():
        source = nested
    if not source.is_dir():
        raise FileNotFoundError(f"buckets directory does not exist: {source}")
    return source


def iter_bucket_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".tombstones" not in path.relative_to(root).parts
    )


def normalized_markers(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [value]
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def top_level(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else "root"


def length_band(chars: int) -> str:
    if chars < 80:
        return "short_lt_80"
    if chars <= 240:
        return "medium_80_240"
    return "long_gt_240"


def evidence_refs(metadata: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw = metadata.get("evidence")
    if isinstance(raw, dict):
        raw = [raw]
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        bucket_id = str(item.get("bucket_id") or item.get("id") or "").strip()
        if bucket_id:
            rows.append(
                {
                    "bucket_id": bucket_id,
                    "moment_id": str(item.get("moment_id") or "").strip(),
                }
            )
    for bucket_key, moment_key in (
        ("evidence_bucket_id", "evidence_moment_id"),
        ("source_bucket_id", "source_moment_id"),
    ):
        bucket_id = str(metadata.get(bucket_key) or "").strip()
        if bucket_id:
            rows.append(
                {
                    "bucket_id": bucket_id,
                    "moment_id": str(metadata.get(moment_key) or "").strip(),
                }
            )
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["bucket_id"], row["moment_id"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def classify_bucket(path: Path, root: Path) -> dict[str, Any]:
    post = frontmatter.load(path)
    metadata = dict(post.metadata)
    content = str(post.content or "")
    bucket_id = str(metadata.get("id") or path.stem).strip()
    tags = normalized_markers(metadata.get("tags"))
    domains = normalized_markers(metadata.get("domain"))
    markers = tags | domains

    bucket = {
        "id": bucket_id,
        "content": content,
        "metadata": metadata,
        "path": str(path),
    }
    moments = parse_bucket_moments(bucket)
    content_moments = [item for item in moments if item.get("source") == "content"]
    section_counts = Counter(str(item.get("section") or "") for item in content_moments)
    comments = metadata.get("comments") if isinstance(metadata.get("comments"), list) else []

    is_profile_fact = bool(
        markers & PROFILE_FACT_MARKERS
        or metadata.get("profile_kind")
        or truthy(metadata.get("profile_fact"))
    )
    is_daily = bool(
        markers & DAILY_MARKERS
        or str(metadata.get("memory_value_source") or "").strip().lower() in DAILY_MARKERS
        or str(metadata.get("bucket_kind") or "").strip().lower() in DAILY_MARKERS
    )
    is_whisper = bool(
        top_level(path, root) == "feel"
        or markers & WHISPER_MARKERS
        or truthy(metadata.get("whisper"))
    )

    scene_count = section_counts.get("scene", 0)
    moment_count = section_counts.get("moment", 0)
    body_count = section_counts.get("body", 0)
    legacy_siblings = sum(
        count
        for section, count in section_counts.items()
        if section not in {"scene", "body"}
    )

    if is_daily:
        policy = "exclude_daily_impression"
    elif is_profile_fact:
        policy = "profile_fact_index_only"
    elif is_whisper:
        policy = "preserve_whisper"
    elif scene_count > 1:
        policy = "multiple_canonical_scenes_error"
    elif scene_count == 1 and (legacy_siblings or body_count):
        policy = "canonical_scene_with_legacy_siblings_review"
    elif scene_count == 1:
        policy = "canonical_scene"
    elif moment_count > 1:
        policy = "multiple_legacy_moments_review"
    elif moment_count == 1:
        policy = "single_legacy_moment_review"
    elif body_count:
        policy = "legacy_body_review"
    else:
        policy = "metadata_or_unknown_review"

    section_lengths: dict[str, list[int]] = {}
    for item in content_moments:
        section = str(item.get("section") or "")
        section_lengths.setdefault(section, []).append(len(str(item.get("text") or "")))

    return {
        "bucket_id": bucket_id,
        "rel_path": path.relative_to(root).as_posix(),
        "top_level": top_level(path, root),
        "policy": policy,
        "section_counts": dict(sorted(section_counts.items())),
        "section_lengths": section_lengths,
        "moment_sections": {
            str(item.get("moment_id") or ""): str(item.get("section") or "")
            for item in content_moments
            if str(item.get("moment_id") or "")
        },
        "comment_count": len(comments),
        "comment_kinds": [str(item.get("kind") or "comment") for item in comments if isinstance(item, dict)],
        "is_profile_fact": is_profile_fact,
        "is_daily": is_daily,
        "is_whisper": is_whisper,
        "evidence_refs": evidence_refs(metadata) if is_profile_fact else [],
    }


def build_report(root: Path) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for path in iter_bucket_files(root):
        try:
            items.append(classify_bucket(path, root))
        except Exception as exc:
            parse_errors.append(
                {
                    "rel_path": path.relative_to(root).as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    policy_counts = Counter(item["policy"] for item in items)
    top_level_counts = Counter(item["top_level"] for item in items)
    section_bucket_counts: Counter[str] = Counter()
    section_instance_counts: Counter[str] = Counter()
    section_length_bands: dict[str, Counter[str]] = {}
    comment_kind_counts: Counter[str] = Counter()
    total_comments = 0

    for item in items:
        for section, count in item["section_counts"].items():
            section_bucket_counts[section] += 1
            section_instance_counts[section] += int(count)
        for section, lengths in item["section_lengths"].items():
            bands = section_length_bands.setdefault(section, Counter())
            for chars in lengths:
                bands[length_band(int(chars))] += 1
        total_comments += int(item["comment_count"])
        comment_kind_counts.update(item["comment_kinds"])

    by_id = {item["bucket_id"]: item for item in items if item["bucket_id"]}
    fact_items = [item for item in items if item["is_profile_fact"]]
    fact_refs = [ref for item in fact_items for ref in item["evidence_refs"]]
    existing_fact_refs = [ref for ref in fact_refs if ref["bucket_id"] in by_id]
    fact_target_policies = Counter(
        by_id[ref["bucket_id"]]["policy"]
        for ref in existing_fact_refs
    )
    canonical_fact_refs = sum(
        1
        for ref in existing_fact_refs
        if by_id[ref["bucket_id"]]["section_counts"].get("scene") == 1
    )
    fact_evidence_sections = Counter()
    for ref in existing_fact_refs:
        target = by_id[ref["bucket_id"]]
        moment_id = ref.get("moment_id") or ""
        if not moment_id:
            fact_evidence_sections["bucket_only"] += 1
        else:
            fact_evidence_sections[target["moment_sections"].get(moment_id, "missing_moment_id")] += 1

    return {
        "mode": "dry_run_read_only",
        "source_buckets_dir": str(root),
        "rules": {
            "legacy_moment_auto_promoted": False,
            "source_files_modified": False,
            "daily_impression_excluded": True,
            "whisper_preserved": True,
            "profile_fact_index_only": True,
        },
        "summary": {
            "markdown_files": len(iter_bucket_files(root)),
            "parsed_buckets": len(items),
            "parse_errors": len(parse_errors),
            "policy_counts": dict(sorted(policy_counts.items())),
            "top_level_counts": dict(sorted(top_level_counts.items())),
            "section_bucket_counts": dict(sorted(section_bucket_counts.items())),
            "section_instance_counts": dict(sorted(section_instance_counts.items())),
            "section_length_bands": {
                section: dict(sorted(bands.items()))
                for section, bands in sorted(section_length_bands.items())
            },
            "comment_count": total_comments,
            "comment_kind_counts": dict(sorted(comment_kind_counts.items())),
            "profile_fact_evidence": {
                "profile_fact_buckets": len(fact_items),
                "evidence_refs": len(fact_refs),
                "refs_to_existing_bucket": len(existing_fact_refs),
                "refs_to_canonical_scene": canonical_fact_refs,
                "target_policy_counts": dict(sorted(fact_target_policies.items())),
                "evidence_moment_sections": dict(sorted(fact_evidence_sections.items())),
            },
        },
        "parse_errors": parse_errors,
        "items": items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Snapshot root or buckets directory")
    parser.add_argument("--output", default="", help="Optional private JSON report path")
    parser.add_argument(
        "--print-items",
        action="store_true",
        help="Print bucket ids and relative paths as well as aggregate counts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_buckets_dir(args.source)
    report = build_report(root)
    printable = report if args.print_items else {key: value for key, value in report.items() if key != "items"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if report["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
