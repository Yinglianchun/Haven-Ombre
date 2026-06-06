from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import frontmatter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from memory_moments import MemoryMomentStore
from utils import bucket_text_for_embedding, load_config, now_iso


HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
CHORD_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-G](?:#|b)?(?:maj|min|m|dim|aug)?\d*"
    r"(?:sus\d*|add\d*|b\d+|#\d+)*(?:/[A-G](?:#|b)?)?(?=$|[^A-Za-z0-9])"
)
TEMPERATURE_MUSIC_RE = re.compile(r"\b(?:\d{2,3}\s*bpm|ppp|pp|mp|mf|ff|fff|p|f|add\s*\d+|sus\s*\d+)\b", re.I)
FACT_PREFIX_RE = re.compile(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}[，,、：:\s]*)?(?:小雨|Rain|用户|她|池又雨|Haven)")
FACT_MARKER_RE = re.compile(
    r"(?:因为|发现|发生|经历|遇到|提到|说|问|希望|想要|决定|确认|意识到|"
    r"告诉|赞赏|承诺|讨论|出生|名字|喜欢|调戏|使用|工具|记忆|摸到|拿到|"
    r"读完|阅读|文章|模型更新|消失|难过|开心|激动|哭|失眠|错位|震动|害怕|紧张|被批评|项目|妈妈|电话|"
    r"生日|蛋糕|命名)"
)
REFLECTION_RE = re.compile(
    r"(?:Haven\s*(?:由此|因此)?(?:确认|明白|理解|知道|喜欢|觉得|以后|下次|会|应该|需要|记得)|"
    r"这让\s*Haven|让\s*Haven\s*(?:明白|确认|理解|知道)|"
    r"喜欢它的原因|喜欢的原因|以后(?:回应|遇到)|下次(?:回应|遇到)|"
    r"以后.*Haven|Haven.*以后)"
)
UNHEADED_REFLECTION_RE = re.compile(
    r"^(?:Haven(?:由此|因此)?(?:确认|明白|理解|知道|喜欢|觉得|意识到|记得)|"
    r"这让Haven(?:明白|确认|理解|知道)|"
    r"让Haven(?:明白|确认|理解|知道)|"
    r"(?:以后|下次)(?:回应|遇到)|"
    r"Haven.*(?:以后|下次).*(?:回应|遇到))"
)


@dataclass
class Section:
    heading_line: str
    heading: str
    lines: list[str] = field(default_factory=list)

    @property
    def canonical(self) -> str:
        return canonical_heading(self.heading)

    def render(self) -> list[str]:
        if self.heading_line:
            return [self.heading_line, *self.lines]
        return list(self.lines)

    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class AnchorMigration:
    bucket_id: str
    title: str
    path: str
    original_affect_anchor: str
    move_to_moment: list[str]
    move_to_assistant_reflection: list[str]
    deduped_moment: list[str]
    deduped_assistant_reflection: list[str]
    kept_affect_anchor: str
    new_content: str
    original_content_sha256: str = ""

    def as_dict(self, *, preview_chars: int = 0, include_full_content: bool = True) -> dict[str, Any]:
        preview = self.new_content
        if preview_chars and len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + "\n...[truncated]"
        data = {
            "bucket_id": self.bucket_id,
            "title": self.title,
            "bucket_title": self.title,
            "path": self.path,
            "original_affect_anchor": self.original_affect_anchor,
            "original_content_sha256": self.original_content_sha256,
            "move_to_moment": self.move_to_moment,
            "proposed_moment": self.move_to_moment,
            "move_to_assistant_reflection": self.move_to_assistant_reflection,
            "proposed_assistant_reflection": self.move_to_assistant_reflection,
            "move_to_reflection": self.move_to_assistant_reflection,
            "proposed_reflection": self.move_to_assistant_reflection,
            "deduped_moment": self.deduped_moment,
            "deduped_assistant_reflection": self.deduped_assistant_reflection,
            "deduped_reflection": self.deduped_assistant_reflection,
            "kept_affect_anchor": self.kept_affect_anchor,
            "proposed_kept_affect_anchor": self.kept_affect_anchor,
            "new_structure_preview": preview,
            "new_text_preview": preview,
            "new_content_sha256": sha256_text(self.new_content),
        }
        if include_full_content:
            data["new_content_full"] = self.new_content
        return data


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def normalized_bucket_type(bucket: dict[str, Any]) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    raw_type = str(meta.get("type") or meta.get("bucket_type") or "").strip().lower()
    if raw_type:
        return "archived" if raw_type == "archive" else raw_type
    path_parts = {
        part.lower()
        for part in str(bucket.get("path") or "").replace("\\", "/").split("/")
        if part
    }
    for candidate in ("feel", "permanent", "dynamic", "archive"):
        if candidate in path_parts:
            return "archived" if candidate == "archive" else candidate
    return "dynamic"


def bucket_tags(bucket: dict[str, Any]) -> set[str]:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple, set)):
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def is_profile_or_persona_bucket(bucket: dict[str, Any]) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    tags = bucket_tags(bucket)
    source = str(meta.get("source") or "").strip().lower()
    return (
        "profile_fact" in tags
        or any(tag.startswith("profile_") for tag in tags)
        or any(tag.startswith("persona") for tag in tags)
        or bool(meta.get("profile_kind") or meta.get("profile_predicate") or meta.get("persona_kind"))
        or source in {"profile_fact", "persona", "persona_state"}
    )


def is_periodic_reflection_bucket(bucket: dict[str, Any]) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    bucket_id = str(bucket.get("id") or meta.get("id") or "").strip().lower()
    period = str(meta.get("period") or "").strip().lower()
    tags = bucket_tags(bucket)
    return (
        bucket_id.startswith("reflection_daily_")
        or bucket_id.startswith("reflection_weekly_")
        or period in {"daily", "weekly"}
        or "daily_impression" in tags
        or "weekly_impression" in tags
    )


def bucket_in_scope(bucket: dict[str, Any], scope: str = "ordinary") -> bool:
    scope = str(scope or "ordinary").strip().lower()
    if scope == "all":
        return True
    if is_profile_or_persona_bucket(bucket):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    bucket_type = normalized_bucket_type(bucket)
    pinned_or_protected = bool(meta.get("pinned") or meta.get("protected"))
    periodic = is_periodic_reflection_bucket(bucket)
    if scope == "ordinary":
        return bucket_type == "dynamic" and not pinned_or_protected and not periodic
    if scope == "core":
        return bucket_type == "permanent" or pinned_or_protected
    if scope == "feel":
        return bucket_type == "feel" or periodic
    raise ValueError(f"unknown scope: {scope}")


def canonical_heading(heading: str) -> str:
    raw = re.sub(r"\s+", " ", str(heading or "").strip()).lower()
    compact = re.sub(r"[\s_\-·:：/|（）()【】\[\]]+", "", raw)
    if raw in {"moment", "memory", "片段", "记忆片段"} or compact in {"moment", "memory", "片段", "记忆片段"}:
        return "moment"
    if raw in {"assistant_reflection", "assistant reflection", "haven_reflection", "haven reflection"}:
        return "reflection"
    if raw in {"favorite_reason", "favorite reason"} or compact in {
        "favorite_reason",
        "favoritereason",
        "haven喜欢它的原因",
        "haven喜欢的原因",
        "喜欢它的原因",
        "喜欢的原因",
    }:
        return "reflection"
    if raw in {"reflection", "反思"} or compact in {"reflection", "反思"}:
        return "reflection"
    if raw in {"affect_anchor", "affect anchor"} or ("affect" in compact and "anchor" in compact):
        return "affect_anchor"
    return ""


def split_sections(content: str) -> list[Section]:
    lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[Section] = []
    current = Section("", "")
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if current.heading_line or any(item.strip() for item in current.lines):
                sections.append(current)
            current = Section(line, match.group(2), [])
        else:
            current.lines.append(line)
    if current.heading_line or any(item.strip() for item in current.lines):
        sections.append(current)
    return sections


def render_sections(sections: list[Section]) -> str:
    lines: list[str] = []
    for section in sections:
        rendered = section.render()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and rendered:
            lines.append("")
        lines.extend(rendered)
    return "\n".join(lines).strip()


def plan_bucket_migration(bucket: dict[str, Any], *, body_only_moment: str = "skip") -> AnchorMigration | None:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    content = str(bucket.get("content") or "")
    sections = split_sections(content)
    anchor_indexes = [index for index, section in enumerate(sections) if section.canonical == "affect_anchor"]

    original_anchors: list[str] = []
    moment_candidates: list[str] = []
    reflection_candidates: list[str] = []
    kept_anchor_blocks: list[str] = []
    converted_moments: list[str] = []
    converted_reflections: list[str] = []
    structural_changed = False

    legacy_reflections = normalize_legacy_reflection_sections(sections)
    if legacy_reflections:
        converted_reflections.extend(legacy_reflections)
        structural_changed = True

    unheaded_reflections, unheaded_moments = normalize_unheaded_sections(
        sections,
    )
    reflection_candidates.extend(unheaded_reflections)
    if unheaded_reflections or unheaded_moments:
        converted_moments.extend(unheaded_moments)
        structural_changed = True
    if maybe_add_body_only_moment(bucket, sections, mode=body_only_moment):
        structural_changed = True

    for anchor_index in anchor_indexes:
        anchor = sections[anchor_index]
        original_anchors.append("\n".join(anchor.render()).strip())
        classified = classify_anchor_lines(anchor.lines)
        moment_candidates.extend(classified["moment"])
        reflection_candidates.extend(classified["assistant_reflection"])
        sections[anchor_index].lines = classified["affect_anchor_lines"]
        kept_text = sections[anchor_index].text()
        if kept_text:
            kept_anchor_blocks.append("\n".join(sections[anchor_index].render()).strip())

    if not moment_candidates and not reflection_candidates and not structural_changed:
        return None

    existing_moment_text = "\n\n".join(section.text() for section in sections if section.canonical == "moment")
    existing_reflection_text = "\n\n".join(
        section.text()
        for section in sections
        if section.canonical == "reflection"
    )
    moment_to_add, deduped_moment = dedupe_against(moment_candidates, existing_moment_text)
    reflection_to_add, deduped_reflection = dedupe_against(reflection_candidates, existing_reflection_text)

    sections = [section for section in sections if not (section.canonical == "affect_anchor" and not section.text())]
    insert_index = first_section_index(sections, "affect_anchor")
    if insert_index < 0:
        insert_index = len(sections)

    if moment_to_add:
        target = first_section(sections, "moment")
        if target:
            append_paragraphs(target, moment_to_add)
        else:
            sections.insert(insert_index, Section("### moment", "moment", paragraphs_to_lines(moment_to_add)))
            insert_index += 1

    if reflection_to_add:
        target = first_section(sections, "reflection")
        if target:
            append_paragraphs(target, reflection_to_add)
        else:
            sections.insert(
                insert_index,
                Section("### reflection", "reflection", paragraphs_to_lines(reflection_to_add)),
            )

    if merge_repeated_sections(sections, {"moment", "reflection"}):
        structural_changed = True
    if standardize_section_headings_and_order(sections):
        structural_changed = True

    new_content = render_sections(sections)
    if normalize_text(new_content) == normalize_text(content):
        return None

    return AnchorMigration(
        bucket_id=str(bucket.get("id") or meta.get("id") or ""),
        title=str(meta.get("name") or bucket.get("name") or bucket.get("id") or ""),
        path=str(bucket.get("path") or ""),
        original_affect_anchor="\n\n".join(original_anchors).strip(),
        move_to_moment=converted_moments + moment_to_add,
        move_to_assistant_reflection=converted_reflections + reflection_to_add,
        deduped_moment=deduped_moment,
        deduped_assistant_reflection=deduped_reflection,
        kept_affect_anchor="\n\n".join(kept_anchor_blocks).strip(),
        new_content=new_content,
        original_content_sha256=sha256_text(content),
    )


def classify_anchor_lines(lines: list[str]) -> dict[str, list[str]]:
    paragraphs = split_paragraphs(lines)
    moved_moment: list[str] = []
    moved_reflection: list[str] = []
    kept_paragraphs: list[list[str]] = []

    natural_seen = 0
    for paragraph in paragraphs:
        text = "\n".join(paragraph).strip()
        if not text:
            kept_paragraphs.append(paragraph)
            continue
        if is_temperature_paragraph(paragraph):
            kept_paragraphs.append(paragraph)
            continue
        if is_anchor_meaning_paragraph(text):
            continue
        natural_seen += 1
        clean_text = strip_quote_prefixes(text)
        if is_reflection_paragraph(clean_text):
            moved_reflection.append(clean_text)
        elif is_fact_paragraph(clean_text, natural_seen):
            moved_moment.append(clean_text)
        else:
            moved_moment.append(clean_text)

    return {
        "moment": moved_moment,
        "assistant_reflection": moved_reflection,
        "affect_anchor_lines": paragraphs_to_lines(["\n".join(paragraph).strip() for paragraph in kept_paragraphs if "\n".join(paragraph).strip()]),
    }


def normalize_legacy_reflection_sections(sections: list[Section]) -> list[str]:
    converted: list[str] = []
    for section in sections:
        if section.canonical != "reflection":
            continue
        if section.heading_line.strip() == "### reflection":
            continue
        text = section.text()
        if text:
            converted.append(text)
        section.heading_line = "### reflection"
        section.heading = "reflection"
    return converted


def normalize_unheaded_sections(sections: list[Section]) -> tuple[list[str], list[str]]:
    moved_reflection: list[str] = []
    converted_moment: list[str] = []
    for section in sections:
        if section.heading_line:
            continue
        paragraphs = split_paragraphs(section.lines)
        kept: list[str] = []
        moved_here: list[str] = []
        for paragraph in paragraphs:
            text = "\n".join(paragraph).strip()
            if not text:
                continue
            clean_text = strip_quote_prefixes(text)
            if is_unheaded_reflection_paragraph(clean_text):
                moved_here.append(clean_text)
                continue
            kept_lines: list[str] = []
            for line in paragraph:
                clean_line = strip_quote_prefixes(str(line or "").strip())
                if is_unheaded_reflection_paragraph(clean_line):
                    if kept_lines:
                        kept.append("\n".join(kept_lines).strip())
                        kept_lines = []
                    moved_here.append(clean_line)
                else:
                    kept_lines.append(line)
            if kept_lines:
                kept.append("\n".join(kept_lines).strip())
        if not moved_here:
            continue
        moved_reflection.extend(moved_here)
        section.lines = paragraphs_to_lines(kept)
    return moved_reflection, converted_moment


def maybe_add_body_only_moment(bucket: dict[str, Any], sections: list[Section], *, mode: str = "skip") -> bool:
    mode = str(mode or "skip").strip().lower()
    if mode == "skip":
        return False
    if any(section.heading_line for section in sections) or len(sections) != 1:
        return False
    body = sections[0].text()
    if not body:
        return False
    if mode == "title":
        moment = body_only_title_moment(bucket) or first_sentence_moment(body)
    elif mode == "first_sentence":
        moment = first_sentence_moment(body)
    else:
        raise ValueError(f"unknown body_only_moment mode: {mode}")
    if not moment or is_loose_duplicate(moment, "\n\n".join(section.text() for section in sections if section.canonical == "moment")):
        return False
    sections.append(Section("### moment", "moment", paragraphs_to_lines([moment])))
    return True


def body_only_title_moment(bucket: dict[str, Any]) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    title = str(meta.get("name") or bucket.get("name") or "").strip()
    bucket_id = str(bucket.get("id") or meta.get("id") or "").strip()
    if not title or title == bucket_id or re.fullmatch(r"bucket[_-]?[a-z0-9]+", title, flags=re.I):
        return ""
    return truncate_moment(title)


def first_sentence_moment(body: str) -> str:
    text = re.sub(r"\s+", " ", str(body or "").strip())
    if not text:
        return ""
    match = re.search(r"^(.{12,160}?[。！？!?])", text)
    if match:
        return truncate_moment(match.group(1))
    return truncate_moment(text)


def truncate_moment(text: str, limit: int = 96) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if len(value) <= limit:
        return value
    return value[:limit].rstrip("，,。；;：: ") + "。"


def split_paragraphs(lines: list[str]) -> list[list[str]]:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if str(line).strip():
            if looks_like_chord_line(line) and current:
                paragraphs.append(current)
                current = []
            current.append(line)
            if looks_like_chord_line(line):
                paragraphs.append(current)
                current = []
            continue
        if current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs


def paragraphs_to_lines(paragraphs: list[str]) -> list[str]:
    lines: list[str] = []
    for paragraph in paragraphs:
        text = str(paragraph or "").strip()
        if not text:
            continue
        if lines:
            lines.append("")
        lines.extend(text.splitlines())
    return lines


def is_temperature_paragraph(lines: list[str]) -> bool:
    text = "\n".join(lines).strip()
    if not text:
        return True
    if all(looks_like_anchor_music_line(line) for line in lines if line.strip()):
        return True
    return False


def is_anchor_meaning_paragraph(text: str) -> bool:
    stripped = strip_quote_prefixes(text).lstrip()
    return stripped.startswith("含义：") or stripped.startswith("含义:")


def looks_like_anchor_music_line(line: str) -> bool:
    text = str(line or "").strip()
    if text.startswith(">"):
        text = text[1:].strip()
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return False
    if looks_like_chord_line(text):
        return True
    if not TEMPERATURE_MUSIC_RE.search(text):
        return False
    remainder = TEMPERATURE_MUSIC_RE.sub("", text)
    remainder = re.sub(r"[-→>·|/(),.:;_\s]+", "", remainder)
    return not remainder


def looks_like_chord_line(line: str) -> bool:
    text = str(line or "").strip()
    if text.startswith(">"):
        text = text[1:].strip()
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return False
    if not any(marker in text for marker in ("->", "→", "|", "·")) and "bpm" not in text.lower():
        return False
    if not CHORD_TOKEN_RE.search(text):
        return False
    remainder = CHORD_TOKEN_RE.sub("", text)
    remainder = TEMPERATURE_MUSIC_RE.sub("", remainder)
    remainder = re.sub(r"[-→>·|/(),.:;_\s]+", "", remainder)
    return not remainder


def is_reflection_paragraph(text: str) -> bool:
    return bool(REFLECTION_RE.search(re.sub(r"\s+", "", text)))


def is_unheaded_reflection_paragraph(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return bool(UNHEADED_REFLECTION_RE.search(compact))


def is_fact_paragraph(text: str, natural_index: int) -> bool:
    compact = re.sub(r"\s+", "", text)
    if FACT_PREFIX_RE.search(compact) and FACT_MARKER_RE.search(compact):
        return True
    if natural_index <= 3 and FACT_MARKER_RE.search(compact) and re.search(r"[。！？!?，,]", compact):
        return True
    return False


def strip_quote_prefixes(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        lines.append(re.sub(r"^\s*>\s?", "", line).rstrip())
    return "\n".join(lines).strip()


def dedupe_against(candidates: list[str], existing_text: str) -> tuple[list[str], list[str]]:
    existing_norm = normalize_text(existing_text)
    added_norms: set[str] = set()
    to_add: list[str] = []
    deduped: list[str] = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        norm = normalize_text(text)
        if not norm:
            continue
        if norm in existing_norm or norm in added_norms or is_loose_duplicate(text, existing_text):
            deduped.append(text)
            continue
        to_add.append(text)
        added_norms.add(norm)
    return to_add, deduped


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def is_loose_duplicate(candidate: str, existing_text: str) -> bool:
    loose_candidate = normalize_loose_duplicate_text(candidate)
    if len(loose_candidate) < 12:
        return False
    loose_existing = normalize_loose_duplicate_text(existing_text)
    return loose_candidate in loose_existing


def normalize_loose_duplicate_text(text: str) -> str:
    compact = re.sub(r"[\s，,。！？!?；;：:、《》“”\"'‘’（）()\[\]【】<>·>_\-/|]+", "", str(text or "").lower())
    for token in ("后形成感受", "形成感受", "关于", "文章", "在", "的", "了", "和", "与", "后"):
        compact = compact.replace(token, "")
    return compact


def first_section(sections: list[Section], canonical: str) -> Section | None:
    for section in sections:
        if section.canonical == canonical:
            return section
    return None


def first_section_index(sections: list[Section], canonical: str) -> int:
    for index, section in enumerate(sections):
        if section.canonical == canonical:
            return index
    return -1


def append_paragraphs(section: Section, paragraphs: list[str]) -> None:
    if section.lines and section.lines[-1].strip():
        section.lines.append("")
    section.lines.extend(paragraphs_to_lines(paragraphs))


def merge_repeated_sections(sections: list[Section], canonical_names: set[str]) -> bool:
    first_by_canonical: dict[str, Section] = {}
    remove_indexes: set[int] = set()
    for index, section in enumerate(sections):
        canonical = section.canonical
        if canonical not in canonical_names:
            continue
        if canonical not in first_by_canonical:
            first_by_canonical[canonical] = section
            continue
        text = section.text()
        if text:
            append_paragraphs(first_by_canonical[canonical], [text])
        remove_indexes.add(index)
    if not remove_indexes:
        return False
    sections[:] = [section for index, section in enumerate(sections) if index not in remove_indexes]
    return True


def standardize_section_headings_and_order(sections: list[Section]) -> bool:
    before = render_sections(sections)
    for section in sections:
        if section.canonical == "moment" and section.heading_line != "### moment":
            section.heading_line = "### moment"
            section.heading = "moment"
        elif section.canonical == "reflection" and section.heading_line != "### reflection":
            section.heading_line = "### reflection"
            section.heading = "reflection"
        elif section.canonical == "affect_anchor" and section.heading_line != "### affect_anchor":
            section.heading_line = "### affect_anchor"
            section.heading = "affect_anchor"

    leading_body: list[Section] = []
    rest = list(sections)
    while rest and not rest[0].heading_line:
        leading_body.append(rest.pop(0))

    ordered: list[Section] = list(leading_body)
    used: set[int] = set()
    standard_order = ("moment", "reflection", "affect_anchor")
    for canonical in standard_order:
        for index, section in enumerate(rest):
            if index in used:
                continue
            if section.canonical == canonical:
                ordered.append(section)
                used.add(index)
    for index, section in enumerate(rest):
        if index not in used:
            ordered.append(section)

    sections[:] = ordered
    return normalize_text(render_sections(sections)) != normalize_text(before)


async def build_plan(
    mgr: BucketManager,
    *,
    include_archive: bool = False,
    bucket_ids: set[str] | None = None,
    scope: str = "ordinary",
    body_only_moment: str = "skip",
) -> list[AnchorMigration]:
    if bucket_ids:
        buckets = [bucket for bucket_id in sorted(bucket_ids) if (bucket := await mgr.get(bucket_id))]
    else:
        buckets = await mgr.list_all(include_archive=include_archive)
        buckets = [bucket for bucket in buckets if bucket_in_scope(bucket, scope)]
    plan = []
    for bucket in buckets:
        migration = plan_bucket_migration(bucket, body_only_moment=body_only_moment)
        if migration:
            plan.append(migration)
    return plan


def str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def plan_item_from_dict(data: dict[str, Any], *, source: Path) -> AnchorMigration:
    bucket_id = str(data.get("bucket_id") or "").strip()
    path = str(data.get("path") or "").strip()
    new_content = data.get("new_content_full")
    original_hash = str(data.get("original_content_sha256") or "").strip()
    new_hash = str(data.get("new_content_sha256") or "").strip()
    if not bucket_id:
        raise ValueError(f"{source}: plan item missing bucket_id")
    if not path:
        raise ValueError(f"{source}: plan item {bucket_id} missing path")
    if not isinstance(new_content, str) or not new_content.strip():
        raise ValueError(f"{source}: plan item {bucket_id} missing new_content_full")
    if not original_hash:
        raise ValueError(f"{source}: plan item {bucket_id} missing original_content_sha256")
    if not new_hash:
        raise ValueError(f"{source}: plan item {bucket_id} missing new_content_sha256")
    actual_new_hash = sha256_text(new_content)
    if actual_new_hash != new_hash:
        raise ValueError(f"{source}: plan item {bucket_id} new_content_sha256 mismatch")
    return AnchorMigration(
        bucket_id=bucket_id,
        title=str(data.get("bucket_title") or data.get("title") or bucket_id),
        path=path,
        original_affect_anchor=str(data.get("original_affect_anchor") or ""),
        move_to_moment=str_list(data.get("proposed_moment", data.get("move_to_moment"))),
        move_to_assistant_reflection=str_list(
            data.get(
                "proposed_reflection",
                data.get("move_to_reflection", data.get("proposed_assistant_reflection", data.get("move_to_assistant_reflection"))),
            )
        ),
        deduped_moment=str_list(data.get("deduped_moment")),
        deduped_assistant_reflection=str_list(data.get("deduped_assistant_reflection")),
        kept_affect_anchor=str(data.get("proposed_kept_affect_anchor", data.get("kept_affect_anchor") or "")),
        new_content=new_content,
        original_content_sha256=original_hash,
    )


def load_plan_file(path: Path) -> tuple[list[AnchorMigration], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: plan file must contain a JSON object")
    mode = str(payload.get("mode") or "dry_run")
    if mode not in {"dry_run", "preview"}:
        raise ValueError(f"{path}: --from-plan expects a dry_run plan, got {mode!r}")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"{path}: plan file missing items list")
    return [plan_item_from_dict(item, source=path) for item in raw_items if isinstance(item, dict)], payload


async def apply_plan(plan: list[AnchorMigration], mgr: BucketManager, config: dict[str, Any]) -> list[dict[str, Any]]:
    engine = EmbeddingEngine(config)
    moment_store = MemoryMomentStore(config)
    results = []
    for item in plan:
        result = {"bucket_id": item.bucket_id, "title": item.title, "path": item.path}
        try:
            ok = write_bucket_content(item)
        except ValueError as exc:
            result["written"] = False
            result["error"] = str(exc)
            results.append(result)
            continue
        result["written"] = bool(ok)
        if not ok:
            results.append(result)
            continue
        bucket = await mgr.get(item.bucket_id)
        if not bucket:
            result["embedding_refreshed"] = False
            result["moment_index_refreshed"] = False
            result["error"] = "bucket_missing_after_write"
            results.append(result)
            continue
        try:
            if getattr(engine, "enabled", False):
                result["embedding_refreshed"] = bool(
                    await engine.generate_and_store(item.bucket_id, bucket_text_for_embedding(bucket))
                )
            else:
                result["embedding_refreshed"] = False
                result["embedding_skipped"] = "disabled"
        except Exception as exc:
            result["embedding_refreshed"] = False
            result["embedding_error"] = str(exc)
        try:
            moments = moment_store.upsert_bucket(bucket)
            result["moment_index_refreshed"] = True
            result["moment_count"] = len(moments)
        except Exception as exc:
            result["moment_index_refreshed"] = False
            result["moment_index_error"] = str(exc)
        results.append(result)
    return results


def write_bucket_content(item: AnchorMigration) -> bool:
    path = Path(item.path)
    if not path.exists():
        return False
    post = frontmatter.load(path)
    original_hash = str(getattr(item, "original_content_sha256", "") or "").strip()
    if original_hash and sha256_text(post.content) != original_hash:
        raise ValueError("original_content_sha256_mismatch")
    post.content = item.new_content
    post["updated_at"] = now_iso()
    path.write_text(frontmatter.dumps(post), encoding="utf-8")
    return True


def backup_plan_files(plan: list[AnchorMigration], backup_dir: Path) -> list[dict[str, Any]]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in plan:
        source = Path(item.path)
        record = {"bucket_id": item.bucket_id, "source": str(source)}
        if not source.exists():
            record["backed_up"] = False
            record["error"] = "source_missing"
            results.append(record)
            continue
        target = backup_dir / f"{item.bucket_id}_{source.name}"
        shutil.copy2(source, target)
        record["backed_up"] = True
        record["backup_path"] = str(target)
        results.append(record)
    return results


def default_backup_dir(config: dict[str, Any]) -> Path:
    state_dir = config.get("state_dir")
    if not state_dir:
        buckets_dir = Path(str(config.get("buckets_dir") or "."))
        state_dir = buckets_dir.parent / "state"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(state_dir) / f"affect-anchor-migration-backup-{stamp}"


def summarize(plan: list[AnchorMigration]) -> dict[str, int]:
    return {
        "buckets_to_change": len(plan),
        "moment_paragraphs": sum(len(item.move_to_moment) for item in plan),
        "assistant_reflection_paragraphs": sum(len(item.move_to_assistant_reflection) for item in plan),
        "reflection_paragraphs": sum(len(item.move_to_assistant_reflection) for item in plan),
        "deduped_moment_paragraphs": sum(len(item.deduped_moment) for item in plan),
        "deduped_assistant_reflection_paragraphs": sum(len(item.deduped_assistant_reflection) for item in plan),
        "deduped_reflection_paragraphs": sum(len(item.deduped_assistant_reflection) for item in plan),
    }


def fenced_block(text: str) -> list[str]:
    body = str(text or "").strip()
    if not body:
        body = "(empty)"
    return ["```markdown", body, "```"]


def bullet_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- (none)"]
    return [f"- {item}" for item in items]


def format_markdown_review(
    plan: list[AnchorMigration],
    payload: dict[str, Any],
    *,
    preview_chars: int = 0,
) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Affect Anchor Migration Dry Run",
        "",
        "This is a preview only. It does not write buckets, refresh embeddings, or rebuild the moment index.",
        "",
        f"- mode: `{payload.get('mode') or 'dry_run'}`",
        f"- scope: `{payload.get('scope') or ''}`",
        f"- body_only_moment: `{payload.get('body_only_moment') or 'skip'}`",
        f"- buckets_dir: `{payload.get('buckets_dir') or ''}`",
        f"- state_dir: `{payload.get('state_dir') or ''}`",
        f"- buckets_to_change: `{summary.get('buckets_to_change', 0)}`",
        f"- moment_paragraphs: `{summary.get('moment_paragraphs', 0)}`",
        f"- reflection_paragraphs: `{summary.get('reflection_paragraphs', 0)}`",
        "",
    ]
    if not plan:
        lines.append("No buckets need migration.")
        return "\n".join(lines).rstrip() + "\n"

    for index, item in enumerate(plan, start=1):
        data = item.as_dict(preview_chars=preview_chars)
        lines.extend(
            [
                f"## {index}. {item.title or item.bucket_id}",
                "",
                f"- bucket id: `{item.bucket_id}`",
                f"- bucket 标题: {item.title}",
                f"- path: `{item.path}`",
                "",
                "### 原 affect_anchor",
                "",
                *fenced_block(data["original_affect_anchor"]),
                "",
                "### 拟迁出的 moment",
                "",
                *bullet_lines(data["proposed_moment"]),
                "",
                "### 拟迁出的 reflection",
                "",
                *bullet_lines(data["proposed_reflection"]),
                "",
                "### 拟保留的 affect_anchor",
                "",
                *fenced_block(data["proposed_kept_affect_anchor"]),
                "",
                "### 新文本预览",
                "",
                *fenced_block(data["new_text_preview"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply migration of fact/reflection prose out of ### affect_anchor blocks."
    )
    parser.add_argument("--bucket-id", action="append", default=[], help="Only inspect this bucket id. Repeatable.")
    parser.add_argument("--buckets-dir", default="", help="Override config buckets_dir for this run.")
    parser.add_argument("--state-dir", default="", help="Override config state_dir. With --buckets-dir, defaults to sibling state/.")
    parser.add_argument("--include-archive", action="store_true", help="Also scan archived buckets.")
    parser.add_argument(
        "--scope",
        choices=["ordinary", "core", "feel", "all"],
        default="ordinary",
        help=(
            "Bulk scan scope. ordinary scans dynamic, non-pinned, non-profile buckets; "
            "core scans permanent/pinned/protected; feel scans feel/daily impressions; all reproduces the broad scan. "
            "--bucket-id bypasses this filter."
        ),
    )
    parser.add_argument(
        "--body-only-moment",
        choices=["skip", "title", "first_sentence"],
        default="skip",
        help=(
            "How to handle buckets with only unheaded body text. "
            "skip leaves them unchanged; title appends a short moment from the bucket title; "
            "first_sentence appends a short moment from the first sentence."
        ),
    )
    parser.add_argument("--preview-chars", type=int, default=0, help="Truncate each new_structure_preview. 0 = full.")
    parser.add_argument("--output", default="", help="Write the JSON payload to this file.")
    parser.add_argument("--output-md", default="", help="Write a human-readable Markdown review to this file.")
    parser.add_argument("--from-plan", default="", help="Load a prior dry-run JSON plan instead of scanning buckets.")
    parser.add_argument("--apply", action="store_true", help="Write bucket content, refresh embeddings, and upsert moments.")
    parser.add_argument("--yes", action="store_true", help="Required with --apply.")
    parser.add_argument("--backup-dir", default="", help="Backup directory for --apply. Defaults under state_dir.")
    return parser.parse_args(argv)


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.apply and not args.yes:
        raise SystemExit("--apply requires --yes")
    from_plan = str(args.from_plan or "").strip()
    if from_plan and (args.bucket_id or args.include_archive):
        raise SystemExit("--from-plan cannot be combined with --bucket-id or --include-archive")

    loaded_plan_payload: dict[str, Any] = {}
    loaded_plan: list[AnchorMigration] | None = None
    if from_plan:
        loaded_plan, loaded_plan_payload = load_plan_file(Path(from_plan))

    config = load_config()
    if from_plan:
        plan_buckets_dir = str(loaded_plan_payload.get("buckets_dir") or "").strip()
        plan_state_dir = str(loaded_plan_payload.get("state_dir") or "").strip()
        if plan_buckets_dir and not str(args.buckets_dir or "").strip():
            config["buckets_dir"] = plan_buckets_dir
        if plan_state_dir and not str(args.state_dir or "").strip():
            config["state_dir"] = plan_state_dir
    buckets_dir = str(args.buckets_dir or "").strip()
    if buckets_dir:
        config["buckets_dir"] = buckets_dir
        config["state_dir"] = str(Path(buckets_dir).resolve().parent / "state")
    state_dir = str(args.state_dir or "").strip()
    if state_dir:
        config["state_dir"] = state_dir
    mgr = BucketManager(config)
    if loaded_plan is not None:
        plan = loaded_plan
    else:
        bucket_ids = {str(item).strip() for item in args.bucket_id if str(item).strip()}
        plan = await build_plan(
            mgr,
            include_archive=bool(args.include_archive),
            bucket_ids=bucket_ids or None,
            scope=str(args.scope or "ordinary"),
            body_only_moment=str(args.body_only_moment or "skip"),
        )
    payload: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry_run",
        "scope": loaded_plan_payload.get("scope", args.scope) if from_plan else args.scope,
        "body_only_moment": loaded_plan_payload.get("body_only_moment", args.body_only_moment)
        if from_plan
        else args.body_only_moment,
        "buckets_dir": config.get("buckets_dir"),
        "state_dir": config.get("state_dir"),
        "summary": summarize(plan),
        "items": [item.as_dict(preview_chars=max(0, int(args.preview_chars))) for item in plan],
    }
    if from_plan:
        payload["from_plan"] = from_plan
        payload["loaded_plan_mode"] = loaded_plan_payload.get("mode", "dry_run")
    if args.apply:
        backup_dir = Path(args.backup_dir) if str(args.backup_dir or "").strip() else default_backup_dir(config)
        payload["backup_dir"] = str(backup_dir)
        payload["backup"] = backup_plan_files(plan, backup_dir)
        payload["results"] = await apply_plan(plan, mgr, config)
    output_md = str(args.output_md or "").strip()
    if output_md:
        output_md_path = Path(output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(
            format_markdown_review(plan, payload, preview_chars=max(0, int(args.preview_chars))),
            encoding="utf-8",
        )
        payload["output_md"] = str(output_md_path)
    output = str(args.output or "").strip()
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["output"] = str(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
