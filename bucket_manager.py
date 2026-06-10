# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets.
# 记忆桶的增删改查、多维索引搜索、激活更新。
#
# Core design:
# 核心逻辑：
#   - Each bucket = one Markdown file (YAML frontmatter + body)
#     每个记忆桶 = 一个 Markdown 文件
#   - Storage by type: permanent / dynamic / archive
#     存储按类型分目录
#   - Multi-dimensional soft index: domain + valence/arousal + fuzzy text
#     多维软索引：主题域 + 情感坐标 + 文本模糊匹配
#   - Search strategy: domain pre-filter → weighted multi-dim ranking
#     搜索策略：主题域预筛 → 多维加权精排
#   - Emotion coordinates based on Russell circumplex model:
#     情感坐标基于环形情感模型（Russell circumplex）：
#       valence (0~1): 0=negative → 1=positive
#       arousal (0~1): 0=calm → 1=excited
#
# Depended on by: server.py, decay_engine.py
# 被谁依赖：server.py, decay_engine.py
# ============================================================

import os
import math
import logging
import re
import shutil
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter
import jieba
from rapidfuzz import fuzz

from utils import generate_bucket_id, sanitize_name, safe_path, now_iso, strip_affect_anchor, strip_wikilinks

logger = logging.getLogger("ombre_brain.bucket")


class BucketManager:
    """
    Memory bucket manager — entry point for all bucket CRUD operations.
    Buckets are stored as Markdown files with YAML frontmatter for metadata
    and body for content. Natively compatible with Obsidian browsing/editing.
    记忆桶管理器 —— 所有桶的 CRUD 操作入口。
    桶以 Markdown 文件存储，YAML frontmatter 存元数据，正文存内容。
    天然兼容 Obsidian 直接浏览和编辑。
    """

    def __init__(self, config: dict):
        # --- Read storage paths from config / 从配置中读取存储路径 ---
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.tombstone_dir = os.path.join(self.base_dir, ".tombstones")
        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)

        # --- Wikilink config / 双链配置 ---
        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "我们", "你们", "他们", "然后", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        # --- Search scoring weights / 检索权重配置 ---
        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 1.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 1.0)  # Added to allow better content-based matching during merge

    # ---------------------------------------------------------
    # Create a new bucket
    # 创建新桶
    # Write content and metadata into a .md file
    # 将内容和元数据写入一个 .md 文件
    # ---------------------------------------------------------
    async def create(
        self,
        content: str,
        tags: list[str] = None,
        importance: int = 5,
        domain: list[str] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        pinned: bool = False,
        protected: bool = False,
        bucket_id: str = None,
        source: str = None,
        created: str = None,
        last_active: str = None,
        updated_at: str = None,
        anchor: bool = False,
        resolved: bool = False,
        digested: bool = False,
        confidence: float | None = None,
        period: str | None = None,
        date: str | None = None,
        extra_metadata: dict | None = None,
    ) -> str:
        """
        Create a new memory bucket, return bucket ID.
        创建一个新的记忆桶，返回桶 ID。

        pinned/protected=True: bucket won't be merged, decayed, or have importance changed.
        Importance is locked to 10 for pinned/protected buckets.
        pinned/protected 桶不参与合并与衰减，importance 强制锁定为 10。
        """
        bucket_id = bucket_id or generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id
        domain = domain or ["未分类"]
        tags = tags or []
        linked_content = content  # wikilink injection disabled; LLM adds [[]] via prompt
        created_at = created or now_iso()
        last_active_at = last_active or created_at
        updated_at_value = updated_at or created_at

        # --- Pinned/protected buckets: lock importance to 10 ---
        # --- 钉选/保护桶：importance 强制锁定为 10 ---
        if pinned or protected:
            importance = 10

        # --- Build YAML frontmatter metadata / 构建元数据 ---
        metadata = {
            "id": bucket_id,
            "name": bucket_name,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type,
            "created": created_at,
            "last_active": last_active_at,
            "updated_at": updated_at_value,
            "activation_count": 0,
        }
        if confidence is not None:
            metadata["confidence"] = max(0.0, min(1.0, float(confidence)))
        if period:
            metadata["period"] = str(period)
        if date:
            metadata["date"] = str(date)
        if pinned:
            metadata["pinned"] = True
        if protected:
            metadata["protected"] = True
        if anchor:
            metadata["anchor"] = True
        if resolved:
            metadata["resolved"] = True
        if digested:
            metadata["digested"] = True
        if source:
            metadata["source"] = source
        if extra_metadata:
            reserved = set(metadata.keys()) | {"content"}
            for key, value in extra_metadata.items():
                if key in reserved or value is None:
                    continue
                metadata[str(key)] = value

        # --- Assemble Markdown file (frontmatter + body) ---
        # --- 组装 Markdown 文件 ---
        post = frontmatter.Post(linked_content, **metadata)

        # --- Choose directory by type + primary domain ---
        # --- 按类型 + 主题域选择存储目录 ---
        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
            if pinned and bucket_type != "permanent":
                metadata["type"] = "permanent"
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        else:
            type_dir = self.dynamic_dir
        if bucket_type == "feel":
            primary_domain = "沉淀物"  # feel subfolder name
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)

        # --- Filename: readable_name_bucketID.md (Obsidian friendly) ---
        # --- 文件名：可读名称_桶ID.md ---
        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket file / 写入桶文件失败: {file_path}: {e}")
            raise

        logger.info(
            f"Created bucket / 创建记忆桶: {bucket_id} ({bucket_name}) → {primary_domain}/"
            + (" [PINNED]" if pinned else "") + (" [PROTECTED]" if protected else "")
        )
        return bucket_id

    # ---------------------------------------------------------
    # Read bucket content
    # 读取桶内容
    # Returns {"id", "metadata", "content", "path"} or None
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        """
        Read a single bucket by ID.
        根据 ID 读取单个桶。
        """
        if not bucket_id or not isinstance(bucket_id, str):
            return None
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    # ---------------------------------------------------------
    # Move bucket between directories
    # 在目录间移动桶文件
    # ---------------------------------------------------------
    def _move_bucket(self, file_path: str, target_type_dir: str, domain: list[str] = None) -> str:
        """
        Move a bucket file to a new type directory, preserving domain subfolder.
        Returns new file path.
        """
        primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            os.rename(file_path, new_path)
            logger.info(f"Moved bucket / 移动记忆桶: {filename} → {target_dir}/")
        return new_path

    # ---------------------------------------------------------
    # Update bucket
    # 更新桶
    # Supports: content, tags, importance, valence, arousal, name, resolved
    # ---------------------------------------------------------
    async def update(self, bucket_id: str, **kwargs) -> bool:
        """
        Update bucket content or metadata fields.
        更新桶的内容或元数据字段。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for update / 加载桶失败: {file_path}: {e}")
            return False

        # --- Pinned/protected buckets: lock importance to 10, ignore importance changes ---
        # --- 钉选/保护桶：importance 不可修改，强制保持 10 ---
        is_pinned = post.get("pinned", False) or post.get("protected", False)
        if is_pinned:
            kwargs.pop("importance", None)  # silently ignore importance update

        # --- Update only fields that were passed in / 只改传入的字段 ---
        if "content" in kwargs:
            post.content = kwargs["content"]  # wikilink injection disabled; LLM adds [[]] via prompt
        if "tags" in kwargs:
            post["tags"] = kwargs["tags"]
        if "importance" in kwargs:
            post["importance"] = max(1, min(10, int(kwargs["importance"])))
        if "domain" in kwargs:
            post["domain"] = kwargs["domain"]
        if "valence" in kwargs:
            post["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            post["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "name" in kwargs:
            post["name"] = sanitize_name(kwargs["name"])
        if "resolved" in kwargs:
            post["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            post["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                post["importance"] = 10  # pinned → lock importance to 10
        if "anchor" in kwargs:
            post["anchor"] = bool(kwargs["anchor"])
        if "digested" in kwargs:
            post["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            post["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))
        if "source" in kwargs:
            post["source"] = str(kwargs["source"])
        if "confidence" in kwargs:
            post["confidence"] = max(0.0, min(1.0, float(kwargs["confidence"])))
        if "period" in kwargs:
            post["period"] = str(kwargs["period"])
        if "date" in kwargs:
            post["date"] = str(kwargs["date"])
        if "comments" in kwargs:
            post["comments"] = kwargs["comments"] if isinstance(kwargs["comments"], list) else []
        if "comment_count" in kwargs:
            post["comment_count"] = max(0, int(kwargs["comment_count"]))
        if "active" in kwargs:
            post["active"] = bool(kwargs["active"])
        if "deprecated" in kwargs:
            post["deprecated"] = bool(kwargs["deprecated"])
        if "profile_kind" in kwargs:
            post["profile_kind"] = str(kwargs["profile_kind"])
        if "subject" in kwargs:
            post["subject"] = str(kwargs["subject"])
        if "predicate" in kwargs:
            post["predicate"] = str(kwargs["predicate"])
        if "object" in kwargs:
            post["object"] = str(kwargs["object"])
        if "evidence" in kwargs:
            post["evidence"] = kwargs["evidence"] if isinstance(kwargs["evidence"], list) else []
        if "source_bucket_ids" in kwargs:
            post["source_bucket_ids"] = kwargs["source_bucket_ids"] if isinstance(kwargs["source_bucket_ids"], list) else []
        if "source_persona_event_ids" in kwargs:
            post["source_persona_event_ids"] = (
                kwargs["source_persona_event_ids"] if isinstance(kwargs["source_persona_event_ids"], list) else []
            )

        # --- Auto-refresh content update time and activation time ---
        # --- 自动刷新内容更新时间与激活时间 ---
        post["updated_at"] = kwargs.get("updated_at") or now_iso()
        post["last_active"] = kwargs.get("last_active") or now_iso()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket update / 写入桶更新失败: {file_path}: {e}")
            return False

        # --- Auto-move: pinned → permanent/ ---
        # --- 自动移动：钉选 → permanent/ ---
        domain = post.get("domain", ["未分类"])
        if kwargs.get("pinned") and post.get("type") != "permanent":
            post["type"] = "permanent"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            self._move_bucket(file_path, self.permanent_dir, domain)

        logger.info(f"Updated bucket / 更新记忆桶: {bucket_id}")
        return True

    async def add_comment(
        self,
        bucket_id: str,
        content: str,
        *,
        author: str = "Haven",
        kind: str = "comment",
        valence: float | None = None,
        arousal: float | None = None,
        source: str | None = None,
        created: str | None = None,
        touch: bool = True,
    ) -> Optional[dict]:
        """
        Append a ring/comment to an existing bucket without changing its body.
        给已有桶追加年轮，不改正文。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path or not content or not str(content).strip():
            return None

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for comment / 加载评论目标失败: {file_path}: {e}")
            return None

        comments = post.get("comments", [])
        if not isinstance(comments, list):
            comments = []

        now = now_iso()
        created_at = str(created or now).strip() or now
        entry = {
            "id": generate_bucket_id(),
            "created": created_at,
            "author": str(author or "Haven"),
            "kind": str(kind or "comment"),
            "content": str(content).strip(),
        }
        if source:
            entry["source"] = str(source)
        if valence is not None:
            entry["valence"] = max(0.0, min(1.0, float(valence)))
            if entry["kind"] == "feel":
                post["model_valence"] = entry["valence"]
        if arousal is not None:
            entry["arousal"] = max(0.0, min(1.0, float(arousal)))

        comments.append(entry)
        post["comments"] = comments
        post["comment_count"] = len(comments)
        post["updated_at"] = now
        if touch:
            post["last_active"] = now
            post["activation_count"] = post.get("activation_count", 0) + 1

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket comment / 写入桶评论失败: {file_path}: {e}")
            return None

        if touch:
            current_time = self._parse_iso_datetime(post.get("created", post.get("last_active", "")))
            if current_time is None:
                current_time = datetime.now(timezone.utc).replace(tzinfo=None)
            await self._time_ripple(bucket_id, current_time)

        logger.info(f"Added bucket comment / 已追加年轮: {bucket_id}#{entry['id']}")
        return entry

    async def delete_comment(
        self,
        bucket_id: str,
        comment_id: str,
        *,
        allowed_author: str | None = None,
        allowed_source: str | None = None,
    ) -> dict:
        file_path = self._find_bucket_file(bucket_id)
        if not file_path or not comment_id:
            return {"status": "not_found"}

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for comment delete / 加载评论删除目标失败: {file_path}: {e}")
            return {"status": "not_found"}

        comments = post.get("comments", [])
        if not isinstance(comments, list):
            return {"status": "not_found"}

        kept = []
        target = None
        for comment in comments:
            if isinstance(comment, dict) and str(comment.get("id") or "") == str(comment_id):
                target = comment
            else:
                kept.append(comment)

        if target is None:
            return {"status": "not_found"}
        if allowed_author is not None and target.get("author") != allowed_author:
            return {"status": "forbidden", "comment": target}
        if allowed_source is not None and target.get("source") != allowed_source:
            return {"status": "forbidden", "comment": target}

        post["comments"] = kept
        post["comment_count"] = len(kept)
        post["updated_at"] = now_iso()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to delete bucket comment / 删除桶评论失败: {file_path}: {e}")
            return {"status": "failed", "comment": target}

        logger.info(f"Deleted bucket comment / 已删除年轮: {bucket_id}#{comment_id}")
        return {"status": "deleted", "comment": target}

    # ---------------------------------------------------------
    # Wikilink injection — DISABLED
    # 自动添加 Obsidian 双链 — 已禁用
    # Now handled by LLM prompts (Gemini adds [[]] for proper nouns)
    # 现在由 LLM prompt 处理（Gemini 对人名/地名/专有名词加 [[]]）
    # ---------------------------------------------------------
    # def _apply_wikilinks(self, content, tags, domain, name): ...
    # def _collect_wikilink_keywords(self, content, tags, domain, name): ...
    # def _normalize_keywords(self, keywords): ...
    # def _extract_auto_keywords(self, content): ...

    # ---------------------------------------------------------
    # Delete bucket
    # 删除桶
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str) -> bool:
        """
        Delete a memory bucket file.
        删除指定的记忆桶文件。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            tombstone = self._build_tombstone(bucket_id, file_path)
            os.remove(file_path)
            self._write_tombstone(tombstone)
        except OSError as e:
            logger.error(f"Failed to delete bucket file / 删除桶文件失败: {file_path}: {e}")
            return False

        logger.info(f"Deleted bucket / 删除记忆桶: {bucket_id}")
        return True

    def _build_tombstone(self, bucket_id: str, file_path: str) -> dict:
        deleted_at = now_iso()
        tombstone = {
            "id": bucket_id,
            "title": bucket_id,
            "type": "archived",
            "domain": ["deleted"],
            "tags": ["deleted"],
            "content": "",
            "valence": 0.5,
            "arousal": 0.5,
            "importance": 1,
            "pinned": False,
            "resolved": True,
            "digested": True,
            "activation_count": 0,
            "created": deleted_at,
            "last_active": deleted_at,
            "updated_at": deleted_at,
            "source": "deleted",
            "deleted_at": deleted_at,
        }
        try:
            post = frontmatter.load(file_path)
            tombstone.update(
                {
                    "title": post.get("name", bucket_id),
                    "type": post.get("type", "archived"),
                    "domain": post.get("domain", ["deleted"]) or ["deleted"],
                    "created": post.get("created", deleted_at),
                }
            )
        except Exception:
            pass
        return tombstone

    def _write_tombstone(self, tombstone: dict) -> None:
        os.makedirs(self.tombstone_dir, exist_ok=True)
        path = safe_path(self.tombstone_dir, f"{tombstone['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tombstone, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------
    # Touch bucket (refresh activation time + increment count)
    # 触碰桶（刷新激活时间 + 累加激活次数）
    # Called on every recall hit; affects decay score.
    # 每次检索命中时调用，影响衰减得分。
    # ---------------------------------------------------------
    async def touch(self, bucket_id: str) -> None:
        """
        Update a bucket's last activation time and count.
        Also triggers time ripple: nearby memories get a slight activation boost.
        更新桶的最后激活时间和激活次数。
        同时触发时间涟漪：时间上相邻的记忆轻微唤醒。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return

        try:
            post = frontmatter.load(file_path)
            post["last_active"] = now_iso()
            post["activation_count"] = post.get("activation_count", 0) + 1

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            # --- Time ripple: boost nearby memories within ±48h ---
            # --- 时间涟漪：±48小时内的记忆轻微唤醒 ---
            current_time = self._parse_iso_datetime(post.get("created", post.get("last_active", "")))
            if current_time is None:
                current_time = datetime.now(timezone.utc).replace(tzinfo=None)
            await self._time_ripple(bucket_id, current_time)
        except Exception as e:
            logger.warning(f"Failed to touch bucket / 触碰桶失败: {bucket_id}: {e}")

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        """
        Slightly boost activation_count of buckets created/activated near the reference time.
        轻微提升时间相邻桶的激活次数（+0.3），不改 last_active 避免递归唤醒。
        Max 5 buckets rippled per touch to bound I/O.
        """
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return

        rippled = 0
        max_ripple = 5
        for bucket in all_buckets:
            if rippled >= max_ripple:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            # Skip pinned/permanent/feel
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue

            created_str = meta.get("created", meta.get("last_active", ""))
            created = self._parse_iso_datetime(created_str)
            if created is None:
                continue
            delta_hours = abs((reference_time - created).total_seconds()) / 3600

            if delta_hours <= hours:
                # Boost activation_count by 0.3 (fractional), don't change last_active
                file_path = self._find_bucket_file(bucket["id"])
                if not file_path:
                    continue
                try:
                    post = frontmatter.load(file_path)
                    current_count = post.get("activation_count", 1)
                    # Store as float for fractional increments; calculate_score handles it
                    post["activation_count"] = round(current_count + 0.3, 1)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(frontmatter.dumps(post))
                    rippled += 1
                except Exception:
                    continue

    # ---------------------------------------------------------
    # Multi-dimensional search (core feature)
    # 多维搜索（核心功能）
    #
    # Strategy: domain pre-filter → weighted multi-dim ranking
    # 策略：主题域预筛 → 多维加权精排
    #
    # Ranking formula:
    #   total = topic(×w_topic) + emotion(×w_emotion)
    #           + time(×w_time) + importance(×w_importance)
    #
    # Per-dimension scores (normalized to 0~1):
    #   topic     = rapidfuzz weighted match (name/tags/domain/body)
    #   emotion   = 1 - Euclidean distance (query v/a vs bucket v/a)
    #   time      = e^(-0.02 × days) (recent memories first)
    #   importance = importance / 10
    # ---------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list[str] = None,
        query_valence: float = None,
        query_arousal: float = None,
        include_archive: bool = True,
    ) -> list[dict]:
        """
        Multi-dimensional indexed search for memory buckets.
        多维索引搜索记忆桶。

        domain_filter: pre-filter by domain (None = search all)
        query_valence/arousal: emotion coordinates for resonance scoring
        """
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=include_archive)

        if not all_buckets:
            return []

        # --- Layer 1: domain pre-filter (fast scope reduction) ---
        # --- 第一层：主题域预筛（快速缩小范围）---
        if domain_filter:
            filter_set = {d.lower() for d in domain_filter}
            candidates = [
                b for b in all_buckets
                if {d.lower() for d in b["metadata"].get("domain", [])} & filter_set
            ]
            # Fall back to full search if pre-filter yields nothing
            # 预筛为空则回退全量搜索
            if not candidates:
                candidates = all_buckets
        else:
            candidates = all_buckets

        # --- Layer 2: weighted multi-dim ranking ---
        # --- 第二层：多维加权精排 ---
        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})

            try:
                # Dim 1: topic relevance (fuzzy text, 0~1)
                topic_score = self._calc_topic_score(query, bucket)

                # Dim 2: emotion resonance (coordinate distance, 0~1)
                emotion_score = self._calc_emotion_score(
                    query_valence, query_arousal, meta
                )

                # Dim 3: time proximity (exponential decay, 0~1)
                time_score = self._calc_time_score(meta)

                # Dim 4: importance (direct normalization)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                # --- Weighted sum / 加权求和 ---
                total = (
                    topic_score * self.w_topic
                    + emotion_score * self.w_emotion
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                # Normalize to 0~100 for readability
                weight_sum = self.w_topic + self.w_emotion + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0

                if normalized >= self.fuzzy_threshold:
                    # Resolved buckets get ranking penalty after thresholding.
                    # 已解决桶先按相关性过阈值，再在排序阶段降权。
                    if meta.get("resolved", False):
                        normalized *= 0.3
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(
                    f"Scoring failed for bucket {bucket.get('id', '?')} / "
                    f"桶评分失败: {e}"
                )
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ---------------------------------------------------------
    # Topic relevance sub-score:
    # name(×3) + domain(×2.5) + tags(×2) + body(×1)
    # 文本相关性子分：桶名(×3) + 主题域(×2.5) + 标签(×2) + 正文(×1)
    # ---------------------------------------------------------
    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        """
        Calculate text dimension relevance score (0~1).
        计算文本维度的相关性得分。
        """
        meta = bucket.get("metadata", {})
        searchable_content = strip_affect_anchor(strip_wikilinks(str(bucket.get("content", ""))))[:1000]

        short_cjk_query = self._short_cjk_topic_query(query)
        if short_cjk_query:
            return self._calc_short_cjk_topic_score(short_cjk_query, meta, searchable_content)

        name_score = fuzz.partial_ratio(query, meta.get("name", "")) * 3
        domain_score = (
            max(
                (fuzz.partial_ratio(query, d) for d in meta.get("domain", [])),
                default=0,
            )
            * 2.5
        )
        tag_score = (
            max(
                (fuzz.partial_ratio(query, tag) for tag in meta.get("tags", [])),
                default=0,
            )
            * 2
        )
        content_score = fuzz.partial_ratio(query, searchable_content) * self.content_weight

        return (name_score + domain_score + tag_score + content_score) / (100 * (3 + 2.5 + 2 + self.content_weight))

    @staticmethod
    def _short_cjk_topic_query(query: str) -> str:
        compact = re.sub(
            r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+",
            "",
            str(query or ""),
        )
        if re.fullmatch(r"[\u4e00-\u9fff]{1,3}", compact):
            return compact
        return ""

    def _calc_short_cjk_topic_score(self, query: str, meta: dict, searchable_content: str) -> float:
        def exact(value: object) -> int:
            return 100 if query in str(value or "") else 0

        name_score = exact(meta.get("name")) * 3
        domain_score = max((exact(d) for d in meta.get("domain", []) or []), default=0) * 2.5
        tag_score = max((exact(tag) for tag in meta.get("tags", []) or []), default=0) * 2
        content_score = exact(searchable_content) * self.content_weight
        weighted = name_score + domain_score + tag_score + content_score
        if weighted <= 0:
            return 0.0

        score = weighted / (100 * (3 + 2.5 + 2 + self.content_weight))
        if content_score:
            score = max(score, 0.36)
        if domain_score or tag_score:
            score = max(score, 0.45)
        if name_score:
            score = max(score, 0.50)
        return min(1.0, score)

    # ---------------------------------------------------------
    # Emotion resonance sub-score:
    # Based on Russell circumplex Euclidean distance
    # 情感共鸣子分：基于环形情感模型的欧氏距离
    # No emotion in query → neutral 0.5 (doesn't affect ranking)
    # ---------------------------------------------------------
    def _calc_emotion_score(
        self, q_valence: float, q_arousal: float, meta: dict
    ) -> float:
        """
        Calculate emotion resonance score (0~1, closer = higher).
        计算情感共鸣度（0~1，越近越高）。
        """
        if q_valence is None or q_arousal is None:
            return 0.5  # No emotion coordinates → neutral / 无情感坐标时给中性分

        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5

        # Euclidean distance, max sqrt(2) ≈ 1.414
        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    # ---------------------------------------------------------
    # Time proximity sub-score:
    # More recent activation → higher score
    # 时间亲近子分：距上次激活越近分越高
    # ---------------------------------------------------------
    def _calc_time_score(self, meta: dict) -> float:
        """
        Calculate time proximity score (0~1, more recent = higher).
        计算时间亲近度。
        """
        last_active_str = meta.get("last_active", meta.get("created", ""))
        last_active = self._parse_iso_datetime(last_active_str)
        if last_active is None:
            days = 30
        else:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            days = max(0.0, (now - last_active).total_seconds() / 86400)
        return math.exp(-0.02 * days)

    def _parse_iso_datetime(self, value) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    # ---------------------------------------------------------
    # List all buckets
    # 列出所有桶
    # ---------------------------------------------------------
    async def list_all(self, include_archive: bool = False) -> list[dict]:
        """
        Recursively walk directories (including domain subdirs), list all buckets.
        递归遍历目录（含域子目录），列出所有记忆桶。
        """
        buckets = []

        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    file_path = os.path.join(root, filename)
                    bucket = self._load_bucket(file_path)
                    if bucket:
                        buckets.append(bucket)

        return buckets

    # ---------------------------------------------------------
    # Statistics (counts per category + total size)
    # 统计信息（各分类桶数量 + 总体积）
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        """
        Return memory bucket statistics (including domain subdirs).
        返回记忆桶的统计数据。
        """
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }

        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        fpath = os.path.join(root, f)
                        try:
                            stats["total_size_kb"] += os.path.getsize(fpath) / 1024
                        except OSError:
                            pass
                        # Per-domain counts / 每个域的桶数量
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = stats["domains"].get(domain_name, 0) + 1

        return stats

    # ---------------------------------------------------------
    # Archive bucket (move from permanent/dynamic into archive)
    # 归档桶（从 permanent/dynamic 移入 archive）
    # Called by decay engine to simulate "forgetting"
    # 由衰减引擎调用，模拟"遗忘"
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str) -> bool:
        """
        Move a bucket into the archive directory (preserving domain subdirs).
        将指定桶移入归档目录（保留域子目录结构）。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            # Read once, get domain info and update type / 一次性读取
            post = frontmatter.load(file_path)
            domain = post.get("domain", ["未分类"])
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            archive_subdir = os.path.join(self.archive_dir, primary_domain)
            os.makedirs(archive_subdir, exist_ok=True)

            dest = safe_path(archive_subdir, os.path.basename(file_path))

            # Update type marker then move file / 更新类型标记后移动文件
            post["type"] = "archived"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            # Use shutil.move for cross-filesystem safety
            # 使用 shutil.move 保证跨文件系统安全
            shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(
                f"Failed to archive bucket / 归档桶失败: {bucket_id}: {e}"
            )
            return False

        logger.info(f"Archived bucket / 归档记忆桶: {bucket_id} → archive/{primary_domain}/")
        return True

    # ---------------------------------------------------------
    # Internal: find bucket file across all three directories
    # 内部：在三个目录中查找桶文件
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id: str) -> Optional[str]:
        """
        Recursively search permanent/dynamic/archive for a bucket file
        matching the given ID.
        在 permanent/dynamic/archive 中递归查找指定 ID 的桶文件。
        """
        if not bucket_id:
            return None
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    # Match by exact ID segment in filename
                    # 通过文件名中的 ID 片段精确匹配
                    name_part = fname[:-3]  # remove .md
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        return os.path.join(root, fname)
        return None

    # ---------------------------------------------------------
    # Internal: load bucket data from .md file
    # 内部：从 .md 文件加载桶数据
    # ---------------------------------------------------------
    def _load_bucket(self, file_path: str) -> Optional[dict]:
        """
        Parse a Markdown file and return structured bucket data.
        解析 Markdown 文件，返回桶的结构化数据。
        """
        try:
            raw = Path(file_path).read_text(encoding="utf-8")
            post = frontmatter.load(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": file_path,
                "content_start_line": _markdown_body_start_line(raw),
            }
        except Exception as e:
            logger.warning(
                f"Failed to load bucket file / 加载桶文件失败: {file_path}: {e}"
            )
            return None


def _markdown_body_start_line(text: str) -> int:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return 1
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            body_start = index + 1
            while body_start <= len(lines) and not lines[body_start - 1].strip():
                body_start += 1
            return max(1, body_start)
    return 1
