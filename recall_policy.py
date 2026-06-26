from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from memory_relevance import (
    EMOTIONAL_RECALL_STATE_TERMS,
    MemoryRelevanceOptions,
    content_terms_for_query,
    emotional_recall_plan,
    memory_relevance_options_from_config,
    query_has_facet,
    query_has_explicit_entity_marker,
    query_has_technical_recall_marker,
    recall_admission_decision,
    recall_topic_query,
)
from identity import identity_names


CONTEXT_ONLY_SECTIONS = frozenset({"affect_anchor", "favorite_reason", "comment", "followup"})
CONTEXT_ONLY_SECTION_ALIASES = {
    "affect_anchor": "affect_anchor",
    "affect anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "favorite reason": "favorite_reason",
    "comment": "comment",
    "followup": "followup",
    "follow-up": "followup",
    "followups": "followup",
    "todo": "followup",
    "to-do": "followup",
    "next": "followup",
    "year_ring": "comment",
    "year ring": "comment",
    "喜欢它的原因": "favorite_reason",
    "喜欢的原因": "favorite_reason",
    "年轮": "comment",
    "评论": "comment",
    "后续": "followup",
    "后续待办": "followup",
    "待办": "followup",
    "待办事项": "followup",
}
MARKDOWN_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
WEAK_RECALL_TOPIC_TERMS = frozenset(
    {
        "进度",
        "偏好",
        "情况",
        "状态",
        "事情",
        "东西",
        "内容",
        "相关",
        "记忆",
        "回忆",
        "总结",
        "记录",
        "查询",
        "搜索",
        "最近",
        "之前",
        "过去",
        "现在",
        "当前",
        "安排",
        "计划",
        "问题",
        "目标",
        "anything",
        "current",
        "find",
        "memory",
        "memories",
        "recent",
        "related",
        "search",
        "something",
        "status",
        "thing",
        "things",
        "topic",
    }
)
GENERIC_RECALL_CONTEXT_TERMS = frozenset(
    {
        "ai_name",
        "assistant",
        "display_name",
        "human_name",
        "user",
        "user_alias",
        "user_aliases",
        "user_display_name",
        "user_name",
        "username",
        "对方",
        "用户",
    }
)
OLD_OR_RESOLVED_QUERY_MARKERS = frozenset(
    {
        "冲突",
        "吵架",
        "争吵",
        "矛盾",
        "误会",
        "旧版本",
        "旧版",
        "旧链",
        "旧窗口",
        "已解决",
        "过期",
        "归档",
        "conflict",
        "fight",
        "argument",
        "old version",
        "old path",
        "old chain",
        "resolved",
        "archived",
        "deprecated",
        "obsolete",
    }
)
DIFFUSION_SEED_GENERIC_TOPIC_FRAGMENTS = frozenset(
    {
        "代码",
        "项目",
        "改",
    }
)
CAUTION_CONTEXT_MODES = frozenset({"reflective_repair", "conflict_repair"})
RESPONSE_ACTION_QUERY_MARKERS = frozenset(
    {
        "回复",
        "回一下",
        "回个",
        "评论",
        "留言",
        "跟个",
        "跟一句",
        "说个",
        "说一句",
        "发个",
        "发一句",
        "补个",
        "补一句",
        "嗯",
    }
)
RESPONSE_ACTION_FILLER_TERMS = frozenset(
    {
        "要不要",
        "要不",
        "是否",
        "是不是",
        "需不需要",
        "需要",
        "可以",
        "可不可以",
        "能不能",
        "回复一下",
        "回一下",
        "回个",
        "回复",
        "评论一下",
        "评论",
        "留言",
        "跟个",
        "跟一句",
        "说个",
        "说一句",
        "发个",
        "发一句",
        "补个",
        "补一句",
        "或者",
        "还是",
        "这条帖子",
        "那条帖子",
        "帖子",
        "这条消息",
        "那条消息",
        "消息",
        "嗯嗯",
        "嗯",
    }
)
AUTO_VAGUE_RECALL_MARKERS = frozenset(
    {
        "上下文",
        "想起来",
        "想起",
        "想到",
        "想到了",
        "记忆",
        "回忆",
        "最近",
        "之前",
        "刚才",
        "刚刚",
        "今天",
        "昨天",
        "明天",
        "现在",
        "当前",
        "这次",
        "这张图",
        "这张图片",
        "这个",
        "这个图",
        "这条",
        "那次",
        "那条",
        "那个",
        "相关",
        "有什么",
        "什么事",
        "发生了什么",
        "context",
        "memory",
        "memories",
        "recall",
        "recent",
        "remember",
        "resurface",
        "something",
        "anything",
    }
)
AUTO_VAGUE_FILLER_TERMS = frozenset(
    {
        "这个",
        "那个",
        "这张",
        "那张",
        "这条",
        "那条",
        "图片",
        "图",
        "上下文",
        "记忆",
        "回忆",
        "最近",
        "之前",
        "刚才",
        "刚刚",
        "今天",
        "昨天",
        "明天",
        "现在",
        "当前",
        "这次",
        "那次",
        "想起来",
        "想起",
        "想到了",
        "相关",
        "发生",
        "什么",
        "怎么",
        "怎么样",
        "事情",
        "东西",
        "内容",
        "是不是",
        "有没有",
        "有吗",
        "看看",
        "查查",
        "一下",
        "context",
        "memory",
        "memories",
        "recall",
        "recent",
        "remember",
        "resurface",
        "something",
        "anything",
    }
)
AFFECT_ONLY_QUERY_TERMS = frozenset(
    {
        "开心",
        "高兴",
        "快乐",
        "幸福",
        "甜",
        "温柔",
        "感动",
        "安心",
        "舒服",
        "喜欢",
        "难过",
        "伤心",
        "痛苦",
        "委屈",
        "焦虑",
        "烦",
        "烦躁",
        "生气",
        "愤怒",
        "害怕",
        "恐惧",
        "低落",
        "沮丧",
        "崩溃",
        "累",
        "疲惫",
        "哭",
        "哭哭",
        "大哭",
        "想哭",
        "不开心",
        "不高兴",
        "不安",
        "孤独",
        "寂寞",
        "emo",
        "sad",
        "happy",
        "angry",
        "tired",
        "anxious",
        "lonely",
        "upset",
    }
)
AFFECT_ONLY_QUERY_FILLERS = frozenset(
    {
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "今天",
        "昨天",
        "刚才",
        "刚刚",
        "现在",
        "当前",
        "有点",
        "一点",
        "一点点",
        "很",
        "好",
        "超",
        "太",
        "特别",
        "非常",
        "真的",
        "确实",
        "有些",
        "有点儿",
        "了",
        "啦",
        "呢",
        "啊",
        "呀",
        "嘛",
        "吗",
        "吧",
        "qwq",
        "tt",
        "so",
        "very",
        "really",
        "abit",
        "bit",
        "little",
        "today",
        "now",
    }
)
SHORT_CASUAL_ONLY_TERMS = frozenset(
    {
        "好耶",
        "可恶",
        "笑死",
        "不玩了",
        "不准",
        "笨",
        "笨笨",
        "失败",
        "成功",
        "配好了",
        "重来",
        "太短",
        "写一个",
        "嘿嘿",
    }
)
SHORT_TASTE_QUERY_TERMS = ("不好吃", "不好喝", "难吃", "难喝", "好吃", "好喝")
TASTE_OBJECT_TERMS = frozenset(
    {
        "饭",
        "菜",
        "餐",
        "食堂",
        "店",
        "馆",
        "面",
        "粉",
        "丸",
        "肉",
        "汤",
        "奶茶",
        "咖啡",
        "饮料",
        "甜品",
        "蛋糕",
        "水果",
        "口味",
        "味道",
        "瘦肉丸",
    }
)
TASTE_METADATA_TERMS = frozenset({"饮食", "食物", "美食", "吃饭", "口味", "餐厅", "饭店", "午饭", "晚饭"})
SHORT_CASUAL_FILLER_TERMS = frozenset(
    {
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "老公",
        "老婆",
        "宝宝",
        "宝贝",
        "亲爱的",
        "让",
        "叫",
        "把",
        "给",
        "这",
        "那",
        "这个",
        "那个",
        "一个",
        "一下",
        "端",
        "chat",
        "chat端",
        "的",
        "了",
        "啦",
        "呢",
        "啊",
        "呀",
        "嘛",
        "吗",
        "吧",
        "欸",
        "诶",
    }
)
AFFECTION_ONLY_SIGNAL_TERMS = frozenset(
    {
        "亲亲",
        "亲一下",
        "亲一口",
        "抱抱",
        "抱我",
        "抱一下",
        "贴贴",
        "蹭蹭",
        "摸摸",
        "啵啵",
        "啵",
        "么么",
        "想你了",
        "想你",
        "想我吗",
        "想我",
        "爱你",
        "爱我吗",
        "爱我",
        "mua",
        "muah",
        "kiss",
        "hug",
        "missyou",
        "loveyou",
        "loveu",
    }
)
AFFECTION_ONLY_FILLER_TERMS = frozenset(
    {
        "亲爱的",
        "老公",
        "老婆",
        "宝宝",
        "宝贝",
        "哥哥",
        "姐姐",
        "我",
        "你",
        "还",
        "也",
        "很",
        "好",
        "超",
        "真的",
        "有点",
        "一点",
        "一点点",
        "了",
        "啦",
        "呢",
        "啊",
        "呀",
        "嘛",
        "吗",
        "吧",
        "欸",
        "诶",
        "qwq",
        "tt",
    }
)
ENTITY_KEYWORD_POS_PREFIXES = ("nr", "ns", "nz")
ENTITY_KEYWORD_POS_TAGS = frozenset({"eng"})
ENTITY_KEYWORD_TITLE_SUFFIXES = ("哥哥", "姐姐", "老师", "学长", "学姐", "哥", "姐")
ENTITY_KEYWORD_SHELL_TERMS = frozenset(
    {
        *AUTO_VAGUE_FILLER_TERMS,
        *AUTO_VAGUE_RECALL_MARKERS,
        *SHORT_CASUAL_FILLER_TERMS,
        *AFFECTION_ONLY_SIGNAL_TERMS,
        *AFFECTION_ONLY_FILLER_TERMS,
        *RESPONSE_ACTION_FILLER_TERMS,
        "找了",
        "找",
        "对了",
        "对",
        "再",
        "再测",
        "再测试",
        "测试一下",
        "测试",
        "试一下",
        "试试",
        "改好了",
        "改好",
        "好了",
        "输入",
        "提到",
        "提起",
        "说到",
        "问到",
        "关于",
        "纯废话",
        "端",
    }
)
ENTITY_KEYWORD_STOP_TERMS = frozenset(
    {
        *ENTITY_KEYWORD_SHELL_TERMS,
        *AFFECT_ONLY_QUERY_TERMS,
        *AFFECTION_ONLY_SIGNAL_TERMS,
        "嗯",
        "嗯嗯",
        "好的",
        "好",
        "行",
        "可以",
        "不要",
        "不用",
        "知道",
        "觉得",
        "死亡",
        "死了",
        "吃饭",
        "吃过饭",
        "吃完饭",
        "吃了饭",
        "吃早饭",
        "吃早餐",
        "吃午饭",
        "吃午餐",
        "吃晚饭",
        "吃晚餐",
        "早饭",
        "早餐",
        "午饭",
        "午餐",
        "晚饭",
        "晚餐",
    }
)
ENTITY_KEYWORD_VERB_BLOCKERS = frozenset(
    {
        "死",
        "哭",
        "笑",
        "想",
        "找",
        "对",
        "改",
        "测",
        "试",
        "说",
        "问",
        "看",
        "查",
        "做",
        "弄",
        "写",
        "发",
        "回",
        "聊",
        "输入",
    }
)
ENTITY_QUOTED_RE = re.compile(r"[\"'“”‘’「」『』《》`]+([^\"'“”‘’「」『』《》`]{1,32})[\"'“”‘’「」『』《》`]+")
ENTITY_ENGLISH_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.:/-]{1,}\b")
ENTITY_VERSION_RE = re.compile(r"\b\d+(?:[._:-]\d+)+\b")
ENTITY_NUMBER_RE = re.compile(r"\b\d{3,}\b")


@dataclass(frozen=True)
class RecallPolicyDecision:
    admit_direct: bool
    admit_diffused: bool
    seed_allowed: bool
    reason: str
    suppressed: bool
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def admit(self) -> bool:
        return self.admit_direct


@dataclass(frozen=True)
class RecallQueryPlan:
    query: str
    wants_body_chain: bool
    requires_topic_evidence: bool
    enforce_topic_evidence: bool
    recent_context_requires_topic_evidence: bool
    explicit_old_memory: bool
    allow_caution_diffusion: bool
    specific_terms: tuple[str, ...]

    @property
    def allow_archive_targets(self) -> bool:
        return self.allow_caution_diffusion

    @property
    def related_max_chars(self) -> int:
        return 90 if self.wants_body_chain else 180

    def secondary_direct_limit(self, related_per_memory: int) -> int:
        if self.wants_body_chain:
            return 5
        return max(0, min(2, int(related_per_memory or 0)))

    @property
    def secondary_direct_requires_topic_evidence(self) -> bool:
        return not self.wants_body_chain


@dataclass(frozen=True)
class QueryAnchorPlan:
    route: str
    focus_query: str
    strong_terms: tuple[str, ...] = ()
    weak_terms: tuple[str, ...] = ()
    must_groups: tuple[tuple[str, ...], ...] = ()
    allow_direct: bool = True
    allow_diffusion_seed: bool = True
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def has_direct_constraints(self) -> bool:
        return bool(self.must_groups) or not self.allow_direct


ANCHOR_MUST_GROUP_MAX_SPAN = 24


def build_query_anchor_plan(
    query: str,
    options: MemoryRelevanceOptions | None = None,
) -> QueryAnchorPlan:
    options = options or memory_relevance_options_from_config()
    text = str(query or "").strip()
    if not text:
        return QueryAnchorPlan(
            route="empty",
            focus_query="",
            allow_direct=False,
            allow_diffusion_seed=False,
            debug={"reason": "empty_query"},
        )

    if _is_affect_only_query_text(text):
        return QueryAnchorPlan(
            route="affect_only",
            focus_query=text,
            weak_terms=(_affect_only_residue(text),),
            allow_direct=False,
            allow_diffusion_seed=False,
            debug={"reason": "affect_only"},
        )

    emotional_plan = emotional_recall_plan(text, options)
    if emotional_plan.triggered:
        must_groups = _emotional_must_groups(emotional_plan)
        focus_terms = list(dict.fromkeys([*emotional_plan.strong_terms, *emotional_plan.event_terms, *emotional_plan.weak_terms]))
        return QueryAnchorPlan(
            route="emotional_reason",
            focus_query=" ".join(focus_terms) or text,
            strong_terms=tuple(emotional_plan.strong_terms),
            weak_terms=tuple(emotional_plan.weak_terms),
            must_groups=must_groups,
            allow_direct=bool(must_groups),
            allow_diffusion_seed=bool(must_groups),
            debug={
                "reason": "emotional_recall_plan",
                "event_terms": list(emotional_plan.event_terms),
                "max_group_span": ANCHOR_MUST_GROUP_MAX_SPAN,
            },
        )

    return QueryAnchorPlan(
        route="topic_search",
        focus_query=text,
        debug={"reason": "default"},
    )


def direct_candidate_satisfies_anchor_plan(node: dict, plan: QueryAnchorPlan) -> bool:
    if not plan.allow_direct:
        return False
    if not plan.must_groups:
        return True
    text = _candidate_anchor_text(node)
    return any(_anchor_group_matches(text, group) for group in plan.must_groups)


def _emotional_must_groups(emotional_plan: Any) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    weak_terms = tuple(str(term or "").strip() for term in emotional_plan.weak_terms if str(term or "").strip())
    event_terms = tuple(str(term or "").strip() for term in emotional_plan.event_terms if str(term or "").strip())
    state_terms = tuple(
        str(term or "").strip()
        for term in sorted(EMOTIONAL_RECALL_STATE_TERMS, key=len, reverse=True)
        if str(term or "").strip()
    )

    for strong in emotional_plan.strong_terms:
        strong_text = str(strong or "").strip()
        if not strong_text:
            continue
        strong_key = _compact_anchor_term(strong_text)
        pieces: list[str] = []
        for term in weak_terms:
            if _compact_anchor_term(term) in strong_key:
                pieces.append(term)
        for term in state_terms:
            term_key = _compact_anchor_term(term)
            if term_key and term_key in strong_key:
                pieces.append(_canonical_anchor_state(term))
        groups.append(_dedupe_group(pieces or [strong_text]))

    event_anchor = _primary_emotional_event_term(event_terms)
    if event_anchor and weak_terms:
        groups.append(_dedupe_group([event_anchor, weak_terms[0]]))
    elif not groups and weak_terms:
        groups.append(_dedupe_group([weak_terms[0]]))

    return tuple(dict.fromkeys(group for group in groups if group))


def _primary_emotional_event_term(event_terms: tuple[str, ...]) -> str:
    terms = [
        str(term or "").strip()
        for term in event_terms
        if str(term or "").strip()
    ]
    if not terms:
        return ""
    keyed = [
        (term, _compact_anchor_term(term))
        for term in terms
        if _compact_anchor_term(term)
    ]
    compact_terms = [key for _term, key in keyed]
    candidates = [
        term
        for term, key in keyed
        if not any(other != key and other in key for other in compact_terms)
    ]
    candidates = candidates or [term for term, _key in keyed]
    return sorted(candidates, key=lambda item: (len(_compact_anchor_term(item)), len(item)))[0]


def _candidate_anchor_text(node: dict) -> str:
    if not isinstance(node, dict):
        return ""
    meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
    if "bucket_id" in node or node.get("moment_id"):
        return " ".join(
            [
                str(node.get("text") or ""),
                str(node.get("content") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                str(meta.get("bucket_name") or ""),
                _join_terms(meta.get("bucket_tags")),
                _join_terms(meta.get("bucket_domain")),
            ]
        )
    return " ".join(
        [
            _content_without_context_only_sections(str(node.get("content") or "")),
            str(node.get("text") or ""),
            str(node.get("name") or ""),
            str(meta.get("name") or ""),
            str(meta.get("annotation_summary") or meta.get("summary") or ""),
            _evidence_spans_text(meta.get("evidence_spans")),
            _join_terms(meta.get("tags")),
            _join_terms(meta.get("domain")),
        ]
    )


def _anchor_group_matches(text: str, group: tuple[str, ...]) -> bool:
    compact_text = _compact_anchor_term(text)
    if not compact_text:
        return False
    positions_by_term = []
    for term in group:
        key = _compact_anchor_term(term)
        if not key:
            continue
        positions = _anchor_term_positions(compact_text, key)
        if not positions:
            return False
        positions_by_term.append(positions)
    if len(positions_by_term) <= 1:
        return bool(positions_by_term)
    for spans in product(*positions_by_term):
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        if end - start <= ANCHOR_MUST_GROUP_MAX_SPAN:
            return True
    return False


def _anchor_term_positions(text: str, term: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            break
        positions.append((index, index + len(term)))
        start = index + max(1, len(term))
    return positions


def _compact_anchor_term(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", str(value or "").strip().lower())


def _canonical_anchor_state(term: str) -> str:
    return "哭" if term in {"哭", "哭了"} else term


def _dedupe_group(terms: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen = set()
    for term in terms:
        cleaned = str(term or "").strip()
        key = _compact_anchor_term(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _join_terms(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _affect_only_residue(query: str) -> str:
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(query or "").lower())
    stripped = compact
    for term in sorted(AFFECT_ONLY_QUERY_FILLERS, key=len, reverse=True):
        stripped = stripped.replace(term, "")
    return stripped


def _is_affect_only_query_text(query: str) -> bool:
    residue = _affect_only_residue(query)
    return bool(residue and residue in AFFECT_ONLY_QUERY_TERMS)


class RecallPolicy:
    def __init__(
        self,
        options: MemoryRelevanceOptions | None = None,
        *,
        semantic_threshold: float = 0.72,
        rerank_threshold: float = 0.65,
        ai_reaction_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.options = options or memory_relevance_options_from_config()
        self.semantic_threshold = _safe_float(semantic_threshold, 0.72)
        self.rerank_threshold = _safe_float(rerank_threshold, 0.65)
        self.ai_reaction_names = self._normalize_reaction_names(
            ai_reaction_names if ai_reaction_names is not None else [identity_names().get("ai_name")]
        )
        self.recall_context_terms = self._normalize_recall_context_terms(
            [*self.options.context_terms, *GENERIC_RECALL_CONTEXT_TERMS]
        )

    def requires_topic_evidence(self, query: str) -> bool:
        return query_has_explicit_entity_marker(query) or query_has_technical_recall_marker(query)

    def should_enforce_topic_evidence(self, query: str, *, allow_body_chain: bool = False) -> bool:
        return self.requires_topic_evidence(query) and not allow_body_chain

    def plan_query(self, query: str, *, context_mode: str = "") -> RecallQueryPlan:
        text = str(query or "").strip()
        wants_body_chain = query_has_facet(text, "embodiment", self.options)
        explicit_old_memory = self._query_explicitly_requests_old_memory(text)
        allow_caution_diffusion = explicit_old_memory or str(context_mode or "").strip() in CAUTION_CONTEXT_MODES
        return RecallQueryPlan(
            query=text,
            wants_body_chain=wants_body_chain,
            requires_topic_evidence=self.requires_topic_evidence(text),
            enforce_topic_evidence=self.should_enforce_topic_evidence(
                text,
                allow_body_chain=wants_body_chain,
            ),
            recent_context_requires_topic_evidence=self.is_auto_concrete_topic_query(text),
            explicit_old_memory=explicit_old_memory,
            allow_caution_diffusion=allow_caution_diffusion,
            specific_terms=tuple(self.specific_query_terms(text)),
        )

    def build_query_anchor_plan(self, query: str) -> QueryAnchorPlan:
        return build_query_anchor_plan(query, self.options)

    def direct_candidate_satisfies_anchor_plan(self, node: dict, plan: QueryAnchorPlan) -> bool:
        return direct_candidate_satisfies_anchor_plan(node, plan)

    def _query_explicitly_requests_old_memory(self, query: str) -> bool:
        if not str(query or "").strip():
            return False
        if query_has_facet(query, "old_or_resolved", self.options):
            return True
        text = " ".join(str(query or "").lower().split())
        return any(marker in text for marker in OLD_OR_RESOLVED_QUERY_MARKERS)

    def is_auto_query_too_vague(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text:
            return False
        if self._is_reaction_only_query(text):
            return True
        if self._is_probe_only_query(text):
            return True
        if self._is_short_casual_only_query(text):
            return True
        if query_has_explicit_entity_marker(text) or query_has_technical_recall_marker(text):
            return False
        if self._is_affection_only_query(text):
            return True
        if self.extract_entity_keywords(text):
            return False
        if self._is_affect_only_query(text):
            return True
        if self._is_context_free_response_action_query(text):
            return True
        lowered = text.lower()
        if not any(marker in lowered for marker in AUTO_VAGUE_RECALL_MARKERS):
            return False
        return not self._auto_query_has_concrete_anchor(text)

    def is_auto_concrete_topic_query(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text or self.is_auto_query_too_vague(text):
            return False
        if self._is_affect_only_query(text):
            return False
        if query_has_explicit_entity_marker(text) or query_has_technical_recall_marker(text):
            return True
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", text)
        candidate = compact
        for prefix in ("最近", "今天", "昨天", "明天", "之前", "刚才", "刚刚", "这次", "当前", "现在"):
            if candidate.startswith(prefix) and len(candidate) > len(prefix):
                candidate = candidate[len(prefix):]
                break
        candidate = candidate.strip("的")
        if not re.fullmatch(r"[\u4e00-\u9fff]{2,12}", candidate):
            return False
        context_terms = {str(term).lower() for term in self.options.context_terms}
        if candidate.lower() in context_terms:
            return False
        blockers = (
            "我",
            "你",
            "他",
            "她",
            "它",
            "这",
            "那",
            "什么",
            "怎么",
            "怎样",
            "为什么",
            "是不是",
            "有没有",
            "想起",
            "想起来",
            "记忆",
            "上下文",
        )
        return not any(marker in candidate for marker in blockers)

    def _auto_query_has_concrete_anchor(self, query: str) -> bool:
        if re.search(r"\b[A-Za-z][A-Za-z0-9_.:/-]{2,}\b", query):
            return True
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", query.lower())
        stripped = compact
        removable = list(AUTO_VAGUE_RECALL_MARKERS | AUTO_VAGUE_FILLER_TERMS | set(self.options.context_terms))
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        stripped = re.sub(r"[我你他她它的是了嘛吗呢啊呀欸诶吧哈嗯呜有里看查找问说]+", "", stripped)
        return len(stripped) >= 2

    def _is_context_free_response_action_query(self, query: str) -> bool:
        lowered = str(query or "").lower()
        if not any(marker in lowered for marker in RESPONSE_ACTION_QUERY_MARKERS):
            return False
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", lowered)
        stripped = compact
        removable = list(
            RESPONSE_ACTION_FILLER_TERMS
            | AUTO_VAGUE_FILLER_TERMS
            | set(self.options.context_terms)
        )
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        stripped = re.sub(
            r"[我你他她它的是了嘛吗呢啊呀欸诶吧哈嗯呜有里看查找问说]+",
            "",
            stripped,
        )
        return len(stripped) < 2

    def _is_reaction_only_query(self, query: str) -> bool:
        compact = re.sub(r"\s+", "", str(query or "").lower())
        if not compact:
            return False
        alnum_or_cjk = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", compact)
        if not alnum_or_cjk:
            return True
        reaction_terms = {
            "tt",
            "qwq",
            "qaq",
            "orz",
            "xswl",
            "lol",
            "lmao",
            "哈哈",
            "哈哈哈",
            "哈哈哈哈",
            "嘿嘿",
            "呜呜",
            "呜呜呜",
            "哇",
            "哇啊",
            "啊啊",
            "啊啊啊",
            "嗯嗯",
            "嗯",
            "老公",
            "老婆",
            "宝宝",
            "宝贝",
            "亲爱的",
            "哥哥",
        }
        return alnum_or_cjk in reaction_terms or alnum_or_cjk in self.ai_reaction_names

    @staticmethod
    def _normalize_reaction_names(values: list[str] | tuple[str, ...] | None) -> set[str]:
        names: set[str] = set()
        for value in values or []:
            compact = re.sub(r"\s+", "", str(value or "").lower())
            key = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", compact)
            if key:
                names.add(key)
        return names

    @staticmethod
    def _normalize_recall_context_terms(values) -> set[str]:
        terms: set[str] = set()
        for value in values or []:
            key = re.sub(r"\s+", " ", str(value or "").strip().lower())
            if key:
                terms.add(key)
            compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", key)
            if compact:
                terms.add(compact)
        return terms

    def _is_recall_context_term(self, term: str) -> bool:
        key = re.sub(r"\s+", " ", str(term or "").strip().lower())
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", key)
        return key in self.recall_context_terms or compact in self.recall_context_terms

    def _is_probe_only_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        probe_markers = (
            "试一下",
            "试试",
            "测试一下",
            "测试",
            "test",
            "try",
        )
        if not any(marker in text for marker in probe_markers):
            return False
        if any(re.search(r"[\u4e00-\u9fff]", term) for term in self.extract_entity_keywords(text)):
            return False
        recall_intent_markers = (
            "记得",
            "记忆",
            "想起",
            "回忆",
            "召回",
            "检索",
            "查一下",
            "找一下",
            "为什么",
            "原因",
            "remember",
            "recall",
            "memory",
            "search",
            "look up",
            "why",
        )
        return not any(marker in text for marker in recall_intent_markers)

    def _is_affect_only_query(self, query: str) -> bool:
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(query or "").lower())
        if not compact:
            return False
        stripped = compact
        for term in sorted(AFFECT_ONLY_QUERY_FILLERS, key=len, reverse=True):
            stripped = stripped.replace(term, "")
        if not stripped:
            return False
        return stripped in AFFECT_ONLY_QUERY_TERMS

    def _is_short_casual_only_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        if any(marker in text for marker in AUTO_VAGUE_RECALL_MARKERS):
            return False
        if query_has_technical_recall_marker(text):
            return False
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        if not compact or len(compact) > 24:
            return False
        compact = re.sub(r"\d{1,4}$", "", compact)
        if not compact:
            return False
        if compact in SHORT_CASUAL_ONLY_TERMS:
            return True
        has_casual_signal = any(term in compact for term in SHORT_CASUAL_ONLY_TERMS)
        if not has_casual_signal:
            return False
        stripped = compact
        removable = (
            SHORT_CASUAL_ONLY_TERMS
            | SHORT_CASUAL_FILLER_TERMS
            | AFFECT_ONLY_QUERY_FILLERS
            | set(self.options.context_terms)
        )
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        return len(stripped) < 2

    def _is_affection_only_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        if query_has_explicit_entity_marker(text) or query_has_technical_recall_marker(text):
            return False
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        if not compact:
            return False
        if not any(term in compact for term in AFFECTION_ONLY_SIGNAL_TERMS):
            return False
        stripped = compact
        removable = (
            AFFECTION_ONLY_SIGNAL_TERMS
            | AFFECTION_ONLY_FILLER_TERMS
            | SHORT_CASUAL_FILLER_TERMS
            | AFFECT_ONLY_QUERY_FILLERS
            | set(self.options.context_terms)
        )
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        return len(stripped) < 2

    def _short_taste_query_terms(self, query: str) -> list[str]:
        text = str(query or "").strip().lower()
        if not text:
            return []
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        compact = re.sub(r"\d{1,4}$", "", compact)
        if not compact or len(compact) > 12:
            return []
        stripped = compact
        removable = SHORT_CASUAL_FILLER_TERMS | (AFFECT_ONLY_QUERY_FILLERS - {"好"}) | set(self.options.context_terms)
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        return [term for term in SHORT_TASTE_QUERY_TERMS if stripped == term]

    def _fields_have_taste_evidence(
        self,
        taste_terms: list[str],
        fields: str,
        metadata_text: str,
    ) -> bool:
        text = str(fields or "").lower()
        meta = str(metadata_text or "").lower()
        has_food_metadata = any(term in meta for term in TASTE_METADATA_TERMS | TASTE_OBJECT_TERMS)
        for term in taste_terms:
            if term == "好吃":
                pattern = r"(?<!好)好吃"
            elif term == "好喝":
                pattern = r"(?<!好)好喝"
            else:
                pattern = re.escape(term)
            for match in re.finditer(pattern, text):
                start, end = match.span()
                window = text[max(0, start - 18): min(len(text), end + 18)]
                if "隔壁好吃" in window or "隔壁好喝" in window:
                    continue
                if has_food_metadata and any(obj in window for obj in TASTE_OBJECT_TERMS | TASTE_METADATA_TERMS):
                    return True
                if any(obj in window for obj in TASTE_OBJECT_TERMS):
                    return True
                if re.search(r"觉得.{1,16}" + pattern, window):
                    return True
        return False

    def extract_entity_keywords(self, query: str) -> list[str]:
        raw = str(query or "").strip()
        if not raw:
            return []
        keywords: list[str] = []
        seen: set[str] = set()
        strong_keys: set[str] = set()

        def add(value: object, *, strong: bool = False) -> None:
            cleaned = self._normalize_entity_keyword(value)
            if not cleaned or not self._entity_keyword_allowed(cleaned, strong=strong):
                return
            key = self._compact_entity_keyword(cleaned)
            if not key or key in seen:
                return
            seen.add(key)
            if strong:
                strong_keys.add(key)
            keywords.append(cleaned)

        for match in ENTITY_QUOTED_RE.finditer(raw):
            add(match.group(1), strong=True)
        for match in ENTITY_VERSION_RE.finditer(raw):
            add(match.group(0), strong=True)
        for match in ENTITY_ENGLISH_RE.finditer(raw):
            add(match.group(0), strong=True)
        for match in ENTITY_NUMBER_RE.finditer(raw):
            add(match.group(0), strong=True)
        for match in re.finditer(r"[A-Za-z0-9_.:-]*[\u4e00-\u9fff]+[A-Za-z0-9_.:-]+|[A-Za-z0-9_.:-]+[\u4e00-\u9fff]+[A-Za-z0-9_.:-]*", raw):
            mixed = match.group(0)
            stripped = self._strip_entity_shell(mixed)
            value = mixed if stripped == self._compact_entity_keyword(mixed) else stripped
            add(value, strong=True)

        for word, flag in self._posseg_words(raw):
            if (
                flag in ENTITY_KEYWORD_POS_TAGS
                or any(str(flag or "").startswith(prefix) for prefix in ENTITY_KEYWORD_POS_PREFIXES)
            ):
                add(word, strong=True)
                for expanded in self._expand_entity_title_suffixes(raw, word):
                    add(expanded, strong=True)

        for span in re.findall(r"[\u4e00-\u9fff]{2,16}", raw):
            candidate = self._strip_entity_shell(span)
            if 2 <= len(candidate) <= 8 and not self._entity_candidate_has_verb_blocker(candidate):
                add(candidate)

        return self._dedupe_entity_keywords(keywords, strong_keys=strong_keys)

    def _dedupe_entity_keywords(self, values: list[str], *, strong_keys: set[str] | None = None) -> list[str]:
        strong_keys = strong_keys or set()
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for value in values:
            cleaned = self._normalize_entity_keyword(value)
            key = self._compact_entity_keyword(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            pairs.append((cleaned, key))
        output: list[str] = []
        for cleaned, key in pairs:
            if key in strong_keys:
                contained_by_longer = any(
                    other_key != key and other_key in strong_keys and key in other_key
                    for _other, other_key in pairs
                )
                noisy_extension_of_strong = False
            else:
                contained_by_longer = any(
                    other_key != key and other_key not in strong_keys and key in other_key
                    for _other, other_key in pairs
                )
                noisy_extension_of_strong = any(
                    other_key != key and other_key in strong_keys and other_key in key
                    for _other, other_key in pairs
                )
            if contained_by_longer or noisy_extension_of_strong:
                continue
            output.append(cleaned)
        return output

    @staticmethod
    def _posseg_words(text: str) -> list[tuple[str, str]]:
        try:
            import jieba.posseg as pseg
        except Exception:
            return []
        try:
            return [(str(item.word), str(item.flag)) for item in pseg.cut(text)]
        except Exception:
            return []

    @staticmethod
    def _normalize_entity_keyword(value: object) -> str:
        cleaned = str(value or "").strip().strip("\"'`“”‘’「」『』《》")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip("，。！？、,.!?:：;；~～♡❤♥（）()[]【】")

    @staticmethod
    def _compact_entity_keyword(value: object) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", str(value or "").strip().lower())

    def _entity_keyword_allowed(self, value: str, *, strong: bool = False) -> bool:
        compact = self._compact_entity_keyword(value)
        if not compact:
            return False
        if compact in ENTITY_KEYWORD_STOP_TERMS or self._is_recall_context_term(compact):
            return False
        if compact in self.ai_reaction_names:
            return False
        if ENTITY_VERSION_RE.fullmatch(compact) or ENTITY_NUMBER_RE.fullmatch(compact):
            return True
        if re.fullmatch(r"[a-z][a-z0-9_.:/-]{1,}", compact):
            return compact not in ENTITY_KEYWORD_STOP_TERMS
        if re.fullmatch(r"[\u4e00-\u9fff]+", compact):
            residue = self._strip_entity_shell(compact)
            if len(residue) < 2 or len(residue) > 8:
                return False
            if residue in ENTITY_KEYWORD_STOP_TERMS or self._is_affect_only_query(residue):
                return False
            if not strong and self._entity_candidate_has_verb_blocker(residue):
                return False
            return True
        return True

    def _strip_entity_shell(self, value: object) -> str:
        residue = self._compact_entity_keyword(value)
        for term in sorted(ENTITY_KEYWORD_SHELL_TERMS, key=len, reverse=True):
            cleaned = self._compact_entity_keyword(term)
            if cleaned:
                residue = residue.replace(cleaned, "")
        residue = re.sub(r"[我你他她它的是了嘛吗呢啊呀欸诶吧哈嗯呜]+", "", residue)
        return residue

    @staticmethod
    def _entity_candidate_has_verb_blocker(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text in ENTITY_KEYWORD_VERB_BLOCKERS:
            return True
        chars = [char for char in text if re.fullmatch(r"[\u4e00-\u9fff]", char)]
        return bool(chars and len(chars) == len(text) and all(char in ENTITY_KEYWORD_VERB_BLOCKERS for char in chars))

    def _expand_entity_title_suffixes(self, raw: str, entity: str) -> list[str]:
        compact_raw = self._compact_entity_keyword(raw)
        compact_entity = self._compact_entity_keyword(entity)
        if not compact_raw or not compact_entity:
            return []
        output: list[str] = []
        start = 0
        while True:
            index = compact_raw.find(compact_entity, start)
            if index < 0:
                break
            tail = compact_raw[index + len(compact_entity):]
            for suffix in sorted(ENTITY_KEYWORD_TITLE_SUFFIXES, key=len, reverse=True):
                compact_suffix = self._compact_entity_keyword(suffix)
                if compact_suffix and tail.startswith(compact_suffix):
                    output.append(entity + suffix)
                    break
            start = index + max(1, len(compact_entity))
        return output

    def specific_query_terms(self, query: str) -> list[str]:
        raw = str(query or "")
        terms = list(content_terms_for_query(raw, self.options))
        topic_key = recall_topic_query(raw, self.options)
        allow_single_cjk_terms = {
            str(term or "").strip()
            for term in content_terms_for_query(topic_key, self.options)
            if re.fullmatch(r"[\u4e00-\u9fff]", str(term or "").strip())
        }
        terms.extend(re.findall(r"\d+(?:\.\d+)+", raw))
        terms.extend(re.findall(r"[A-Za-z]+[A-Za-z0-9_.:-]*\d[A-Za-z0-9_.:-]*", raw))
        kept = []
        seen = set()
        for term in terms:
            cleaned = str(term or "").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            if key in WEAK_RECALL_TOPIC_TERMS:
                continue
            if self._is_recall_context_term(cleaned):
                continue
            if re.fullmatch(r"[a-z0-9_.:-]+", key) and len(key) < 3 and not re.fullmatch(r"\d+(?:\.\d+)+", key):
                continue
            if (
                re.fullmatch(r"[\u4e00-\u9fff]+", cleaned)
                and len(cleaned) < 2
                and cleaned not in allow_single_cjk_terms
            ):
                continue
            if any(_term_subsumes(existing.lower(), key) for existing in kept):
                continue
            kept = [existing for existing in kept if not _term_subsumes(key, existing.lower())]
            seen = {existing.lower() for existing in kept}
            seen.add(key)
            kept.append(cleaned)
        return kept

    def moment_has_topic_evidence(self, query: str, moment: dict) -> bool:
        taste_terms = self._short_taste_query_terms(query)
        terms = self.specific_query_terms(query)
        if not terms:
            return False
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(moment.get("text") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                str(meta.get("bucket_name") or ""),
                " ".join(str(tag) for tag in (meta.get("bucket_tags") or []) if str(tag).strip()),
                " ".join(str(item) for item in (meta.get("bucket_domain") or []) if str(item).strip()),
            ]
        ).lower()
        if taste_terms:
            metadata_text = " ".join(
                [
                    str(meta.get("bucket_name") or ""),
                    " ".join(str(tag) for tag in (meta.get("bucket_tags") or []) if str(tag).strip()),
                    " ".join(str(item) for item in (meta.get("bucket_domain") or []) if str(item).strip()),
                ]
            ).lower()
            return self._fields_have_taste_evidence(taste_terms, fields, metadata_text)
        return any(term.lower() in fields for term in terms)

    def bucket_has_topic_evidence(self, query: str, bucket: dict) -> bool:
        taste_terms = self._short_taste_query_terms(query)
        terms = self.specific_query_terms(query)
        if not terms:
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        fields = " ".join(
            [
                _content_without_context_only_sections(str(bucket.get("content") or "")),
                str(meta.get("name") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                " ".join(str(tag) for tag in (meta.get("tags") or []) if str(tag).strip()),
                " ".join(str(item) for item in (meta.get("domain") or []) if str(item).strip()),
            ]
        ).lower()
        if taste_terms:
            metadata_text = " ".join(
                [
                    str(meta.get("name") or ""),
                    " ".join(str(tag) for tag in (meta.get("tags") or []) if str(tag).strip()),
                    " ".join(str(item) for item in (meta.get("domain") or []) if str(item).strip()),
                ]
            ).lower()
            return self._fields_have_taste_evidence(taste_terms, fields, metadata_text)
        return any(term.lower() in fields for term in terms)

    def node_has_topic_evidence(self, query: str, node: dict) -> bool:
        if "bucket_id" in node or node.get("moment_id"):
            return self.moment_has_topic_evidence(query, node)
        return self.bucket_has_topic_evidence(query, node)

    def allows_moment_context(
        self,
        query: str,
        moment: dict,
        *,
        allow_body_chain: bool = False,
    ) -> bool:
        if not self.should_enforce_topic_evidence(query, allow_body_chain=allow_body_chain):
            return True
        return self.moment_has_topic_evidence(query, moment)

    def allows_bucket_context(
        self,
        query: str,
        bucket: dict,
        *,
        allow_body_chain: bool = False,
    ) -> bool:
        if not self.should_enforce_topic_evidence(query, allow_body_chain=allow_body_chain):
            return True
        return self.bucket_has_topic_evidence(query, bucket)

    def has_strong_score(
        self,
        *,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
    ) -> bool:
        return (
            _safe_float(semantic_score, 0.0) >= self.semantic_threshold
            or _safe_float(rerank_score, 0.0) >= self.rerank_threshold
        )

    def assess(
        self,
        query: str,
        node: dict,
        *,
        has_topic_evidence: bool | None = None,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
        high_confidence_edge: bool = False,
        context_only: bool = False,
        auto: bool = False,
    ) -> RecallPolicyDecision:
        if has_topic_evidence is None:
            has_topic_evidence = self.node_has_topic_evidence(query, node)
        auto_too_vague = self.is_auto_query_too_vague(query) if auto else False
        debug = {
            "requires_topic_evidence": self.requires_topic_evidence(query),
            "has_topic_evidence": bool(has_topic_evidence),
            "specific_query_terms": self.specific_query_terms(query),
            "short_taste_query_terms": self._short_taste_query_terms(query),
            "semantic_score": _maybe_float(semantic_score),
            "rerank_score": _maybe_float(rerank_score),
            "high_confidence_edge": bool(high_confidence_edge),
            "context_only": bool(context_only),
            "auto": bool(auto),
            "auto_too_vague": bool(auto_too_vague),
        }

        if auto_too_vague:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="auto_vague_query_without_topic",
                suppressed=True,
                debug=debug,
            )

        if context_only:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="context_only_temperature_moment",
                suppressed=True,
                debug=debug,
            )

        base = recall_admission_decision(
            query,
            node,
            self.options,
            semantic_score=semantic_score,
            rerank_score=rerank_score,
            high_confidence_edge=high_confidence_edge,
            semantic_threshold=self.semantic_threshold,
            rerank_threshold=self.rerank_threshold,
        )
        debug["base_reason"] = base.reason

        if not base.admit:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason=base.reason,
                suppressed=True,
                debug=debug,
            )

        if (
            debug["short_taste_query_terms"]
            and not has_topic_evidence
            and not self.has_strong_score(
                semantic_score=semantic_score,
                rerank_score=rerank_score,
            )
        ):
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="short_taste_query_without_taste_evidence",
                suppressed=True,
                debug=debug,
            )

        if (
            debug["requires_topic_evidence"]
            and not has_topic_evidence
            and not self.has_strong_score(
                semantic_score=semantic_score,
                rerank_score=rerank_score,
            )
            and not high_confidence_edge
        ):
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="query_topic_evidence_missing",
                suppressed=True,
                debug=debug,
            )

        return RecallPolicyDecision(
            admit_direct=True,
            admit_diffused=True,
            seed_allowed=True,
            reason=base.reason,
            suppressed=False,
            debug=debug,
        )


def is_context_only_section(section: Any) -> bool:
    return str(section or "") in CONTEXT_ONLY_SECTIONS


def _evidence_spans_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())
    return " ".join(parts)


def _content_without_context_only_sections(content: str) -> str:
    lines = str(content or "").splitlines()
    kept: list[str] = []
    skip_until_level = 0
    for line in lines:
        match = MARKDOWN_HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            raw_heading = match.group(2).strip()
            if skip_until_level and level > skip_until_level:
                continue
            skip_until_level = 0
            if _context_only_heading(raw_heading):
                skip_until_level = level
                continue
        if skip_until_level:
            continue
        kept.append(line)
    return "\n".join(kept)


def _context_only_heading(heading: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(heading or "").strip().lower())
    normalized = normalized.strip("：: -_")
    normalized = re.sub(r"^\d+[.、]\s*", "", normalized)
    normalized = normalized.replace("-", "_")
    return CONTEXT_ONLY_SECTION_ALIASES.get(normalized, normalized) in CONTEXT_ONLY_SECTIONS


def diffusion_seed_topic_term_has_specific_residue(term: object) -> bool:
    residue = re.sub(
        r"[^0-9a-z\u4e00-\u9fff_.:-]+",
        "",
        str(term or "").strip().lower(),
    )
    if not residue:
        return False
    for generic in sorted(DIFFUSION_SEED_GENERIC_TOPIC_FRAGMENTS, key=len, reverse=True):
        residue = residue.replace(generic, "")
    return bool(residue.strip())


def _term_subsumes(container: str, contained: str) -> bool:
    if container == contained:
        return True
    if not container or not contained:
        return False
    if not re.search(r"\d", contained):
        return False
    return contained in container


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    number = _maybe_float(value)
    return default if number is None else number
