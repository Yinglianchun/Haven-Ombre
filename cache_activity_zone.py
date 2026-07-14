"""
cache_activity_zone.py -- 缓存区"活动区"管理
system_standard.txt 结构（三段式）：
  [人设]
  ═══ PERSONA_SEPARATOR ═══
  [工具区 — TOOL INSTRUCTIONS，纯静态规则]
  ═══ CACHE_ACTIVITY_SEPARATOR ═══
  [缓存区活动区 — 画像 / 事件线 / 时间线 / 闹钟 / 今日I，代码自动维护]

五个 block（PORTRAIT_MEMORY / EVENT_LINE / DAILY_TIMELINE / ALARM / I_TODAY）各自有独立的 BEGIN/END 标记，
互不覆盖；gateway / event_line / alarm_clock / daily_timeline / self_i 只替换各自的 block。

放在缓存区最下面（而不是塞进工具区或用单独 cache_control 断点）的原因：
- 这些是"变动大但只追加/定期刷新"的内容，和人设/工具区（几乎不变）分开
- 复用 system message 本身已有的 cache_control 断点，不额外占用 Anthropic 4 断点上限
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("ombre_brain.cache_activity_zone")

CACHE_ACTIVITY_SEPARATOR = "═══ CACHE ACTIVITY ZONE — AUTO-MANAGED (PORTRAIT / EVENT LINE / TIMELINE / ALARM / I_TODAY) ═══"

BLOCK_ORDER = ("PORTRAIT_MEMORY", "EVENT_LINE", "DAILY_TIMELINE", "ALARM", "I_TODAY")

_DEFAULT_TEXT = {
    "PORTRAIT_MEMORY": "【画像记忆】（长期画像事实，只读）\n（当前无画像事实）",
    "EVENT_LINE": "【事件线】（跨天持续事件）\n（当前无进行中的长期事件）",
    "DAILY_TIMELINE": "【时间线】（今日已封存小时）\n（暂无记录）",
    "ALARM": "【闹钟提醒】（未来事件预警，窗口期内出现）\n（当前无到期提醒）",
    "I_TODAY": "【今日自我认知 I】\n（今日尚未写入）",
}


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"


def default_block_text(name: str) -> str:
    return _DEFAULT_TEXT.get(name, "")


def update_block(persona_file: str, block_name: str, content: str) -> None:
    """替换 system_standard.txt 中指定 block 的内容，不影响人设/工具区及其他 block。"""
    path = Path(persona_file)
    if not path.exists():
        logger.warning("update_block: persona_file not found | %s", persona_file)
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("update_block: read failed | %s", exc)
        return

    begin, end = _markers(block_name)
    content = str(content or "").strip()
    new_block = f"{begin}\n{content}\n{end}"

    if begin in text and end in text:
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        text, count = pattern.subn(new_block, text, count=1)
        if count == 0:
            logger.warning("update_block: marker replace failed | block=%s", block_name)
            return
    else:
        if CACHE_ACTIVITY_SEPARATOR not in text:
            # 兼容旧 separator 文案
            for old_sep in (
                "═══ CACHE ACTIVITY ZONE — AUTO-MANAGED (PORTRAIT / EVENT LINE / TIMELINE / ALARM) ═══",
                "═══ CACHE ACTIVITY ZONE — AUTO-MANAGED (EVENT LINE / TIMELINE / ALARM) ═══",
            ):
                if old_sep in text:
                    text = text.replace(old_sep, CACHE_ACTIVITY_SEPARATOR, 1)
                    break
            else:
                text = text.rstrip() + "\n\n" + CACHE_ACTIVITY_SEPARATOR + "\n"
        # 按 BLOCK_ORDER 插入到正确位置（缺失时不总是追加到末尾）
        inserted = False
        try:
            idx = BLOCK_ORDER.index(block_name)
        except ValueError:
            idx = -1
        if idx >= 0:
            # 找下一个已存在的 block，插到它前面
            for later in BLOCK_ORDER[idx + 1:]:
                later_begin, _ = _markers(later)
                pos = text.find(later_begin)
                if pos >= 0:
                    text = text[:pos] + new_block + "\n\n" + text[pos:]
                    inserted = True
                    break
            if not inserted and idx > 0:
                # 插到前一个 block 的 END 后面
                for earlier in reversed(BLOCK_ORDER[:idx]):
                    _, earlier_end = _markers(earlier)
                    pos = text.find(earlier_end)
                    if pos >= 0:
                        pos = pos + len(earlier_end)
                        text = text[:pos] + "\n\n" + new_block + text[pos:]
                        inserted = True
                        break
        if not inserted:
            text = text.rstrip() + "\n\n" + new_block + "\n"

    try:
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.warning("update_block: write failed | %s", exc)


def ensure_all_blocks(persona_file: str) -> None:
    """确保 CACHE_ACTIVITY_SEPARATOR 及三个 block 都存在（缺失的补默认值）。幂等，可重复调用。"""
    path = Path(persona_file)
    if not path.exists():
        return
    for name in BLOCK_ORDER:
        begin, _end = _markers(name)
        text = path.read_text(encoding="utf-8")
        if begin not in text:
            update_block(persona_file, name, default_block_text(name))


def get_cache_activity_part(persona_file: str, separator: str) -> str:
    """读取缓存区活动区整体文本（separator 及之后），供 update_persona_part / 工具同步 保留使用。"""
    path = Path(persona_file)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if separator in text:
        return separator + "\n" + text.split(separator, 1)[1].lstrip("\n")
    return ""
