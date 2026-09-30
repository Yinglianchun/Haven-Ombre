"""
gateway.py 改造（事件线 + 闹钟 + 时间线缓存区迁移）：
1. 新增模块级常量 _NEW_TAG_NAMES 及通用带属性标签流式过滤器 _make_attr_tag_filter
2. __init__：初始化 EventLine / AlarmClock，确保 system_standard.txt 三段结构完整
3. Phase6 Step2-pre：历史剥离扩展到 5 个新标签
4. path1 (OpenAI 风格 SSE) 流式过滤器接入新标签过滤 + flush
5. path2 (Anthropic 风格 SSE) 流式过滤器接入新标签过滤 + flush
6. 移除 Phase6 Step2b（时间线已迁移进 system_standard.txt，断点释放）
7. _record_conversation_turn：提取并处理 5 个新标签（早于 early return）
"""
import re
import subprocess
from pathlib import Path

GW = Path("/opt/Ombre-Brain/gateway.py")
gw = GW.read_text(encoding="utf-8")
orig_len = len(gw)


def must_replace(old: str, new: str, label: str, count: int = 1):
    global gw
    n = gw.count(old)
    if n < 1:
        print(f"❌ [{label}] 未找到匹配文本")
        raise SystemExit(1)
    gw = gw.replace(old, new, count)
    print(f"✅ [{label}] 替换完成 (匹配 {n} 处，替换 {count} 处)")


# ── 1. 模块级常量 + 通用标签过滤器（插入到 logger 定义之后）────────────────────

OLD_LOGGER = 'logger = logging.getLogger("ombre_brain.gateway")'
NEW_LOGGER = OLD_LOGGER + '''

# ── 事件线 / 闹钟 隐藏标签（Phase 8）────────────────────────────────────────
_NEW_TAG_NAMES = ("event_create", "event_update", "event_close", "alarm_set", "alarm_cancel")


def _make_attr_tag_filter(tag_names):
    """
    返回 (feed, flush) 闭包函数，用于流式剥离带属性的隐藏标签，
    如 <event_create title="...">...</event_create>（属性长度不定，与 memory_note 定长标签不同）。
    """
    prefixes = tuple(f"<{t}" for t in tag_names)
    max_prefix_len = max((len(p) for p in prefixes), default=0)
    state = {"buf": "", "suppress": False, "close_tag": None}

    def feed(raw_delta: str) -> str:
        state["buf"] += raw_delta
        out = []
        while True:
            buf = state["buf"]
            if state["suppress"]:
                ci = buf.find(state["close_tag"])
                if ci == -1:
                    if len(buf) > 20000:
                        state["buf"] = ""
                    return "".join(out)
                state["buf"] = buf[ci + len(state["close_tag"]):]
                state["suppress"] = False
                state["close_tag"] = None
                continue
            oi = buf.find("<")
            if oi == -1:
                out.append(buf)
                state["buf"] = ""
                return "".join(out)
            if oi > 0:
                out.append(buf[:oi])
                buf = buf[oi:]
                state["buf"] = buf
            matched = None
            for i, pfx in enumerate(prefixes):
                if buf.startswith(pfx):
                    nxt = buf[len(pfx):len(pfx) + 1]
                    if nxt == "":
                        return "".join(out)
                    if nxt in (">", " ", "\\t", "\\n", "/"):
                        matched = tag_names[i]
                        break
            if matched:
                gi = buf.find(">")
                if gi == -1:
                    return "".join(out)
                state["buf"] = buf[gi + 1:]
                state["suppress"] = True
                state["close_tag"] = f"</{matched}>"
                continue
            is_partial = any(pfx.startswith(buf) for pfx in prefixes) and len(buf) <= max_prefix_len
            if is_partial:
                return "".join(out)
            out.append(buf[:1])
            state["buf"] = buf[1:]

    def flush() -> str:
        if state["suppress"]:
            state["buf"] = ""
            return ""
        remaining = state["buf"]
        state["buf"] = ""
        return remaining

    return feed, flush


_EVENT_ALARM_TAG_RE = re.compile(
    r'<(event_create|event_update|event_close|alarm_set|alarm_cancel)\\b[^>]*>(.*?)</\\1>',
    re.DOTALL,
)
_EVENT_ALARM_ATTR_RE = re.compile(r'(\\w+)\\s*=\\s*"([^"]*)"')


def _extract_event_alarm_tags(text: str) -> tuple[str, list[dict]]:
    """从文本中提取并剥离事件线/闹钟标签，返回 (剥离后文本, 动作列表)。"""
    actions: list[dict] = []

    def _repl(m):
        tag = m.group(1)
        # 重新在整段匹配文本中解析属性（group(0) 含开标签）
        open_tag_end = m.group(0).find(">")
        attrs_raw = m.group(0)[:open_tag_end] if open_tag_end >= 0 else ""
        attrs = dict(_EVENT_ALARM_ATTR_RE.findall(attrs_raw))
        inner = m.group(2)
        actions.append({"tag": tag, "attrs": attrs, "text": inner.strip()})
        return ""

    stripped = _EVENT_ALARM_TAG_RE.sub(_repl, text)
    return stripped, actions
'''

must_replace(OLD_LOGGER, NEW_LOGGER, "模块级常量+过滤器")


# ── 2. __init__：初始化 EventLine / AlarmClock ──────────────────────────────

OLD_INIT = '''        # ── PersonaManager 动态人设（已关闭自动采集，dashboard 手动编辑仍可用）──
        self._persona_manager = None
        self._memory_hint = ""
        logger.info("PersonaManager disabled (auto-collect & nightly update off)")'''

NEW_INIT = OLD_INIT + '''

        # ── EventLine / AlarmClock 事件线 + 闹钟（Phase 8）──
        try:
            from event_line import EventLine as _EventLine
            from alarm_clock import AlarmClock as _AlarmClock
            import cache_activity_zone as _cache_zone
            _p8_state_dir = str(self.config.get("state_dir") or "/state")
            _p8_persona_file = _p8_state_dir + "/system_standard.txt"
            _cache_zone.ensure_all_blocks(_p8_persona_file)
            self._event_line = _EventLine(state_dir=_p8_state_dir, persona_file=_p8_persona_file)
            self._alarm_clock = _AlarmClock(state_dir=_p8_state_dir, persona_file=_p8_persona_file)
            logger.info("EventLine & AlarmClock initialized")
        except Exception as _e8:
            self._event_line = None
            self._alarm_clock = None
            logger.warning("EventLine/AlarmClock init failed (non-fatal): %s", _e8)'''

must_replace(OLD_INIT, NEW_INIT, "__init__ EventLine/AlarmClock")


# ── 3. Phase6 Step2-pre：历史剥离扩展到 5 个新标签 ───────────────────────────

OLD_STEP2PRE = '''        # -- Phase 6 Step2-pre: 从 incoming 消息历史里剥离 memory_note 残留 --
        try:
            import re as _re_mn
            _MN_RE = _re_mn.compile(r'<memory_note>.*?</memory_note>', _re_mn.DOTALL)
            for _mi, _msg in enumerate(forward_payload.get("messages", [])):
                if not isinstance(_msg, dict):
                    continue
                if _msg.get("role") != "assistant":
                    continue
                _mc = _msg.get("content", "")
                if isinstance(_mc, str) and "<memory_note>" in _mc:
                    forward_payload["messages"][_mi] = dict(_msg, content=_MN_RE.sub("", _mc).rstrip())
                elif isinstance(_mc, list):
                    _new_blocks = []
                    for _blk in _mc:
                        if isinstance(_blk, dict) and _blk.get("type") == "text":
                            _bt = _blk.get("text", "")
                            if "<memory_note>" in _bt:
                                _blk = dict(_blk, text=_MN_RE.sub("", _bt).rstrip())
                        _new_blocks.append(_blk)
                    forward_payload["messages"][_mi] = dict(_msg, content=_new_blocks)
        except Exception as _mn_pre_e:
            logger.debug("memory_note history strip error | %s", _mn_pre_e)'''

NEW_STEP2PRE = '''        # -- Phase 6 Step2-pre: 从 incoming 消息历史里剥离 memory_note / 事件线 / 闹钟 标签残留 --
        try:
            import re as _re_mn
            _MN_RE = _re_mn.compile(
                r'<(memory_note|event_create|event_update|event_close|alarm_set|alarm_cancel)\\b[^>]*>.*?</\\1>',
                _re_mn.DOTALL,
            )
            _MN_MARKERS = ("<memory_note>", "<event_create", "<event_update", "<event_close", "<alarm_set", "<alarm_cancel")
            for _mi, _msg in enumerate(forward_payload.get("messages", [])):
                if not isinstance(_msg, dict):
                    continue
                if _msg.get("role") != "assistant":
                    continue
                _mc = _msg.get("content", "")
                if isinstance(_mc, str) and any(_mk in _mc for _mk in _MN_MARKERS):
                    forward_payload["messages"][_mi] = dict(_msg, content=_MN_RE.sub("", _mc).rstrip())
                elif isinstance(_mc, list):
                    _new_blocks = []
                    for _blk in _mc:
                        if isinstance(_blk, dict) and _blk.get("type") == "text":
                            _bt = _blk.get("text", "")
                            if any(_mk in _bt for _mk in _MN_MARKERS):
                                _blk = dict(_blk, text=_MN_RE.sub("", _bt).rstrip())
                        _new_blocks.append(_blk)
                    forward_payload["messages"][_mi] = dict(_msg, content=_new_blocks)
        except Exception as _mn_pre_e:
            logger.debug("memory_note/event/alarm history strip error | %s", _mn_pre_e)'''

must_replace(OLD_STEP2PRE, NEW_STEP2PRE, "Step2-pre 历史剥离")


# ── 4. path1 (OpenAI 风格 SSE) 接入新标签过滤 ────────────────────────────────

OLD_P1_SETUP = '''            _MN_OPEN = "<memory_note>"
            _MN_CLOSE = "</memory_note>"

            def _p1_filter_delta(raw_delta: str) -> str:'''
NEW_P1_SETUP = '''            _MN_OPEN = "<memory_note>"
            _MN_CLOSE = "</memory_note>"
            _p1_ea_feed, _p1_ea_flush = _make_attr_tag_filter(_NEW_TAG_NAMES)

            def _p1_filter_delta(raw_delta: str) -> str:'''
must_replace(OLD_P1_SETUP, NEW_P1_SETUP, "path1 filter 初始化")

OLD_P1_RET1 = '''                            keep = min(len(_MN_CLOSE) - 1, len(_p1_txt_buf))
                            _p1_txt_buf = _p1_txt_buf[-keep:] if keep else ""
                            return out
                    else:
                        oi = _p1_txt_buf.find(_MN_OPEN)'''
NEW_P1_RET1 = '''                            keep = min(len(_MN_CLOSE) - 1, len(_p1_txt_buf))
                            _p1_txt_buf = _p1_txt_buf[-keep:] if keep else ""
                            return _p1_ea_feed(out)
                    else:
                        oi = _p1_txt_buf.find(_MN_OPEN)'''
must_replace(OLD_P1_RET1, NEW_P1_RET1, "path1 return 1")

OLD_P1_RET2 = '''                            out += _p1_txt_buf[:safe]
                            _p1_txt_buf = _p1_txt_buf[safe:]
                            return out

            def _p1_filter_chunk(raw_bytes: bytes) -> bytes:'''
NEW_P1_RET2 = '''                            out += _p1_txt_buf[:safe]
                            _p1_txt_buf = _p1_txt_buf[safe:]
                            return _p1_ea_feed(out)

            def _p1_filter_chunk(raw_bytes: bytes) -> bytes:'''
must_replace(OLD_P1_RET2, NEW_P1_RET2, "path1 return 2")

OLD_P1_FLUSH = '''                if _p1_txt_buf and not _p1_supp:
                    # 残留文本不含标签，包装成最后一个 data 事件发出
                    try:
                        _flush_evt = _json.dumps({"choices": [{"delta": {"content": _p1_txt_buf}, "index": 0, "finish_reason": None}]})
                        yield ("data: " + _flush_evt + "\\n\\n").encode("utf-8")
                    except Exception:
                        pass'''
NEW_P1_FLUSH = '''                _p1_flush_text = (_p1_txt_buf if (_p1_txt_buf and not _p1_supp) else "") + _p1_ea_flush()
                if _p1_flush_text:
                    # 残留文本不含标签，包装成最后一个 data 事件发出
                    try:
                        _flush_evt = _json.dumps({"choices": [{"delta": {"content": _p1_flush_text}, "index": 0, "finish_reason": None}]})
                        yield ("data: " + _flush_evt + "\\n\\n").encode("utf-8")
                    except Exception:
                        pass'''
must_replace(OLD_P1_FLUSH, NEW_P1_FLUSH, "path1 flush")


# ── 5. path2 (Anthropic 风格 SSE) 接入新标签过滤 ─────────────────────────────

OLD_P2_SETUP = '''            _MN_OPEN = "<memory_note>"
            _MN_CLOSE = "</memory_note>"

            def _mn_filter(raw: str) -> str:'''
NEW_P2_SETUP = '''            _MN_OPEN = "<memory_note>"
            _MN_CLOSE = "</memory_note>"
            _mn_ea_feed, _mn_ea_flush = _make_attr_tag_filter(_NEW_TAG_NAMES)

            def _mn_filter(raw: str) -> str:'''
must_replace(OLD_P2_SETUP, NEW_P2_SETUP, "path2 filter 初始化")

OLD_P2_RET1 = '''                            keep = min(len(_MN_CLOSE) - 1, len(_mn_buf))
                            _mn_buf = _mn_buf[-keep:] if keep else ""
                            return out
                    else:
                        oi = _mn_buf.find(_MN_OPEN)'''
NEW_P2_RET1 = '''                            keep = min(len(_MN_CLOSE) - 1, len(_mn_buf))
                            _mn_buf = _mn_buf[-keep:] if keep else ""
                            return _mn_ea_feed(out)
                    else:
                        oi = _mn_buf.find(_MN_OPEN)'''
must_replace(OLD_P2_RET1, NEW_P2_RET1, "path2 return 1")

OLD_P2_RET2 = '''                            out += _mn_buf[:safe]
                            _mn_buf = _mn_buf[safe:]
                            return out
            next_block_index = 0'''
NEW_P2_RET2 = '''                            out += _mn_buf[:safe]
                            _mn_buf = _mn_buf[safe:]
                            return _mn_ea_feed(out)
            next_block_index = 0'''
must_replace(OLD_P2_RET2, NEW_P2_RET2, "path2 return 2")

OLD_P2_FLUSH = '''                # flush memory_note filter buffer（防止末尾正常文字被吞）
                if _mn_buf and not _mn_supp and text_block_index is not None:'''
NEW_P2_FLUSH = '''                # flush memory_note / 事件线 / 闹钟 filter buffer（防止末尾正常文字被吞）
                _mn_buf = (_mn_buf if (_mn_buf and not _mn_supp) else "") + _mn_ea_flush()
                if _mn_buf and text_block_index is not None:'''
must_replace(OLD_P2_FLUSH, NEW_P2_FLUSH, "path2 flush")


# ── 6. 移除 Phase6 Step2b（时间线迁移进 system_standard.txt，断点释放）───────────

OLD_STEP2B = '''
        # -- Phase 6 Step2b: inject sealed timeline as cached prefix block --
        try:
            _tl2b = getattr(getattr(self, "context_phase5", None), "daily_timeline", None)
            if _tl2b and forward_payload.get("messages"):
                _sealed2b = _tl2b.get_sealed_text()
                if _sealed2b:
                    _um2b = forward_payload["messages"][-1]
                    if isinstance(_um2b, dict) and _um2b.get("role") == "user":
                        _orig2b = _um2b.get("content", "")
                        _cb2b = {
                            "type": "text",
                            "text": "今日时间线（已封存）：\\n" + _sealed2b,
                            "cache_control": {"type": "ephemeral"},
                        }
                        if isinstance(_orig2b, str):
                            _nc2b = [_cb2b, {"type": "text", "text": _orig2b}]
                        elif isinstance(_orig2b, list):
                            _nc2b = [_cb2b] + list(_orig2b)
                        else:
                            _nc2b = [_cb2b]
                        forward_payload["messages"][-1] = dict(_um2b, content=_nc2b)
                        logger.info(
                            "Phase6 Step2b: sealed timeline cached | lines=%d",
                            _sealed2b.count("\\n") + 1,
                        )
        except Exception as _e6s2b:
            logger.warning("Phase6 Step2b: failed | %s", _e6s2b)
'''

NEW_STEP2B = '''
        # -- Phase 6 Step2b: 已废弃。时间线现同步进 system_standard.txt 的 DAILY_TIMELINE
        #    缓存区块（见 daily_timeline.sync_to_system_standard），复用 system message 的
        #    cache_control 断点，不再单独占用 Anthropic 4 断点上限中的一个。
        try:
            _al_periodic = getattr(self, "_alarm_clock", None)
            if _al_periodic is not None:
                _al_periodic.sync()
        except Exception as _e6s2b:
            logger.warning("Phase6 Step2b (alarm periodic refresh): failed | %s", _e6s2b)
'''

must_replace(OLD_STEP2B, NEW_STEP2B, "Step2b 移除+闹钟周期刷新")


# ── 7. _record_conversation_turn：提取并处理事件线/闹钟标签 ───────────────────

OLD_PM_PRE = '''        # ── PersonaManager: 提前提取 memory_note（在 early return 之前）──
        _pm_pre = getattr(self, "_persona_manager", None)'''

NEW_PM_PRE = '''        # ── EventLine/AlarmClock: 提取并处理事件线/闹钟标签（在 early return 之前）──
        _el_pre = getattr(self, "_event_line", None)
        _al_pre = getattr(self, "_alarm_clock", None)
        if (_el_pre is not None or _al_pre is not None) and isinstance(assistant_message, dict):
            try:
                _raw_ea = self._coerce_message_text(assistant_message.get("content"))
                if _raw_ea and ("<event_" in _raw_ea or "<alarm_" in _raw_ea):
                    _stripped_ea, _actions_ea = _extract_event_alarm_tags(_raw_ea)
                    if _actions_ea:
                        asyncio.create_task(
                            self._handle_event_alarm_actions(_actions_ea, session_id)
                        )
                        if isinstance(assistant_message.get("content"), str):
                            assistant_message = dict(assistant_message)
                            assistant_message["content"] = _stripped_ea
            except Exception as _eae:
                logger.debug("EventLine/AlarmClock tag extract error | %s", _eae)

        # ── PersonaManager: 提前提取 memory_note（在 early return 之前）──
        _pm_pre = getattr(self, "_persona_manager", None)'''

must_replace(OLD_PM_PRE, NEW_PM_PRE, "_record_conversation_turn 标签提取")


# ── 8. 新增 _handle_event_alarm_actions 方法 ─────────────────────────────────

OLD_METHOD_ANCHOR = '''    def _record_conversation_turn(
        self,
        *,
        session_id: str,
        round_id: int,'''

NEW_METHOD_ANCHOR = '''    async def _handle_event_alarm_actions(self, actions: list[dict], session_id: str) -> None:
        """处理从 AI 回复中提取到的事件线/闹钟标签动作，异步任务调用，不阻塞主流程。"""
        _el = getattr(self, "_event_line", None)
        _al = getattr(self, "_alarm_clock", None)

        def _to_int_or_none(v):
            try:
                return int(str(v).strip())
            except Exception:
                return None

        for action in actions:
            tag = action.get("tag")
            attrs = action.get("attrs", {}) or {}
            text = action.get("text", "")
            try:
                if tag == "event_create" and _el is not None:
                    _el.create_event(attrs.get("title", ""), text, _to_int_or_none(attrs.get("progress")))
                elif tag == "event_update" and _el is not None:
                    ok, err = _el.append_entry(attrs.get("id", ""), text, _to_int_or_none(attrs.get("progress")))
                    if not ok:
                        logger.info("event_update rejected | session=%s id=%s reason=%s", session_id, attrs.get("id"), err)
                elif tag == "event_close" and _el is not None:
                    closed = _el.close_event(attrs.get("id", ""), text)
                    if closed:
                        try:
                            from event_line import EventLine as _EL_cls
                            bucket_text = _EL_cls.format_for_bucket(closed)
                            await self.bucket_mgr.create(
                                content=bucket_text,
                                tags=["事件线归档"],
                                name=closed.get("title", "")[:40],
                                source="event_line_archive",
                            )
                            logger.info("EventLine archived to bucket | session=%s id=%s", session_id, attrs.get("id"))
                        except Exception as _arc_e:
                            logger.warning("EventLine archive to bucket failed | %s", _arc_e)
                    else:
                        logger.info("event_close: id not found | session=%s id=%s", session_id, attrs.get("id"))
                elif tag == "alarm_set" and _al is not None:
                    _al.create_alarm(
                        title=attrs.get("title", ""),
                        trigger_date=attrs.get("trigger_date", ""),
                        lead_days=_to_int_or_none(attrs.get("lead_days")) or 3,
                        note=text,
                        recurrence=attrs.get("recurrence", "none"),
                    )
                elif tag == "alarm_cancel" and _al is not None:
                    if not _al.cancel_alarm(attrs.get("id", "")):
                        logger.info("alarm_cancel: id not found | session=%s id=%s", session_id, attrs.get("id"))
            except Exception as _e:
                logger.warning("event/alarm tag handling failed | tag=%s session=%s err=%s", tag, session_id, _e)

    def _record_conversation_turn(
        self,
        *,
        session_id: str,
        round_id: int,'''

must_replace(OLD_METHOD_ANCHOR, NEW_METHOD_ANCHOR, "新增 _handle_event_alarm_actions 方法")


GW.write_text(gw, encoding="utf-8")
print(f"\\n文件大小变化: {orig_len} -> {len(gw)} 字符")

r = subprocess.run(["python3", "-m", "py_compile", "/opt/Ombre-Brain/gateway.py"], capture_output=True, text=True)
if r.returncode != 0:
    print("❌ gateway.py 语法错误:", r.stderr[:2000])
    raise SystemExit(1)
print("✅ gateway.py 语法验证通过")
