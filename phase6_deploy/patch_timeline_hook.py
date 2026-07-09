"""
patch_timeline_hook.py
在 gateway.py 做两处插入：
  1. __init__ 末尾：初始化 DailyTimeline（在 context_phase5 初始化块之后）
  2. _record_conversation_turn：Phase5 环写入之后插入 maybe_update 调用
"""
import sys
from pathlib import Path

GW = Path("/opt/Ombre-Brain/gateway.py")
src = GW.read_text(encoding="utf-8")

# ── 补丁 1：__init__ 初始化 DailyTimeline ──────────────────────────────────
INIT_ANCHOR = """        except Exception as _e:
            self.context_phase5 = None"""

INIT_INSERT = """        except Exception as _e:
            self.context_phase5 = None

        # ── DailyTimeline 每日时间线（Phase 7）──
        try:
            from daily_timeline import DailyTimeline as _DailyTimeline
            _tl_p5 = getattr(self, "context_phase5", None)
            if _tl_p5 is not None:
                _tl_p5.daily_timeline = _DailyTimeline(config=self.config)
                logger.info("daily_timeline initialized")
        except Exception as _etl:
            logger.warning("daily_timeline init failed (non-fatal): %s", _etl)"""

if "daily_timeline initialized" in src:
    print("SKIP patch1: already applied")
elif INIT_ANCHOR not in src:
    print("ERROR patch1: anchor not found")
    sys.exit(1)
else:
    src = src.replace(INIT_ANCHOR, INIT_INSERT, 1)
    print("OK patch1: DailyTimeline __init__ inserted")

# ── 补丁 2：_record_conversation_turn hook ─────────────────────────────────
HOOK_ANCHOR = """        # ── Phase 5：写入 7 轮上下文环 ──
        if getattr(self, "context_phase5", None):
            self.context_phase5.record_turn_if_eligible(user_text, assistant_text)"""

HOOK_INSERT = """        # ── Phase 5：写入 7 轮上下文环 ──
        if getattr(self, "context_phase5", None):
            self.context_phase5.record_turn_if_eligible(user_text, assistant_text)

        # ── DailyTimeline 时间线更新（Phase 7）──
        _tl = getattr(getattr(self, "context_phase5", None), "daily_timeline", None)
        if _tl is not None:
            _tl_asst = str(
                assistant_message.get("content", "")
                if isinstance(assistant_message, dict)
                else assistant_message or ""
            ).strip()[:600]
            asyncio.create_task(
                _tl.maybe_update(
                    user_text=user_text or "",
                    assistant_text=_tl_asst,
                )
            )"""

if "DailyTimeline 时间线更新" in src:
    print("SKIP patch2: already applied")
elif HOOK_ANCHOR not in src:
    print("ERROR patch2: anchor not found")
    sys.exit(1)
else:
    src = src.replace(HOOK_ANCHOR, HOOK_INSERT, 1)
    print("OK patch2: maybe_update hook inserted")

GW.write_text(src, encoding="utf-8")
print("gateway.py written OK")
