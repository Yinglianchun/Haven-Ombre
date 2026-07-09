"""
patch_timeline_step2b.py
在 Phase 6 Step2 之后插入 Step2b：
  把封存时间线作为带 cache_control 的第一个 content block 插入 user message。
"""
import sys
from pathlib import Path

GW = Path("/opt/Ombre-Brain/gateway.py")
src = GW.read_text(encoding="utf-8")

if "Step2b" in src:
    print("SKIP: Step2b already applied")
    sys.exit(0)

ANCHOR = (
    '                        logger.info("Phase6 Step2: messages[] replaced | count=2")\n'
    '        except Exception as _e6s2:\n'
    '            logger.warning("Phase6 Step2: failed | %s", _e6s2)'
)

if ANCHOR not in src:
    print("ERROR: anchor not found")
    idx = src.find("Phase6 Step2: messages[] replaced")
    if idx >= 0:
        print("Context:", repr(src[max(0,idx-20):idx+120]))
    sys.exit(1)

# Step2b 代码（注入进 gateway.py 的 Python 代码字符串）
STEP2B_CODE = (
    '        # -- Phase 6 Step2b: inject sealed timeline as cached prefix block --\n'
    '        try:\n'
    '            _tl2b = getattr(getattr(self, "context_phase5", None), "daily_timeline", None)\n'
    '            if _tl2b and forward_payload.get("messages"):\n'
    '                _sealed2b = _tl2b.get_sealed_text()\n'
    '                if _sealed2b:\n'
    '                    _um2b = forward_payload["messages"][-1]\n'
    '                    if isinstance(_um2b, dict) and _um2b.get("role") == "user":\n'
    '                        _orig2b = _um2b.get("content", "")\n'
    '                        _cb2b = {\n'
    '                            "type": "text",\n'
    '                            "text": "\u4eca\u65e5\u65f6\u95f4\u7ebf\uff08\u5df2\u5c01\u5b58\uff09\uff1a\\n" + _sealed2b,\n'
    '                            "cache_control": {"type": "ephemeral"},\n'
    '                        }\n'
    '                        if isinstance(_orig2b, str):\n'
    '                            _nc2b = [_cb2b, {"type": "text", "text": _orig2b}]\n'
    '                        elif isinstance(_orig2b, list):\n'
    '                            _nc2b = [_cb2b] + list(_orig2b)\n'
    '                        else:\n'
    '                            _nc2b = [_cb2b]\n'
    '                        forward_payload["messages"][-1] = dict(_um2b, content=_nc2b)\n'
    '                        logger.info(\n'
    '                            "Phase6 Step2b: sealed timeline cached | lines=%d",\n'
    '                            _sealed2b.count("\\n") + 1,\n'
    '                        )\n'
    '        except Exception as _e6s2b:\n'
    '            logger.warning("Phase6 Step2b: failed | %s", _e6s2b)\n'
)

src = src.replace(ANCHOR, ANCHOR + "\n\n" + STEP2B_CODE, 1)
GW.write_text(src, encoding="utf-8")
print("OK: Step2b inserted")
