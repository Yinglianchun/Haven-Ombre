"""
fix_tl_btn.py  – 用正则替换 renderTimeline 里的坏 onclick 按钮行
"""
import re
from pathlib import Path

DASH = Path("/opt/Ombre-Brain/dashboard.html")
src = DASH.read_text(encoding="utf-8")

# 匹配整个坏按钮行（不依赖精确转义字符）
BAD_PAT = re.compile(
    r"'\s*\+\s*'<button class=\"tl-edit-btn\"[^']*onclick=\"openTlModal\([^\"]*\)\"[^']*>✏</button>'",
    re.DOTALL
)

GOOD_LINE = (
    "      + '<button class=\"tl-edit-btn\" title=\"编辑\" "
    "data-tldate=\"' + escHtml(dateStr) + '\" data-tlhour=\"' + h + '\" "
    "onclick=\"openTlByAttr(this)\">✏</button>'"
)

m = BAD_PAT.search(src)
if m:
    print("Found broken btn line:", repr(m.group()[:80]))
    src = BAD_PAT.sub(GOOD_LINE, src, count=1)
    print("OK: replaced with data-attr version")
elif "openTlByAttr(this)" in src:
    print("SKIP: already fixed")
else:
    # fallback: look for the specific problematic substring
    idx = src.find("openTlModal(''")
    if idx < 0:
        idx = src.find("openTlModal(\\'\\'" )
    if idx < 0:
        print("ERROR: cannot find broken line")
        import sys; sys.exit(1)
    # find line boundaries
    start = src.rfind("\n", 0, idx) + 1
    end = src.find("\n", idx)
    print("Line to replace:", repr(src[start:end]))
    src = src[:start] + GOOD_LINE + src[end:]
    print("OK: fallback line replacement done")

DASH.write_text(src, encoding="utf-8")
print("written OK")
