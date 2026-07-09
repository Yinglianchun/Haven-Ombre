"""
fix_tl_render.py
修复 dashboard.html 里 renderTimeline 中 onclick 字符串转义错误。
用 data-date / data-hour 属性替换 inline JSON.stringify，彻底消除语法错误。
"""
from pathlib import Path

DASH = Path("/opt/Ombre-Brain/dashboard.html")
src = DASH.read_text(encoding="utf-8")

# ── 替换坏掉的按钮行 ────────────────────────────────────────────────────────
# 原来 (syntactically broken)
OLD_BTN = """      + '<button class="tl-edit-btn" title="编辑" onclick="openTlModal('' + dateStr + '',' + h + ',' + JSON.stringify(e).replace(/'/g,"\'") + ')">✏</button>'"""

# 修复后：用 data-* 属性，onclick 只传 this
NEW_BTN = """      + '<button class="tl-edit-btn" title="编辑" data-tldate="' + escHtml(dateStr) + '" data-tlhour="' + h + '" onclick="openTlByAttr(this)">✏</button>'"""

if OLD_BTN not in src:
    # 也许已经被修复或格式稍有不同，打印附近内容
    idx = src.find("openTlModal(''")
    if idx >= 0:
        print("FOUND at", idx, "context:", repr(src[max(0,idx-30):idx+80]))
    else:
        print("ERROR: broken line not found; checking alternate...")
        idx2 = src.find("tl-edit-btn")
        if idx2 >= 0:
            print("tl-edit-btn found at:", idx2)
            print(repr(src[idx2:idx2+200]))
        raise SystemExit(1)
else:
    src = src.replace(OLD_BTN, NEW_BTN, 1)
    print("OK: broken button line replaced")

# ── 插入 openTlByAttr 辅助函数（在 openTlModal 之前）──────────────────────
OLD_FUNC = "function openTlModal(dateStr, hour, entry) {"
NEW_FUNC = """\
var _tlEntries = {};

function openTlByAttr(btn) {
  var date = btn.getAttribute('data-tldate');
  var hour = parseInt(btn.getAttribute('data-tlhour'), 10);
  var entry = _tlEntries[hour] || {};
  openTlModal(date, hour, entry);
}

function openTlModal(dateStr, hour, entry) {"""

if "openTlByAttr" in src:
    print("SKIP: openTlByAttr already present")
elif OLD_FUNC not in src:
    print("ERROR: openTlModal function not found")
    raise SystemExit(1)
else:
    src = src.replace(OLD_FUNC, NEW_FUNC, 1)
    print("OK: openTlByAttr inserted")

# ── 修复 renderTimeline：在 cards.innerHTML 赋值前保存 _tlEntries ───────────
OLD_STORE = "  cards.innerHTML = html;\n}"
NEW_STORE = "  _tlEntries = hours;\n  cards.innerHTML = html;\n}"

# 只替换 renderTimeline 内的那一处（第一次出现）
render_idx = src.find("function renderTimeline(")
if render_idx < 0:
    print("ERROR: renderTimeline not found")
    raise SystemExit(1)

render_end = src.find("\nfunction ", render_idx + 1)
render_body = src[render_idx:render_end]
if "_tlEntries = hours" in render_body:
    print("SKIP: _tlEntries already stored in renderTimeline")
elif OLD_STORE in render_body:
    src = src[:render_idx] + render_body.replace(OLD_STORE, NEW_STORE, 1) + src[render_end:]
    print("OK: _tlEntries store inserted in renderTimeline")
else:
    print("WARNING: could not find 'cards.innerHTML = html;' in renderTimeline, skipping")

DASH.write_text(src, encoding="utf-8")
print("dashboard.html written OK")
