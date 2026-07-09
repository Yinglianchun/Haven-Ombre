"""
fix_tl_layout.py
1. 让 #tl-section 横跨两列（grid-column: 1/-1）
2. 加大 tl-card 内边距 + 字号，缓解视觉拥挤
"""
from pathlib import Path

DASH = Path("/opt/Ombre-Brain/dashboard.html")
src = DASH.read_text(encoding="utf-8")

OLD_CSS = "/* ── Daily Timeline Cards ── */\n#tl-section { margin-top: 16px; }"
NEW_CSS = (
    "/* ── Daily Timeline Cards ── */\n"
    "#tl-section {\n"
    "  grid-column: 1 / -1;\n"
    "  margin-top: 4px;\n"
    "}\n"
    "#tl-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }"
)

if OLD_CSS not in src:
    print("ERROR: CSS anchor not found")
    # show what's there
    idx = src.find("#tl-section")
    if idx >= 0:
        print("Found #tl-section at:", idx)
        print(repr(src[idx:idx+100]))
    raise SystemExit(1)

src = src.replace(OLD_CSS, NEW_CSS, 1)

# 同时加大 tl-card padding 和字号
OLD_CARD = ".tl-card {\n  background: var(--card-bg, #fff);\n  border: 1px solid var(--border, #e5e7eb);\n  border-radius: 10px;\n  padding: 10px 14px;\n  margin-bottom: 8px;\n  display: flex;\n  gap: 10px;\n  align-items: flex-start;\n}"
NEW_CARD = ".tl-card {\n  background: var(--card-bg, #fff);\n  border: 1px solid var(--border, #e5e7eb);\n  border-radius: 12px;\n  padding: 14px 18px;\n  display: flex;\n  gap: 12px;\n  align-items: flex-start;\n}"

if OLD_CARD in src:
    src = src.replace(OLD_CARD, NEW_CARD, 1)
    print("OK: tl-card padding enlarged")
else:
    print("SKIP: tl-card not matched (may already be fixed)")

OLD_BODY = ".tl-body { flex: 1; font-size: 13px; line-height: 1.5; }"
NEW_BODY = ".tl-body { flex: 1; font-size: 13px; line-height: 1.7; }"
if OLD_BODY in src:
    src = src.replace(OLD_BODY, NEW_BODY, 1)
    print("OK: line-height increased")

DASH.write_text(src, encoding="utf-8")
print("OK: dashboard.html written")
