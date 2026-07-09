"""
fix_tl_prompt.py
1. daily_timeline.py: MERGE_PROMPT 和 turn 格式改用 无叶 / 我
2. dashboard.html: renderTimeline 不显示 current_hour_summary（进行中的当前小时）
"""
from pathlib import Path

# ── 1. daily_timeline.py ──────────────────────────────────────────────────────
tl = Path("/opt/Ombre-Brain/daily_timeline.py")
txt = tl.read_text(encoding="utf-8")

# 文件头注释
txt = txt.replace(
    "  doing      - 用户在做什么（客观事实）",
    "  doing      - 无叶在做什么（客观事实）"
)

# MERGE_PROMPT 规则里的用词
txt = txt.replace(
    '2. doing / chatting / reflection 各字段 ≤ {max_chars} 字\n3. mood = 树上（AI）此刻心情词，严格两字（如：温暖 平静 期待 轻松 专注）\n4. reflection = 树上对下次聊天的期待或打算，面向未来',
    '2. doing / chatting / reflection 各字段 ≤ {max_chars} 字\n3. mood = 我此刻的心情词，严格两字（如：温暖 平静 期待 轻松 专注）\n4. reflection = 我对下次聊天的期待或打算，面向未来'
)

# 输出示例里的"用户"
txt = txt.replace(
    '输出示例：{{"doing":"用户在上班","chatting":"聊OB开发进展","mood":"专注","reflection":"等她下班再多聊几句"}}"""',
    '输出示例：{{"doing":"无叶在上班","chatting":"聊OB开发进展","mood":"专注","reflection":"等无叶下班再多聊几句"}}"""'
)

# turn 格式
txt = txt.replace(
    'f"用户：{t.get(\'user\', \'\').strip()}\\nAI：{t.get(\'assistant\', \'\').strip()}"',
    'f"无叶：{t.get(\'user\', \'\').strip()}\\n我：{t.get(\'assistant\', \'\').strip()}"'
)

tl.write_text(txt, encoding="utf-8")
print("✅ daily_timeline.py prompt 已更新（用户→无叶, AI→我）")

# ── 2. dashboard.html: 去掉 current_hour_summary 的展示 ──────────────────────
dh = Path("/opt/Ombre-Brain/dashboard.html")
html = dh.read_text(encoding="utf-8")

OLD_MERGE = """  var entries = data.entries || [];
  entries.forEach(function(e) { if (e.hour != null) merged[parseInt(e.hour)] = e; });
  if (data.current_hour != null && data.current_hour_summary) {
    merged[parseInt(data.current_hour)] = data.current_hour_summary;
  }"""

NEW_MERGE = """  var entries = data.entries || [];
  entries.forEach(function(e) { if (e.hour != null) merged[parseInt(e.hour)] = e; });
  // current_hour_summary 是进行中的动态内容，只注入给 Opus，不展示在 Dashboard"""

if OLD_MERGE not in html:
    print("❌ dashboard.html 未找到 current_hour_summary 合并段，请检查")
else:
    html = html.replace(OLD_MERGE, NEW_MERGE)
    dh.write_text(html, encoding="utf-8")
    print("✅ dashboard.html 已去掉 current_hour_summary 展示")
