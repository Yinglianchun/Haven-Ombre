"""
fix_tl_render2.py
把 dashboard.html 的 renderTimeline 改为同时读 hours{} 和 entries[]，
再加入 current_hour_summary，三路合并后统一渲染。
"""
from pathlib import Path

TARGET = Path("/opt/Ombre-Brain/dashboard.html")
text = TARGET.read_text(encoding="utf-8")

OLD = """function renderTimeline(dateStr, data) {
  var cards = document.getElementById('tl-cards');
  var hours = data.hours || {};
  var keys = Object.keys(hours).map(Number).sort(function(a,b){return a-b;});
  if (!keys.length) {
    cards.innerHTML = '<div style="font-size:13px;color:var(--text-light);padding:8px 0">这一天暂无记录。</div>';
    return;
  }
  var fieldLabels = {doing:'在做', chatting:'聊了', mood:'心情', reflection:'展望'};
  var html = keys.map(function(h) {
    var e = hours[h] || {};"""

NEW = """function renderTimeline(dateStr, data) {
  var cards = document.getElementById('tl-cards');
  // 合并三路数据：hours{}（手动）+ entries[]（自动封存）+ current_hour_summary（当前小时）
  var merged = {};
  var hours = data.hours || {};
  Object.keys(hours).forEach(function(k) { merged[parseInt(k)] = hours[k]; });
  var entries = data.entries || [];
  entries.forEach(function(e) { if (e.hour != null) merged[parseInt(e.hour)] = e; });
  if (data.current_hour != null && data.current_hour_summary) {
    merged[parseInt(data.current_hour)] = data.current_hour_summary;
  }
  var keys = Object.keys(merged).map(Number).sort(function(a,b){return a-b;});
  if (!keys.length) {
    cards.innerHTML = '<div style="font-size:13px;color:var(--text-light);padding:8px 0">这一天暂无记录。</div>';
    return;
  }
  var fieldLabels = {doing:'在做', chatting:'聊了', mood:'心情', reflection:'展望'};
  var html = keys.map(function(h) {
    var e = merged[h] || {};"""

if OLD not in text:
    print("❌ 未找到目标片段，请检查 dashboard.html 版本")
    import sys; sys.exit(1)

# 同时把 _tlEntries = hours 改为 _tlEntries = merged
text2 = text.replace(OLD, NEW)
text2 = text2.replace("  _tlEntries = hours;\n  cards.innerHTML = html;",
                       "  _tlEntries = merged;\n  cards.innerHTML = html;")

TARGET.write_text(text2, encoding="utf-8")
print("✅ renderTimeline 已修复，支持 hours{} + entries[] + current_hour_summary 三路合并")
