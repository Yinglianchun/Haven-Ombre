"""
patch_dashboard_timeline.py
给 dashboard.html 添加：
  1. CSS: 时间线卡片样式
  2. HTML: reflection-day-panel 后加 #tl-section
  3. JS:  loadTimeline / renderTimeline / 编辑弹窗
  4. 在 selectReflectionDate 里钩入 loadTimeline
"""
import sys
from pathlib import Path

DASH = Path("/opt/Ombre-Brain/dashboard.html")
src = DASH.read_text(encoding="utf-8")

if "tl-section" in src:
    print("SKIP: already patched")
    sys.exit(0)

# ── 1. CSS ─────────────────────────────────────────────────────────────────
CSS_ANCHOR = ".reflection-day-panel {"
if CSS_ANCHOR not in src:
    print("ERROR: CSS anchor not found")
    sys.exit(1)

CSS_INSERT = """\
/* ── Daily Timeline Cards ── */
#tl-section { margin-top: 16px; }
.tl-card {
  background: var(--card-bg, #fff);
  border: 1px solid var(--border, #e5e7eb);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 8px;
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.tl-hour {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent, #7c6ef7);
  min-width: 36px;
  padding-top: 2px;
}
.tl-body { flex: 1; font-size: 13px; line-height: 1.5; }
.tl-field { margin-bottom: 3px; }
.tl-label { font-size: 11px; color: var(--text-light, #9ca3af); margin-right: 4px; }
.tl-edit-btn {
  border: none; background: none; cursor: pointer;
  color: var(--text-light, #9ca3af); font-size: 13px; padding: 2px 4px;
}
.tl-edit-btn:hover { color: var(--accent, #7c6ef7); }
/* Modal */
#tl-modal {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.45); z-index: 9000;
  align-items: center; justify-content: center;
}
#tl-modal.open { display: flex; }
#tl-modal-box {
  background: var(--card-bg, #fff);
  border-radius: 12px; padding: 24px 28px;
  width: 420px; max-width: 94vw;
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
#tl-modal h3 { margin: 0 0 14px; font-size: 15px; }
.tl-modal-row { margin-bottom: 10px; }
.tl-modal-label { font-size: 12px; color: var(--text-light, #9ca3af); margin-bottom: 3px; }
.tl-modal-input {
  width: 100%; box-sizing: border-box;
  border: 1px solid var(--border, #e5e7eb);
  border-radius: 6px; padding: 6px 8px;
  font-size: 13px; resize: vertical;
  background: var(--input-bg, #f9fafb);
}
.tl-modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 14px; }
.tl-modal-cancel { padding: 6px 16px; border: 1px solid var(--border, #e5e7eb); border-radius: 6px; background: none; cursor: pointer; font-size: 13px; }
.tl-modal-save { padding: 6px 16px; border: none; border-radius: 6px; background: var(--accent, #7c6ef7); color: #fff; cursor: pointer; font-size: 13px; }

"""

src = src.replace(CSS_ANCHOR, CSS_INSERT + CSS_ANCHOR, 1)
print("OK: CSS inserted")

# ── 2. HTML ────────────────────────────────────────────────────────────────
HTML_ANCHOR = '<section class="reflection-card reflection-day-panel" id="reflection-day-panel"></section>'
if HTML_ANCHOR not in src:
    print("ERROR: HTML anchor not found")
    sys.exit(1)

HTML_INSERT = (
    HTML_ANCHOR
    + "\n"
    + "      <section class=\"reflection-card\" id=\"tl-section\" style=\"display:none\">"
    + "<div class=\"reflection-card-head\"><h3>每小时记录</h3>"
    + "<span id=\"tl-section-label\" style=\"font-size:12px;color:var(--text-light)\"></span></div>"
    + "<div id=\"tl-cards\"></div></section>\n"
    + "<!-- Timeline edit modal -->\n"
    + "<div id=\"tl-modal\"><div id=\"tl-modal-box\">\n"
    + "  <h3 id=\"tl-modal-title\">编辑时间线</h3>\n"
    + "  <div class=\"tl-modal-row\"><div class=\"tl-modal-label\">在做什么 (doing)</div>"
    + "<textarea class=\"tl-modal-input\" id=\"tl-f-doing\" rows=\"2\"></textarea></div>\n"
    + "  <div class=\"tl-modal-row\"><div class=\"tl-modal-label\">聊了什么 (chatting)</div>"
    + "<textarea class=\"tl-modal-input\" id=\"tl-f-chatting\" rows=\"2\"></textarea></div>\n"
    + "  <div class=\"tl-modal-row\"><div class=\"tl-modal-label\">AI 心情 (mood)</div>"
    + "<textarea class=\"tl-modal-input\" id=\"tl-f-mood\" rows=\"2\"></textarea></div>\n"
    + "  <div class=\"tl-modal-row\"><div class=\"tl-modal-label\">AI 展望 (reflection)</div>"
    + "<textarea class=\"tl-modal-input\" id=\"tl-f-reflection\" rows=\"2\"></textarea></div>\n"
    + "  <div class=\"tl-modal-actions\">"
    + "<button class=\"tl-modal-cancel\" onclick=\"closeTlModal()\">取消</button>"
    + "<button class=\"tl-modal-save\" onclick=\"saveTlModal()\">保存</button></div>\n"
    + "</div></div>\n"
)

src = src.replace(HTML_ANCHOR, HTML_INSERT, 1)
print("OK: HTML inserted")

# ── 3. JS ──────────────────────────────────────────────────────────────────
JS_ANCHOR = "function selectReflectionDate(dateStr) {"
if JS_ANCHOR not in src:
    print("ERROR: JS anchor not found")
    sys.exit(1)

JS_FUNCTIONS = """\
// ── Daily Timeline ──
var _tlCurrentDate = null;
var _tlCurrentHour = null;

async function loadTimeline(dateStr) {
  _tlCurrentDate = dateStr;
  var section = document.getElementById('tl-section');
  var label = document.getElementById('tl-section-label');
  var cards = document.getElementById('tl-cards');
  if (!section || !cards) return;
  section.style.display = '';
  label.textContent = dateStr;
  cards.innerHTML = '<div class="loading" style="padding:8px 0;font-size:13px">加载时间线…</div>';
  try {
    var res = await authFetch(BASE + '/api/timeline/' + dateStr);
    if (!res || res.status === 404) {
      cards.innerHTML = '<div style="font-size:13px;color:var(--text-light);padding:8px 0">这一天暂无记录。</div>';
      return;
    }
    var data = await res.json();
    renderTimeline(dateStr, data);
  } catch(e) {
    cards.innerHTML = '<div style="color:#ef4444;font-size:13px">加载失败: ' + e.message + '</div>';
  }
}

function renderTimeline(dateStr, data) {
  var cards = document.getElementById('tl-cards');
  var hours = data.hours || {};
  var keys = Object.keys(hours).map(Number).sort(function(a,b){return a-b;});
  if (!keys.length) {
    cards.innerHTML = '<div style="font-size:13px;color:var(--text-light);padding:8px 0">这一天暂无记录。</div>';
    return;
  }
  var fieldLabels = {doing:'在做', chatting:'聊了', mood:'心情', reflection:'展望'};
  var html = keys.map(function(h) {
    var e = hours[h] || {};
    var hStr = String(h).padStart(2,'0') + ':00';
    var body = ['doing','chatting','mood','reflection'].map(function(f) {
      var v = (e[f] || '').trim();
      if (!v) return '';
      return '<div class="tl-field"><span class="tl-label">' + fieldLabels[f] + '</span>' + escHtml(v) + '</div>';
    }).filter(Boolean).join('');
    if (!body) body = '<div class="tl-field" style="color:var(--text-light)">（空）</div>';
    return '<div class="tl-card">'
      + '<div class="tl-hour">' + hStr + '</div>'
      + '<div class="tl-body">' + body + '</div>'
      + '<button class="tl-edit-btn" title="编辑" onclick="openTlModal(\'' + dateStr + '\',' + h + ',' + JSON.stringify(e).replace(/'/g,"\\'") + ')">✏</button>'
      + '</div>';
  }).join('');
  cards.innerHTML = html;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function openTlModal(dateStr, hour, entry) {
  _tlCurrentDate = dateStr;
  _tlCurrentHour = hour;
  document.getElementById('tl-modal-title').textContent = dateStr + ' ' + String(hour).padStart(2,'0') + ':00 编辑';
  ['doing','chatting','mood','reflection'].forEach(function(f){
    var el = document.getElementById('tl-f-' + f);
    if (el) el.value = (entry && entry[f]) || '';
  });
  document.getElementById('tl-modal').classList.add('open');
}

function closeTlModal() {
  document.getElementById('tl-modal').classList.remove('open');
}

async function saveTlModal() {
  if (!_tlCurrentDate || _tlCurrentHour === null) return;
  var body = {};
  ['doing','chatting','mood','reflection'].forEach(function(f){
    var el = document.getElementById('tl-f-' + f);
    if (el) body[f] = el.value.trim();
  });
  try {
    var res = await authFetch(
      BASE + '/api/timeline/' + _tlCurrentDate + '/' + _tlCurrentHour,
      {method:'PATCH', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)}
    );
    if (res && res.ok) {
      closeTlModal();
      loadTimeline(_tlCurrentDate);
    } else {
      alert('保存失败');
    }
  } catch(e) {
    alert('保存出错: ' + e.message);
  }
}

"""

src = src.replace(JS_ANCHOR, JS_FUNCTIONS + JS_ANCHOR, 1)
print("OK: JS functions inserted")

# ── 4. Hook loadTimeline into selectReflectionDate ─────────────────────────
OLD_SELECT = """\
function selectReflectionDate(dateStr) {
  reflectionSelectedDate = dateStr;
  var parsed = parseLocalDate(dateStr);
  if (parsed) reflectionCurrentMonth = monthStart(parsed);
  renderReflectionReview();
}"""

NEW_SELECT = """\
function selectReflectionDate(dateStr) {
  reflectionSelectedDate = dateStr;
  var parsed = parseLocalDate(dateStr);
  if (parsed) reflectionCurrentMonth = monthStart(parsed);
  renderReflectionReview();
  loadTimeline(dateStr);
}"""

if OLD_SELECT not in src:
    print("ERROR: selectReflectionDate body not matched")
    sys.exit(1)
src = src.replace(OLD_SELECT, NEW_SELECT, 1)
print("OK: selectReflectionDate hooked")

DASH.write_text(src, encoding="utf-8")
print("dashboard.html written OK")
