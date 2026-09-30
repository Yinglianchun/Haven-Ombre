"""
patch_dashboard_event_line.py
在人设 tab 后新增「事件线」页：按事件折叠，日期卡片展示每日记录，可编辑/删除。
"""
from __future__ import annotations

import sys
from pathlib import Path

DASH = Path("/opt/Ombre-Brain/dashboard.html")
src = DASH.read_text(encoding="utf-8")

if "event-line-view" in src:
    print("SKIP: event-line already patched")
    sys.exit(0)

# ── 1. CSS ─────────────────────────────────────────────────────────────────
CSS_ANCHOR = "/* ── Daily Timeline Cards ── */"
if CSS_ANCHOR not in src:
    CSS_ANCHOR = ".reflection-day-panel {"
CSS_INSERT = """\
/* ── Event Line Cards ── */
#event-line-view .reflection-board { display: block; max-width: 920px; }
.el-event-block {
  background: var(--card-bg, #fff);
  border: 1px solid var(--border, #e5e7eb);
  border-radius: 12px;
  margin-bottom: 14px;
  overflow: hidden;
}
.el-event-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  background: var(--soft-bg, #faf8f5);
  border-bottom: 1px solid var(--border, #eee);
}
.el-event-head:hover { background: #f3efe8; }
.el-event-chevron { font-size: 12px; color: var(--text-light, #9ca3af); width: 14px; }
.el-event-title { flex: 1; font-size: 15px; font-weight: 600; color: var(--text, #333); }
.el-event-meta { font-size: 12px; color: var(--text-light, #9ca3af); }
.el-event-actions { display: flex; gap: 6px; }
.el-event-actions button {
  border: none; background: none; cursor: pointer;
  font-size: 12px; color: var(--text-light, #9ca3af); padding: 4px 6px;
}
.el-event-actions button:hover { color: var(--accent, #8b6a42); }
.el-event-body { padding: 10px 12px 12px; }
.el-event-body.collapsed { display: none; }
.el-date-card {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border, #ece8e2);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 8px;
}
.el-date-label {
  min-width: 52px;
  font-size: 14px;
  font-weight: 700;
  color: #c45c4a;
  line-height: 1.4;
  padding-top: 2px;
}
.el-date-body { flex: 1; font-size: 13px; line-height: 1.55; }
.el-progress-row { margin-bottom: 4px; }
.el-progress-label { font-size: 11px; color: var(--text-light, #9ca3af); margin-right: 6px; }
.el-progress-val { font-size: 13px; font-weight: 600; color: var(--accent, #8b6a42); }
.el-entry-text { color: var(--text, #444); white-space: pre-wrap; word-break: break-word; }
.el-card-actions { display: flex; flex-direction: column; gap: 4px; }
.el-card-actions button {
  border: none; background: none; cursor: pointer;
  color: var(--text-light, #9ca3af); font-size: 13px; padding: 2px 4px;
}
.el-card-actions button:hover { color: var(--accent, #8b6a42); }
.el-card-actions .danger:hover { color: #ef4444; }
.el-stale-badge {
  display: inline-block;
  font-size: 11px;
  color: #b45309;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 999px;
  padding: 2px 8px;
  margin-left: 8px;
}
#el-modal {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.45); z-index: 9100;
  align-items: center; justify-content: center;
}
#el-modal.open { display: flex; }
#el-modal-box {
  background: var(--card-bg, #fff);
  border-radius: 12px; padding: 24px 28px;
  width: 460px; max-width: 94vw;
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
#el-modal h3 { margin: 0 0 14px; font-size: 15px; }
.el-modal-row { margin-bottom: 10px; }
.el-modal-label { font-size: 12px; color: var(--text-light, #9ca3af); margin-bottom: 3px; }
.el-modal-input {
  width: 100%; box-sizing: border-box;
  border: 1px solid var(--border, #e5e7eb);
  border-radius: 6px; padding: 6px 8px;
  font-size: 13px; resize: vertical;
  background: var(--input-bg, #f9fafb);
}

"""

src = src.replace(CSS_ANCHOR, CSS_INSERT + CSS_ANCHOR, 1)
print("OK: CSS")

# ── 2. Tab ─────────────────────────────────────────────────────────────────
TAB_OLD = '<div class="tab" data-tab="persona-edit">人设</div>'
TAB_NEW = TAB_OLD + '\n<div class="tab" data-tab="event-line">事件线</div>'
if TAB_OLD not in src:
    print("ERROR: persona-edit tab not found")
    sys.exit(1)
src = src.replace(TAB_OLD, TAB_NEW, 1)
print("OK: tab")

# ── 3. View HTML ───────────────────────────────────────────────────────────
VIEW_ANCHOR = '<div class="content" id="profile-view" style="display:none">'
VIEW_INSERT = '''\
<div class="content" id="event-line-view" style="display:none">
  <div class="reflection-shell">
    <div class="reflection-toolbar">
      <div class="reflection-title">
        <h2>事件线</h2>
        <p>跨天持续事件按日期记录每日进度；可折叠、编辑、删除。修改后自动同步到 AI 缓存区。</p>
      </div>
      <div style="display:flex;gap:8px;flex-shrink:0">
        <button onclick="openElCreateModal()" style="background:var(--accent,#8b6a42);border:none;border-radius:6px;padding:7px 16px;cursor:pointer;font-size:13px;color:#fff;font-weight:500">＋ 新建事件</button>
        <button onclick="loadEventLine()" style="background:none;border:1px solid var(--border,#ddd);border-radius:6px;padding:7px 14px;cursor:pointer;font-size:13px;color:var(--text-light)">↻ 刷新</button>
      </div>
    </div>
    <div class="reflection-board" id="event-line-board">
      <div class="loading" style="padding:16px;font-size:13px;color:var(--text-light)">加载事件线…</div>
    </div>
  </div>
</div>

<!-- 事件线编辑弹窗 -->
<div id="el-modal"><div id="el-modal-box">
  <h3 id="el-modal-title">编辑记录</h3>
  <div class="el-modal-row" id="el-row-title" style="display:none">
    <div class="el-modal-label">事件名称</div>
    <input class="el-modal-input" id="el-f-title" type="text" maxlength="60">
  </div>
  <div class="el-modal-row" id="el-row-date" style="display:none">
    <div class="el-modal-label">日期 (YYYY-MM-DD)</div>
    <input class="el-modal-input" id="el-f-date" type="text">
  </div>
  <div class="el-modal-row">
    <div class="el-modal-label">今日进度 (%)</div>
    <input class="el-modal-input" id="el-f-progress" type="number" min="0" max="100" placeholder="可留空">
  </div>
  <div class="el-modal-row">
    <div class="el-modal-label">记录内容</div>
    <textarea class="el-modal-input" id="el-f-text" rows="4" maxlength="200"></textarea>
  </div>
  <div class="tl-modal-actions">
    <button class="tl-modal-cancel" onclick="closeElModal()">取消</button>
    <button class="tl-modal-save" onclick="saveElModal()">保存</button>
  </div>
</div></div>

'''
if VIEW_ANCHOR not in src:
    print("ERROR: profile-view anchor not found")
    sys.exit(1)
src = src.replace(VIEW_ANCHOR, VIEW_INSERT + VIEW_ANCHOR, 1)
print("OK: view")

# ── 4. Tab switch JS ───────────────────────────────────────────────────────
SW_OLD = "    if (document.getElementById('persona-edit-view')) document.getElementById('persona-edit-view').style.display = target === 'persona-edit' ? '' : 'none';"
SW_NEW = SW_OLD + "\n    if (document.getElementById('event-line-view')) document.getElementById('event-line-view').style.display = target === 'event-line' ? '' : 'none';"
if SW_OLD not in src:
    print("ERROR: tab switch anchor not found")
    sys.exit(1)
src = src.replace(SW_OLD, SW_NEW, 1)

LOAD_OLD = "    if (target === 'persona-edit') loadPersonaEdit();"
LOAD_NEW = LOAD_OLD + "\n    if (target === 'event-line') loadEventLine();"
if LOAD_OLD not in src:
    print("ERROR: load hook anchor not found")
    sys.exit(1)
src = src.replace(LOAD_OLD, LOAD_NEW, 1)
print("OK: tab switch")

# ── 5. JS functions ────────────────────────────────────────────────────────
JS_ANCHOR = "// ── 人设 Tab ──────────────────────────────────────────────────────────────"
JS_BLOCK = r'''// ── 事件线 Tab ────────────────────────────────────────────────────────────
var _elEvents = [];
var _elStaleIds = [];
var _elCollapsed = {};
var _elModalMode = '';
var _elModalEventId = '';
var _elModalEntryIndex = -1;

function elEsc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function elFmtDate(dateStr) {
  if (!dateStr || dateStr.length < 10) return dateStr || '—';
  var p = dateStr.split('-');
  return (p[1] ? parseInt(p[1],10) : '') + '/' + (p[2] ? parseInt(p[2],10) : '');
}

async function loadEventLine() {
  var board = document.getElementById('event-line-board');
  if (!board) return;
  board.innerHTML = '<div class="loading" style="padding:16px;font-size:13px;color:var(--text-light)">加载事件线…</div>';
  try {
    var res = await authFetch(BASE + '/api/event-line');
    if (!res || !res.ok) throw new Error('HTTP ' + (res ? res.status : 'fail'));
    var data = await res.json();
    _elEvents = data.events || [];
    _elStaleIds = data.stale_ids || [];
    renderEventLine();
  } catch (e) {
    board.innerHTML = '<div style="color:#ef4444;font-size:13px;padding:16px">加载失败: ' + elEsc(e.message) + '</div>';
  }
}

function renderEventLine() {
  var board = document.getElementById('event-line-board');
  if (!board) return;
  if (!_elEvents.length) {
    board.innerHTML = '<div style="font-size:13px;color:var(--text-light);padding:16px">暂无进行中的事件。点击右上角「新建事件」开始记录。</div>';
    return;
  }
  var html = _elEvents.map(function(ev) {
    var id = ev.id || '';
    var title = ev.title || '未命名事件';
    var collapsed = !!_elCollapsed[id];
    var stale = _elStaleIds.indexOf(id) >= 0;
    var entries = ev.entries || [];
    var cards = entries.map(function(entry, idx) {
      var date = entry.date || '';
      var progress = entry.progress;
      var progText = (progress === null || progress === undefined || progress === '') ? '—' : (progress + '%');
      return '<div class="el-date-card">'
        + '<div class="el-date-label">' + elEsc(elFmtDate(date)) + '</div>'
        + '<div class="el-date-body">'
        + '<div class="el-progress-row"><span class="el-progress-label">今日进度</span><span class="el-progress-val">' + elEsc(progText) + '</span></div>'
        + '<div class="el-entry-text">' + elEsc(entry.text || '') + '</div>'
        + '</div>'
        + '<div class="el-card-actions">'
        + '<button title="编辑" onclick="openElEditModal(\'' + id + '\',' + idx + ')">✎</button>'
        + '<button class="danger" title="删除" onclick="deleteElEntry(\'' + id + '\',' + idx + ')">×</button>'
        + '</div></div>';
    }).join('');
    if (!cards) cards = '<div style="font-size:13px;color:var(--text-light);padding:8px 4px">暂无每日记录。</div>';
    return '<section class="el-event-block" data-event-id="' + elEsc(id) + '">'
      + '<div class="el-event-head" onclick="toggleElEvent(\'' + id + '\', event)">'
      + '<span class="el-event-chevron">' + (collapsed ? '▶' : '▼') + '</span>'
      + '<div class="el-event-title">' + elEsc(title) + ' · 每日记录'
      + (stale ? '<span class="el-stale-badge">超1天未更新</span>' : '') + '</div>'
      + '<div class="el-event-meta">更新 ' + elEsc(ev.last_updated || ev.created || '') + '</div>'
      + '<div class="el-event-actions" onclick="event.stopPropagation()">'
      + '<button onclick="openElTitleModal(\'' + id + '\')">改标题</button>'
      + '<button class="danger" onclick="deleteElEvent(\'' + id + '\')">删事件</button>'
      + '</div></div>'
      + '<div class="el-event-body' + (collapsed ? ' collapsed' : '') + '" id="el-body-' + id + '">' + cards + '</div>'
      + '</section>';
  }).join('');
  board.innerHTML = html;
}

function toggleElEvent(id, evt) {
  if (evt && evt.target && evt.target.closest && evt.target.closest('.el-event-actions')) return;
  _elCollapsed[id] = !_elCollapsed[id];
  var body = document.getElementById('el-body-' + id);
  var chev = document.querySelector('[data-event-id="' + id + '"] .el-event-chevron');
  if (body) body.classList.toggle('collapsed', !!_elCollapsed[id]);
  if (chev) chev.textContent = _elCollapsed[id] ? '▶' : '▼';
}

function openElCreateModal() {
  _elModalMode = 'create';
  _elModalEventId = '';
  _elModalEntryIndex = -1;
  document.getElementById('el-modal-title').textContent = '新建事件';
  document.getElementById('el-row-title').style.display = '';
  document.getElementById('el-row-date').style.display = 'none';
  document.getElementById('el-f-title').value = '';
  document.getElementById('el-f-date').value = '';
  document.getElementById('el-f-progress').value = '';
  document.getElementById('el-f-text').value = '';
  document.getElementById('el-modal').classList.add('open');
}

function openElTitleModal(eventId) {
  var ev = _elEvents.find(function(e){ return e.id === eventId; });
  if (!ev) return;
  _elModalMode = 'title';
  _elModalEventId = eventId;
  _elModalEntryIndex = -1;
  document.getElementById('el-modal-title').textContent = '修改事件标题';
  document.getElementById('el-row-title').style.display = '';
  document.getElementById('el-row-date').style.display = 'none';
  document.getElementById('el-f-title').value = ev.title || '';
  document.getElementById('el-f-progress').value = '';
  document.getElementById('el-f-text').value = '';
  document.getElementById('el-modal').classList.add('open');
}

function openElEditModal(eventId, entryIndex) {
  var ev = _elEvents.find(function(e){ return e.id === eventId; });
  if (!ev || !ev.entries || !ev.entries[entryIndex]) return;
  var entry = ev.entries[entryIndex];
  _elModalMode = 'entry';
  _elModalEventId = eventId;
  _elModalEntryIndex = entryIndex;
  document.getElementById('el-modal-title').textContent = (ev.title || '事件') + ' · ' + (entry.date || '') + ' 编辑';
  document.getElementById('el-row-title').style.display = 'none';
  document.getElementById('el-row-date').style.display = '';
  document.getElementById('el-f-date').value = entry.date || '';
  document.getElementById('el-f-progress').value = (entry.progress === null || entry.progress === undefined) ? '' : entry.progress;
  document.getElementById('el-f-text').value = entry.text || '';
  document.getElementById('el-modal').classList.add('open');
}

function closeElModal() {
  document.getElementById('el-modal').classList.remove('open');
}

async function saveElModal() {
  try {
    if (_elModalMode === 'create') {
      var title = document.getElementById('el-f-title').value.trim();
      var text = document.getElementById('el-f-text').value.trim();
      var progRaw = document.getElementById('el-f-progress').value;
      if (!title || !text) { alert('请填写事件名称和首条记录'); return; }
      var body = { title: title, text: text };
      if (progRaw !== '') body.progress = parseInt(progRaw, 10);
      var res = await authFetch(BASE + '/api/event-line', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
      if (!res || !res.ok) throw new Error(await res.text());
    } else if (_elModalMode === 'title') {
      var newTitle = document.getElementById('el-f-title').value.trim();
      if (!newTitle) { alert('标题不能为空'); return; }
      var res2 = await authFetch(BASE + '/api/event-line/' + encodeURIComponent(_elModalEventId), { method: 'PATCH', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ title: newTitle }) });
      if (!res2 || !res2.ok) throw new Error(await res2.text());
    } else if (_elModalMode === 'entry') {
      var payload = {
        text: document.getElementById('el-f-text').value.trim(),
        date: document.getElementById('el-f-date').value.trim(),
      };
      var pRaw = document.getElementById('el-f-progress').value;
      payload.progress = pRaw === '' ? null : parseInt(pRaw, 10);
      var res3 = await authFetch(BASE + '/api/event-line/' + encodeURIComponent(_elModalEventId) + '/entries/' + _elModalEntryIndex, { method: 'PATCH', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (!res3 || !res3.ok) throw new Error(await res3.text());
    }
    closeElModal();
    await loadEventLine();
  } catch (e) {
    alert('保存失败: ' + e.message);
  }
}

async function deleteElEntry(eventId, entryIndex) {
  if (!confirm('确定删除这条每日记录？')) return;
  try {
    var res = await authFetch(BASE + '/api/event-line/' + encodeURIComponent(eventId) + '/entries/' + entryIndex, { method: 'DELETE' });
    if (!res || !res.ok) throw new Error(await res.text());
    await loadEventLine();
  } catch (e) {
    alert('删除失败: ' + e.message);
  }
}

async function deleteElEvent(eventId) {
  if (!confirm('确定删除整个事件及其全部每日记录？')) return;
  try {
    var res = await authFetch(BASE + '/api/event-line/' + encodeURIComponent(eventId), { method: 'DELETE' });
    if (!res || !res.ok) throw new Error(await res.text());
    await loadEventLine();
  } catch (e) {
    alert('删除失败: ' + e.message);
  }
}

'''
if JS_ANCHOR not in src:
    print("ERROR: JS anchor not found")
    sys.exit(1)
src = src.replace(JS_ANCHOR, JS_BLOCK + JS_ANCHOR, 1)
print("OK: JS")

DASH.write_text(src, encoding="utf-8")
print("DONE: dashboard event-line tab patched")
