from pathlib import Path


def test_dashboard_comments_show_author_and_time_without_emotion_fields():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "var commentTime = c.original_feel_created || c.created || '';" in html
    assert "let dashboardAiAuthor = 'Haven';" in html
    assert "function commentAuthorName(comment)" in html
    assert '<div class="comment-meta">' in html
    assert '<span class="comment-author">' in html
    assert '<span class="comment-time">' in html

    comments_block = html.split("var commentsHtml = comments.length", 1)[1].split("var commentFormHtml", 1)[0]
    assert "commentAuthorName(c)" in comments_block
    assert "c.valence" not in comments_block
    assert "c.arousal" not in comments_block


def test_dashboard_comment_enter_submit_has_no_visible_send_key():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    form_block = html.split("var commentFormHtml =", 1)[1].split("content.innerHTML =", 1)[0]

    assert "handleCommentKeydown(event)" in form_block
    assert "comment-send-button" not in html
    assert 'aria-label="发送"' not in form_block


def test_dashboard_bucket_detail_loads_moment_diagnostics():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    detail_block = html.split("async function showDetail", 1)[1].split("function startBucketContentEdit", 1)[0]

    assert 'id="bucket-moments-block"' in detail_block
    assert "loadBucketMoments(id);" in detail_block
    assert "BASE + '/api/moments?bucket_id='" in html
    assert "function renderBucketMoments(data)" in html
    assert "function renderMomentItem(moment, index)" in html
    assert "function renderMomentEdges(edges)" in html
    assert "function renderMomentEdge(edge)" in html
    assert "Moment Edges" in html
    assert "moment-source-window" in html
    assert "原文窗口" in html
    assert "m.source_window" in html
    assert "runtime_gate" in html
    assert ".detail-moments" in html
    assert ".moment-edge-list" in html


def test_dashboard_bucket_list_has_bulk_delete_controls():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    list_view = html.split('id="list-view"', 1)[1].split('id="breath-view"', 1)[0]

    assert 'id="bucket-bulk-toolbar"' in list_view
    assert "toggleBucketBulkMode()" in list_view
    assert "selectCurrentBuckets()" in list_view
    assert "deleteSelectedBuckets()" in list_view
    assert "BASE + '/api/buckets/delete'" in html
    assert "confirm: 'DELETE'" in html
    assert "bucketBulkDeleteBlockReason" in html
    assert "受保护记忆不能批量删除" in html


def test_dashboard_breath_debug_loads_diffusion_paths():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    breath_block = html.split("async function runBreathDebug()", 1)[1].split("function breathGateTrace", 1)[0]

    assert 'id="diffusion-results"' in html
    assert "loadDiffusionDebug(query);" in breath_block
    assert "BASE + '/api/diffusion-debug?q='" in html
    assert "function renderDiffusionDebug(data)" in html
    assert "function renderDiffusionRow(item, index, kind)" in html
    assert ".diffusion-panel" in html


def test_dashboard_breath_debug_loads_recall_moment_candidates():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    breath_block = html.split("async function runBreathDebug()", 1)[1].split("async function loadDiffusionDebug", 1)[0]

    assert 'id="recall-results"' in html
    assert "loadRecallDebug(query);" in breath_block
    assert "BASE + '/api/recall-debug?q='" in html
    assert "function renderRecallDebug(data)" in html
    assert "function renderRecallCandidate(candidate)" in html
    assert "selected_secondary" in html
    assert "direct_render.shape" in html
    assert "render ' + c.direct_render.shape" in html
    assert "Moment 命中" in html


def test_dashboard_breath_view_loads_gateway_injection_debug():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert 'id="gateway-injections-panel"' in html
    assert 'id="gateway-session-filter"' in html
    assert "loadGatewayInjections()" in html
    assert "BASE + '/api/gateway-injections?limit=10" in html
    assert "function renderGatewayInjections(data)" in html
    assert "function renderGatewayInjectionItem(item)" in html
    assert "function gatewayDirectRenderSummaries(payload)" in html
    assert "function gatewayChainDebugSummaries(payload)" in html
    assert "direct render:" in html
    assert "diffused chain:" in html
    assert 'class="gateway-injections-content"' in html
    assert 'class="gateway-injection-list"' in html
    assert 'class="gateway-injection-item"' in html
    assert '.gateway-injection-controls input[type="text"]' in html
    assert '.gateway-injection-controls input[type="checkbox"]' in html
    assert "gateway-injection-empty" in html
    assert "row.direct_render" in html
    assert "diffused_moment_debug" in html
    assert "Gateway 最近注入" in html


def test_dashboard_exposes_profile_fact_page():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert 'data-tab="profile"' in html
    assert 'id="profile-view"' in html
    assert 'id="profile-facts-list"' in html
    assert 'id="profile-summary"' in html
    assert "loadProfileFacts()" in html
    assert "renderProfileFactCard" in html
    assert "runProfileFactAction" in html
    assert "editProfileFact" in html
    assert "BASE + '/api/profile-facts'" in html
    assert "BASE + '/api/profile-facts/'" in html
    assert "打开证据" in html
    assert "Profile Facts" in html
    assert ".profile-card" in html


def test_dashboard_exposes_gateway_memory_cooldown_settings():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]
    config_view = html.split('id="config-view"', 1)[1].split('id="detail-panel"', 1)[0]

    assert 'data-tab="memory-config"' in html
    assert html.index('data-tab="config"') < html.index('data-tab="memory-config"')
    assert html.index('data-tab="memory-config"') < html.index('data-tab="import"')
    assert 'id="memory-config-view"' in html
    assert 'id="memory-config-status"' in html
    assert "<h3>记忆浮现</h3>" in html
    assert 'id="cfg-recent-context-enabled"' in html
    assert 'id="cfg-persona-context-enabled"' in html
    assert 'id="cfg-persona-context-rounds"' in html
    assert 'id="cfg-gateway-cooldown"' in html
    assert 'id="cfg-gateway-rounds"' in html
    assert 'id="cfg-direct-render-mode"' in html
    assert 'id="cfg-retrieval-mode"' in html
    assert 'id="cfg-diffusion-enabled"' in html
    assert 'id="cfg-diffusion-topk"' in html
    assert 'id="cfg-diffusion-min"' in html
    assert 'id="cfg-chain-walk"' in html
    assert 'id="cfg-chain-hops"' in html
    assert 'id="cfg-chain-confidence"' in html
    assert 'id="cfg-chain-frontier"' in html
    assert "breath 和 Gateway 都会立即读取新的扩散参数" in html
    assert "Gateway 需要重启" not in html
    assert "cfg.gateway.cooldown_hours" in html
    assert "cfg.gateway.skip_recent_rounds" in html
    assert "cfg.gateway.current_inner_state_interval_rounds" in html
    assert "cfg.gateway.direct_render_mode" in html
    assert "cfg.gateway.retrieval_mode" in html
    assert "((cfg.gateway && cfg.gateway.recent_context_budget) ?? 300) > 0 ? 'true' : 'false'" in load_block
    assert "personaRounds > 0 ? 'true' : 'false'" in load_block
    assert "setConfigStatus" in html
    assert 'id="cfg-gateway-cooldown"' not in config_view
    assert "cfg.memory_diffusion || {}" in load_block
    assert "diffusion.chain_walk_enabled" in load_block
    assert "diffusion.chain_min_confidence" in load_block
    assert "cooldown_hours: floatValue('cfg-gateway-cooldown', 6)" in html
    assert "skip_recent_rounds: numberValue('cfg-gateway-rounds', 5)" in html
    assert "current_inner_state_interval_rounds: personaContextRounds," in html
    assert "direct_render_mode: document.getElementById('cfg-direct-render-mode').value," in html
    assert "retrieval_mode: document.getElementById('cfg-retrieval-mode').value," in html
    assert "memory_diffusion: {" in save_block
    assert "top_k: numberValue('cfg-diffusion-topk', 4)," in save_block
    assert "min_activation: floatValue('cfg-diffusion-min', 0.18)," in save_block
    assert "chain_walk_enabled: document.getElementById('cfg-chain-walk').value === 'true'," in save_block
    assert "chain_min_confidence: floatValue('cfg-chain-confidence', 0.72)," in save_block


def test_dashboard_exposes_reflection_affect_anchor_switches():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "<h3>记忆关系整理</h3>" in html
    assert 'id="cfg-reflection-enabled"' in html
    assert 'id="cfg-reflection-auto"' in html
    assert 'id="cfg-reflection-memory-anchor"' in html
    assert 'id="cfg-reflection-weather-anchor"' in html
    assert 'id="cfg-reflection-model"' in html
    assert 'id="cfg-reflection-url"' in html
    assert 'id="cfg-reflection-key"' in html
    assert "cfg.reflection.enabled" in load_block
    assert "cfg.reflection.auto_enabled" in load_block
    assert "cfg.reflection.memory_affect_anchor_enabled" in load_block
    assert "cfg.reflection.relationship_weather_affect_anchor_enabled" in load_block
    assert "enabled: document.getElementById('cfg-reflection-enabled').value === 'true'," in save_block
    assert "auto_enabled: document.getElementById('cfg-reflection-auto').value === 'true'," in save_block
    assert "memory_affect_anchor_enabled: document.getElementById('cfg-reflection-memory-anchor').value === 'true'," in save_block
    assert "relationship_weather_affect_anchor_enabled: document.getElementById('cfg-reflection-weather-anchor').value === 'true'," in save_block
    assert "if (reflectionKeyVal) body.reflection.api_key = reflectionKeyVal;" in html


def test_dashboard_exposes_persona_config_and_env_persist_button():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "<h3>Persona State</h3>" in html
    assert 'id="cfg-persona-enabled"' in html
    assert 'id="cfg-persona-model"' in html
    assert 'id="cfg-persona-url"' in html
    assert 'id="cfg-persona-key"' in html
    assert "saveConfig(true, true)" in html
    assert "保存密钥到 .env" in html
    assert "cfg.persona.enabled" in load_block
    assert "cfg.persona.api_key_masked" in load_block
    assert "enabled: document.getElementById('cfg-persona-enabled').value === 'true'," in save_block
    assert "model: document.getElementById('cfg-persona-model').value," in save_block
    assert "base_url: document.getElementById('cfg-persona-url').value," in save_block
    assert "persist_env: !!persistEnv" in save_block
    assert "if (personaKeyVal) body.persona.api_key = personaKeyVal;" in html


def test_dashboard_dream_background_control_uses_auto_enabled_only():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]
    dream_block = save_block.split("dream: {", 1)[1].split("gateway: {", 1)[0]
    dream_lines = [line.strip() for line in dream_block.splitlines()]

    assert 'id="cfg-dream-engine-enabled"' in html
    assert 'id="cfg-dream-inject"' in html
    assert 'id="cfg-dream-retain"' in html
    assert "document.getElementById('cfg-dream-engine-enabled').value = cfg.dream.enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-enabled').value = cfg.dream.auto_enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-inject').value = cfg.dream.inject_enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-retain').value = cfg.dream.retain_after_inject ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-enabled').value = cfg.dream.enabled" not in load_block
    assert "enabled: document.getElementById('cfg-dream-engine-enabled').value === 'true'," in dream_lines
    assert "auto_enabled: document.getElementById('cfg-dream-enabled').value === 'true'," in dream_lines
    assert "enabled: document.getElementById('cfg-dream-enabled').value === 'true'," not in dream_lines
    assert "surface_enabled: document.getElementById('cfg-dream-surface').value === 'true'," in dream_lines
    assert "inject_enabled: document.getElementById('cfg-dream-inject').value === 'true'," in dream_lines
    assert "retain_after_inject: document.getElementById('cfg-dream-retain').value === 'true'," in dream_lines


def test_dashboard_config_number_zero_values_are_preserved():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "document.getElementById('cfg-dehy-temp').value = cfg.dehydration.temperature ?? 0.1;" in load_block
    assert "document.getElementById('cfg-merge').value = cfg.merge_threshold ?? 75;" in load_block
    assert "temperature: floatValue('cfg-dehy-temp', 0.1)," in save_block
    assert "merge_threshold: numberValue('cfg-merge', 75)," in save_block
    assert "cfg.dehydration.temperature || 0.1" not in load_block
    assert "cfg.merge_threshold || 75" not in load_block
    assert "parseFloat(document.getElementById('cfg-dehy-temp').value) || 0.1" not in save_block
    assert "parseInt(document.getElementById('cfg-merge').value) || 75" not in save_block


def test_dashboard_import_file_input_resets_after_selection():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    import_block = html.split("// --- Import functions ---", 1)[1].split("async function pollImportStatus", 1)[0]

    assert "const selectedFile = fileInput.files[0];" in import_block
    assert "fileInput.value = '';" in import_block
    assert "startImport(selectedFile);" in import_block
