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


def test_dashboard_bucket_detail_can_edit_event_date_separately():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    detail_block = html.split("async function showDetail", 1)[1].split("function loadBucketMoments", 1)[0]
    date_submit = html.split("async function submitBucketDateEdit", 1)[1].split("async function submitBucketTitleEdit", 1)[0]

    assert 'id="bucket-date-view"' in detail_block
    assert 'id="bucket-date-edit"' in detail_block
    assert 'name="date" type="date"' in detail_block
    assert "startBucketDateEdit()" in detail_block
    assert "function dateInputValue(value)" in html
    assert "JSON.stringify({date: newDate})" in date_submit
    assert "await loadBuckets();" in date_submit
    assert "JSON.stringify({name:" not in date_submit
    assert "JSON.stringify({content:" not in date_submit


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


def test_dashboard_feel_filter_excludes_daily_impressions():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "{ key: 'daily_impression', label: '日印象' }" in html
    assert "currentFilter === 'feel'" in html
    assert "b.type === 'feel' && !isDailyImpressionBucket(b)" in html
    assert "currentFilter === 'daily_impression'" in html
    assert "buckets.filter(isDailyImpressionBucket)" in html


def test_dashboard_exposes_darkroom_door_without_release_or_body_fields():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    door_block = html.split('id="darkroom-door"', 1)[1].split('<div class="search-bar">', 1)[0]
    render_block = html.split("function renderDarkroomDoor", 1)[1].split("async function loadDarkroomDoor", 1)[0]
    load_block = html.split("async function loadDarkroomDoor", 1)[1].split("function monthStart", 1)[0]

    assert "Darkroom Door" in door_block
    assert "BASE + '/api/darkroom/status'" in load_block
    assert "loadDarkroomDoor();" in html
    assert "previous_completeness" in render_block
    assert "last_completeness" in render_block
    assert "darkroom_release" not in html
    assert ".content" not in render_block
    assert ".note" not in render_block


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
    assert "date trace yes" in html
    assert "date trace no" in html
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
    assert "generateProfileFactProposals" in html
    assert "confirmProfileFactProposal" in html
    assert "generateAnchorProposals" in html
    assert "confirmAnchorProposal" in html
    assert "BASE + '/api/profile-facts'" in html
    assert "BASE + '/api/profile-facts/'" in html
    assert "BASE + '/api/profile-fact-proposals'" in html
    assert "BASE + '/api/profile-fact-proposals/confirm'" in html
    assert "BASE + '/api/anchor-proposals'" in html
    assert "BASE + '/api/anchor-proposals/confirm'" in html
    assert 'id="profile-proposal-bucket-id"' in html
    assert 'id="profile-proposal-list"' in html
    assert 'id="anchor-proposal-bucket-id"' in html
    assert 'id="anchor-proposal-list"' in html
    assert "打开证据" in html
    assert "生成画像候选" in html
    assert "生成 Anchor 候选" in html
    assert "确认写入" in html
    assert "确认标为 Anchor" in html
    assert "Profile Facts" in html
    assert ".profile-card" in html
    assert ".profile-proposal-panel" in html


def test_dashboard_exposes_portrait_state_panel():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    profile_tab_block = html.split("if (target === 'profile')", 1)[1].split("if (target === 'word-map')", 1)[0]
    load_buckets_block = html.split("async function loadBuckets", 1)[1].split("function getActiveTab()", 1)[0]

    assert 'id="portrait-state-panel"' in html
    assert 'id="portrait-state-summary"' in html
    assert 'id="portrait-state-content"' in html
    assert 'id="portrait-maintain-message"' in html
    assert "Portrait State" in html
    assert "只读，不写 profile_fact、anchor 或 Core Memory" in html
    assert "loadPortraitState()" in html
    assert "refreshPortraitView()" in html
    assert "runPortraitMaintain" in html
    assert "resetPortraitState" in html
    assert "Recent Timeline" in html
    assert "renderPortraitSelfAnchor" in html
    assert "自我总入口" in html
    assert "loadBuckets({ skipProfileRefresh: true });" in html
    assert "var deleteSpec = item._delete ? escAttr(jsString(JSON.stringify(item._delete))) : '';" in html
    assert "renderPortraitState" in html
    assert "renderPortraitScope" in html
    assert "renderPortraitCandidates" in html
    assert "renderPortraitEvidence" in html
    assert "BASE + '/api/portrait-state'" in html
    assert "BASE + '/api/portrait-maintain'" in html
    assert "BASE + '/api/portrait-state/reset'" in html
    assert "body: JSON.stringify({ force: true })" in html
    assert "body: JSON.stringify({ confirm: 'RESET' })" in html
    assert "read only" in html
    assert "手动生成" in html
    assert "清空画像" in html
    assert "state.recent_timeline" in html
    assert "state.stable_candidates" in html
    assert "state.profile_fact_candidates" in html
    assert "state.self_anchor_entry" in html
    assert "renderPortraitSelfAnchor(state.self_anchor_entry)" in html
    assert "cfg-self-anchor-entry" in html
    assert "renderPortraitScope('persona'" not in html
    assert ".portrait-state-grid" in html
    assert ".portrait-candidate-grid" in html
    assert "loadPortraitState();" in profile_tab_block
    assert "loadProfileFacts();" in profile_tab_block
    assert "loadPortraitState();" in load_buckets_block


def test_dashboard_keeps_compact_legacy_filter_row_and_compatible_filters():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    build_block = html.split("function buildFilters()", 1)[1].split("function filterBuckets", 1)[0]
    filter_block = html.split("function filterBuckets", 1)[1].split("function bucketBulkDeleteBlockReason", 1)[0]

    assert "function isSelfAnchorBucket" in html
    assert "tag:self_anchor" in build_block
    assert "label: '自我'" in build_block
    assert "label: '全部'" in build_block
    assert "label: '📌 钉选'" in build_block
    assert "label: '⚓ Anchor'" in build_block
    assert "label: 'Feel'" in build_block
    assert "label: '日印象'" in build_block
    assert "label: '⚡ 未解决'" in build_block
    assert "label: '🌿 已消化'" in build_block
    assert "label: '📦 归档'" in build_block
    assert "旧标签 / legacy domain" not in build_block
    assert "canonicalDomains" not in build_block
    assert "kindFilters" not in build_block
    assert "statusFilters" not in build_block
    assert "flagFilters" not in build_block
    assert "key: 'profile'" not in build_block
    assert "label: '画像'" not in build_block
    assert "filters.onclick = function(e)" in build_block
    assert "filters.addEventListener" not in build_block
    assert "currentFilter === 'profile'" not in filter_block
    assert "currentFilter === 'tag:self_anchor'" in filter_block
    assert "currentFilter.startsWith('canonical_domain:')" in filter_block
    assert "currentFilter.startsWith('legacy_domain:')" in filter_block
    assert "&& !isDailyImpressionBucket(b)" in filter_block
    assert "&& !isSelfAnchorBucket(b)" in filter_block


def test_dashboard_hides_confirm_button_for_active_profile_facts():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "var confirmButton = status.cls === 'active'" in html
    assert "? ''" in html
    assert "confirmButton +\n        '<button type=\"button\" onclick=\"editProfileFact" in html
    assert "runProfileFactAction(\\'' + jsString(id) + '\\', \\'confirm\\')" in html


def test_dashboard_keeps_proposal_confirm_success_messages():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    profile_block = html.split("async function confirmProfileFactProposal", 1)[1].split("function setAnchorProposalMessage", 1)[0]
    anchor_block = html.split("async function confirmAnchorProposal", 1)[1].split("function renderProfileFacts", 1)[0]

    assert "var successMessage = '已写入画像事实 ' + (data.id || '');" in profile_block
    assert profile_block.index("renderProfileFactProposals({ proposals: profileFactProposals, rejected: [] });") < profile_block.index("setProfileProposalMessage(successMessage, 'ok');")
    assert "var successMessage = data.status === 'already_anchor' ? '已经是 Anchor。' : '已标为 Anchor ' + (data.id || '');" in anchor_block
    assert anchor_block.index("renderAnchorProposals({ proposals: anchorProposals, rejected: [] });") < anchor_block.index("setAnchorProposalMessage(successMessage, 'ok');")


def test_dashboard_exposes_word_map_page():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert 'data-tab="word-map"' in html
    assert 'id="word-map-view"' in html
    assert 'id="word-map-summary"' in html
    assert 'id="word-map-nodes"' in html
    assert 'id="word-map-edges"' in html
    assert 'id="identity-aliases"' in html
    assert 'id="word-map-boundary"' in html
    assert "loadWordMap()" in html
    assert "rebuildWordMap()" in html
    assert "rebuildIdentitySemantics()" in html
    assert "BASE + '/api/word-map?nodes=20&edges=20'" in html
    assert "BASE + '/api/word-map/rebuild'" in html
    assert "BASE + '/api/word-map/cards" not in html
    assert "BASE + '/api/identity-semantics?limit=50'" in html
    assert "BASE + '/api/identity-semantics/rebuild'" in html
    assert ".word-map-card" in html
    assert "不会自动注入 Gateway" in html
    assert "未配置私有 canonical" in html


def test_dashboard_exposes_gateway_memory_cooldown_settings():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]
    config_view = html.split('id="config-view"', 1)[1].split('id="detail-panel"', 1)[0]

    assert 'data-tab="memory-config"' in html
    assert html.index('data-tab="config"') < html.index('data-tab="memory-config"')
    assert html.index('data-tab="config"') < html.index('data-tab="upstream-config"')
    assert html.index('data-tab="upstream-config"') < html.index('data-tab="memory-config"')
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
    assert 'id="cfg-word-map-hint-enabled"' in html
    assert 'id="cfg-query-planner-enabled"' in html
    assert 'id="cfg-query-planner-model"' not in html
    assert 'id="cfg-memory-detail-recall-enabled"' in html
    assert 'id="cfg-memory-detail-recall-max-ids"' in html
    assert 'id="cfg-memory-detail-recall-budget"' in html
    assert "Targeted Memory Detail 不依赖这个开关" in html
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
    assert "cfg.gateway.word_map_hint_enabled" in html
    assert "cfg.gateway.query_planner_enabled" in html
    assert "cfg.gateway.query_planner_model" not in html
    assert "cfg.gateway.memory_detail_recall_enabled" in html
    assert "cfg.gateway.memory_detail_recall_max_ids" in html
    assert "cfg.gateway.memory_detail_recall_budget" in html
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
    assert "word_map_hint_enabled: document.getElementById('cfg-word-map-hint-enabled').value === 'true'," in html
    assert "query_planner_enabled: document.getElementById('cfg-query-planner-enabled').value === 'true'," in html
    assert "query_planner_model: document.getElementById('cfg-query-planner-model').value," not in html
    assert "memory_detail_recall_enabled: document.getElementById('cfg-memory-detail-recall-enabled').value === 'true'," in html
    assert "memory_detail_recall_max_ids: numberValue('cfg-memory-detail-recall-max-ids', 3)," in html
    assert "memory_detail_recall_budget: numberValue('cfg-memory-detail-recall-budget', 1200)," in html
    assert "candidate.memory_diffusion = {" in save_block
    assert "top_k: numberValue('cfg-diffusion-topk', 4)," in save_block
    assert "min_activation: floatValue('cfg-diffusion-min', 0.18)," in save_block
    assert "chain_walk_enabled: document.getElementById('cfg-chain-walk').value === 'true'," in save_block
    assert "chain_min_confidence: floatValue('cfg-chain-confidence', 0.72)," in save_block


def test_dashboard_config_save_sends_changed_sections_and_forces_memory_panel():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("try {", 1)[0]

    assert "let configSnapshot = null;" in html
    assert "configSnapshot = JSON.parse(JSON.stringify(cfg || {}));" in load_block
    assert "function addChangedSection(target, sectionName, candidate, force)" in save_block
    assert "if (force) {" in save_block
    assert "var activeTarget = activeTab ? activeTab.dataset.tab : '';" in save_block
    assert "var body = {" in save_block
    assert "forceSections.gateway = true;" in save_block
    assert "forceSections.recall = true;" in save_block
    assert "forceSections.memory_diffusion = true;" in save_block
    assert "addChangedSection(body, sectionName, candidate[sectionName], !!forceSections[sectionName]);" in save_block
    assert "if (!body.persona) body.persona = {};" in save_block
    assert "body.persona.api_key = personaKeyVal;" in save_block


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
    reflection_block = save_block.split("candidate.reflection = {", 1)[1].split("};", 1)[0]
    assert "base_url: document.getElementById('cfg-reflection-url').value," in reflection_block
    assert "enabled: document.getElementById('cfg-reflection-enabled').value === 'true'," in reflection_block
    assert "model: document.getElementById('cfg-reflection-model').value," in reflection_block
    assert "if (!body.reflection) body.reflection = {};" in html
    assert "body.reflection.api_key = reflectionKeyVal;" in html


def test_dashboard_exposes_portrait_maintainer_controls():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "<h3>每日画像 Portrait</h3>" in html
    assert 'id="cfg-portrait-enabled"' in html
    assert 'id="cfg-portrait-auto"' in html
    assert 'id="cfg-portrait-auto-initial"' in html
    assert 'id="cfg-portrait-daily"' in html
    assert 'id="cfg-portrait-material-limit"' in html
    assert 'id="cfg-portrait-first-limit"' in html
    assert "cfg.portrait.enabled" in load_block
    assert "cfg.portrait.auto_enabled" in load_block
    assert "cfg.portrait.auto_initial_enabled" in load_block
    assert "cfg.portrait.daily_enabled" in load_block
    assert "cfg.portrait.first_run_material_limit" in load_block
    assert "Portrait API" in load_block
    portrait_block = save_block.split("candidate.portrait = {", 1)[1].split("};", 1)[0]
    assert "enabled: document.getElementById('cfg-portrait-enabled').value === 'true'," in portrait_block
    assert "auto_enabled: document.getElementById('cfg-portrait-auto').value === 'true'," in portrait_block
    assert "auto_initial_enabled: document.getElementById('cfg-portrait-auto-initial').value === 'true'," in portrait_block
    assert "daily_enabled: document.getElementById('cfg-portrait-daily').value === 'true'," in portrait_block
    assert "material_limit: numberValue('cfg-portrait-material-limit', 18)," in portrait_block
    assert "first_run_material_limit: numberValue('cfg-portrait-first-limit', 160)," in portrait_block
    assert "'portrait'," in save_block


def test_dashboard_exposes_persona_config_and_env_persist_button():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "<h3>Persona State</h3>" in html
    assert 'id="cfg-persona-enabled"' in html
    assert 'id="cfg-persona-event-recording"' in html
    assert 'id="cfg-persona-model"' in html
    assert 'id="cfg-persona-url"' in html
    assert 'id="cfg-persona-key"' in html
    assert "saveConfig(true, true)" in html
    assert "保存密钥到 .env" in html
    assert "cfg.persona.enabled" in load_block
    assert "cfg.persona.event_recording_enabled" in load_block
    assert "cfg.persona.api_key_masked" in load_block
    persona_block = save_block.split("candidate.persona = {", 1)[1].split("};", 1)[0]
    assert "enabled: document.getElementById('cfg-persona-enabled').value === 'true'," in persona_block
    assert "event_recording_enabled: document.getElementById('cfg-persona-event-recording').value === 'true'," in persona_block
    assert "model: document.getElementById('cfg-persona-model').value," in persona_block
    assert "base_url: document.getElementById('cfg-persona-url').value," in persona_block
    assert "persist_env: !!persistEnv" in save_block
    assert "if (!body.persona) body.persona = {};" in html
    assert "body.persona.api_key = personaKeyVal;" in html


def test_dashboard_exposes_gateway_upstream_editor():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]
    upstream_view = html.split('id="upstream-config-view"', 1)[1].split('id="config-view"', 1)[0]
    config_view = html.split('id="config-view"', 1)[1].split('id="detail-panel"', 1)[0]

    assert 'data-tab="upstream-config"' in html
    assert 'id="upstream-config-view"' in html
    assert 'id="upstream-config-status"' in upstream_view
    assert "<h3>Gateway 上游模型</h3>" in html
    assert "<h3>Gateway 上游模型</h3>" not in config_view
    assert 'id="cfg-upstreams-list"' in upstream_view
    assert "function renderGatewayUpstreams()" in html
    assert "function collectGatewayUpstreams(includeKeyValues)" in html
    assert "function collectGatewayUpstreamsForSave(includeKeyValues)" in html
    assert "gatewayUpstreams = (((cfg.gateway || {}).upstreams) || []).map(normalizeGatewayUpstream);" in load_block
    assert "document.getElementById('upstream-config-view').style.display = target === 'upstream-config' ? '' : 'none';" in html
    assert "if (target === 'upstream-config') loadConfig();" in html
    assert "if (activeTarget === 'upstream-config')" in save_block
    assert "var upstreamsResult = collectGatewayUpstreamsForSave(!!persistEnv);" in save_block
    assert "candidate.gateway = { upstreams: upstreamsResult.upstreams };" in save_block
    assert "api_key_values" in html
    assert "填写了真实 key 时，请使用“应用并保存密钥到 .env”" in html
    assert "'upstream-config-status'" in html
    assert "document.querySelectorAll('[id^=\"cfg-upstream-key-values-\"]')" in html


def test_dashboard_dream_controls_load_and_save_runtime_fields():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]
    dream_block = save_block.split("candidate.dream = {", 1)[1].split("};", 1)[0]
    dream_lines = [line.strip() for line in dream_block.splitlines()]

    assert 'id="cfg-dream-engine-enabled"' in html
    assert 'id="cfg-dream-inject"' in html
    assert 'id="cfg-dream-retain"' in html
    assert "document.getElementById('cfg-dream-engine-enabled').value = cfg.dream.enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-enabled').value = cfg.dream.auto_enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-inject').value = cfg.dream.inject_enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-retain').value = cfg.dream.retain_after_inject ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-enabled').value = cfg.dream.enabled" not in load_block
    assert "base_url: document.getElementById('cfg-dream-url').value," in dream_lines
    assert "enabled: document.getElementById('cfg-dream-engine-enabled').value === 'true'," in dream_lines
    assert "model: document.getElementById('cfg-dream-model').value," in dream_lines
    assert "inject_enabled: document.getElementById('cfg-dream-inject').value === 'true'," in dream_lines
    assert "retain_after_inject: document.getElementById('cfg-dream-retain').value === 'true'," in dream_lines


def test_dashboard_config_number_zero_values_are_preserved():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "document.getElementById('cfg-dehy-temp').value = cfg.dehydration.temperature ?? 0.1;" in load_block
    assert "document.getElementById('cfg-merge').value = cfg.merge_threshold ?? 90;" in load_block
    assert "temperature: floatValue('cfg-dehy-temp', 0.1)," not in save_block
    assert "merge_threshold: numberValue('cfg-merge', 90)," not in save_block
    assert "cfg.dehydration.temperature || 0.1" not in load_block
    assert "cfg.merge_threshold || 90" not in load_block
    assert "parseFloat(document.getElementById('cfg-dehy-temp').value) || 0.1" not in save_block
    assert "parseInt(document.getElementById('cfg-merge').value) || 90" not in save_block


def test_dashboard_import_file_input_resets_after_selection():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    import_block = html.split("// --- Import functions ---", 1)[1].split("async function pollImportStatus", 1)[0]

    assert "const selectedFile = fileInput.files[0];" in import_block
    assert "fileInput.value = '';" in import_block
    assert "startImport(selectedFile);" in import_block
