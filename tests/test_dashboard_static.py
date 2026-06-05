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
def test_dashboard_exposes_gateway_memory_cooldown_settings():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    config_view = html.split('id="config-view"', 1)[1].split('id="memory-config-view"', 1)[0]

    assert 'data-tab="memory-config">记忆浮现' in html
    assert 'id="memory-config-view"' in html
    assert "memory-config-view').style.display = target === 'memory-config'" in html
    assert "if (target === 'memory-config') loadConfig();" in html
    assert "<h3>记忆浮现</h3>" in html
    assert 'id="cfg-gateway-cooldown"' in html
    assert 'id="cfg-gateway-rounds"' in html
    assert 'id="cfg-recent-context-enabled"' in html
    assert 'id="cfg-recent-context-budget"' in html
    assert 'id="cfg-persona-context-enabled"' in html
    assert 'id="cfg-persona-context-rounds"' in html
    assert "cfg.gateway.cooldown_hours" in html
    assert "cfg.gateway.skip_recent_rounds" in html
    assert "cfg.gateway.recent_context_budget" in html
    assert "cfg.gateway.current_inner_state_interval_rounds" in html
    assert "cooldown_hours: floatValue('cfg-gateway-cooldown', 6)" in html
    assert "skip_recent_rounds: numberValue('cfg-gateway-rounds', 5)" in html
    assert "recent_context_budget: recentContextBudget" in html
    assert "current_inner_state_interval_rounds: personaContextRounds" in html
    assert 'id="cfg-recent-context-enabled"' not in config_view
    assert "memory_diffusion" not in html
    assert "retrieval_mode" not in html


def test_dashboard_config_save_sends_only_changed_sections():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("try {", 1)[0]

    assert "let configSnapshot = null;" in html
    assert "configSnapshot = JSON.parse(JSON.stringify(cfg || {}));" in load_block
    assert "function addChangedSection(target, sectionName, candidate)" in save_block
    assert "var activeTarget = activeTab ? activeTab.dataset.tab : '';" in save_block
    assert "var body = {" in save_block
    assert "if (candidate[sectionName]) addChangedSection(body, sectionName, candidate[sectionName]);" in save_block
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
    assert "cfg-reflection-enabled" not in reflection_block
    assert "cfg-reflection-model" not in reflection_block
    assert "if (!body.reflection) body.reflection = {};" in html
    assert "body.reflection.api_key = reflectionKeyVal;" in html


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
    persona_block = save_block.split("candidate.persona = {", 1)[1].split("};", 1)[0]
    assert "base_url: document.getElementById('cfg-persona-url').value," in persona_block
    assert "cfg-persona-enabled" not in persona_block
    assert "cfg-persona-model" not in persona_block
    assert "persist_env: !!persistEnv" in save_block
    assert "if (!body.persona) body.persona = {};" in html
    assert "body.persona.api_key = personaKeyVal;" in html


def test_dashboard_dream_controls_load_but_config_save_only_writes_url_and_key():
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
    assert not any("cfg-dream-engine-enabled" in line for line in dream_lines)
    assert not any("cfg-dream-model" in line for line in dream_lines)
    assert not any("cfg-dream-inject" in line for line in dream_lines)


def test_dashboard_config_number_zero_values_are_preserved():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "document.getElementById('cfg-dehy-temp').value = cfg.dehydration.temperature ?? 0.1;" in load_block
    assert "document.getElementById('cfg-merge').value = cfg.merge_threshold ?? 75;" in load_block
    assert "temperature: floatValue('cfg-dehy-temp', 0.1)," not in save_block
    assert "merge_threshold: numberValue('cfg-merge', 75)," not in save_block
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
