"""
patch_event_line_api.py -- server.py 事件线 Dashboard API
"""
from __future__ import annotations

import sys
from pathlib import Path

SERVER = Path("/opt/Ombre-Brain/server.py")
MARKER = '@mcp.custom_route("/api/event-line"'
DASHBOARD_MARKER = '@mcp.custom_route("/dashboard", methods=["GET"])'

src = SERVER.read_text(encoding="utf-8")
if MARKER in src:
    print("SKIP: event-line API already present")
    sys.exit(0)

if DASHBOARD_MARKER not in src:
    print("ERROR: dashboard route anchor not found")
    sys.exit(1)

API_BLOCK = '''
def _dashboard_event_line():
    from event_line import EventLine as _EL
    state_dir = str(config.get("state_dir") or "/state")
    persona_file = os.path.join(state_dir, "system_standard.txt")
    return _EL(state_dir=state_dir, persona_file=persona_file)


@mcp.custom_route("/api/event-line", methods=["GET"])
async def api_event_line_list(request):
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        el = _dashboard_event_line()
        events = el.list_events()
        stale_ids = [e.get("id") for e in el.check_stale()]
        return JSONResponse({"events": events, "stale_ids": stale_ids})
    except Exception as e:
        logger.warning("EventLine GET failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/event-line", methods=["POST"])
async def api_event_line_create(request):
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be object"}, status_code=400)
    title = str(body.get("title") or "").strip()
    text = str(body.get("text") or "").strip()
    if not title or not text:
        return JSONResponse({"error": "title and text are required"}, status_code=400)
    progress = body.get("progress")
    if progress is not None:
        try:
            progress = int(progress)
        except Exception:
            return JSONResponse({"error": "progress must be int"}, status_code=400)
    try:
        el = _dashboard_event_line()
        event_id = el.create_event(title, text, progress=progress)
        return JSONResponse({"ok": True, "id": event_id, "event": el.get_event(event_id)})
    except Exception as e:
        logger.warning("EventLine POST failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/event-line/{event_id}", methods=["PATCH"])
async def api_event_line_patch(request):
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    event_id = str(request.path_params.get("event_id") or "")
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be object"}, status_code=400)
    try:
        el = _dashboard_event_line()
        if "title" in body:
            ok, msg = el.update_title(event_id, str(body.get("title") or ""))
            if not ok:
                return JSONResponse({"error": msg}, status_code=404)
        return JSONResponse({"ok": True, "event": el.get_event(event_id)})
    except Exception as e:
        logger.warning("EventLine PATCH failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/event-line/{event_id}", methods=["DELETE"])
async def api_event_line_delete(request):
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    event_id = str(request.path_params.get("event_id") or "")
    try:
        el = _dashboard_event_line()
        if not el.delete_event(event_id):
            return JSONResponse({"error": "event_id 不存在"}, status_code=404)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.warning("EventLine DELETE failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/event-line/{event_id}/entries/{entry_index}", methods=["PATCH"])
async def api_event_line_entry_patch(request):
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    event_id = str(request.path_params.get("event_id") or "")
    entry_index = request.path_params.get("entry_index", "")
    if not str(entry_index).isdigit():
        return JSONResponse({"error": "invalid entry_index"}, status_code=400)
    idx = int(entry_index)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be object"}, status_code=400)
    progress = ...
    if "progress" in body:
        raw = body.get("progress")
        if raw is None or raw == "":
            progress = None
        else:
            try:
                progress = int(raw)
            except Exception:
                return JSONResponse({"error": "progress must be int or null"}, status_code=400)
    try:
        el = _dashboard_event_line()
        ok, msg = el.edit_entry(
            event_id,
            idx,
            text=str(body["text"]).strip() if "text" in body else None,
            progress=progress,
            date=str(body["date"]).strip() if "date" in body else None,
        )
        if not ok:
            return JSONResponse({"error": msg}, status_code=404)
        return JSONResponse({"ok": True, "event": el.get_event(event_id)})
    except Exception as e:
        logger.warning("EventLine entry PATCH failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/event-line/{event_id}/entries/{entry_index}", methods=["DELETE"])
async def api_event_line_entry_delete(request):
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    event_id = str(request.path_params.get("event_id") or "")
    entry_index = request.path_params.get("entry_index", "")
    if not str(entry_index).isdigit():
        return JSONResponse({"error": "invalid entry_index"}, status_code=400)
    idx = int(entry_index)
    try:
        el = _dashboard_event_line()
        ok, msg = el.delete_entry(event_id, idx)
        if not ok:
            return JSONResponse({"error": msg}, status_code=404)
        return JSONResponse({"ok": True, "event": el.get_event(event_id)})
    except Exception as e:
        logger.warning("EventLine entry DELETE failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


'''

# Need Path import in server if missing
if "from pathlib import Path" not in src and "import pathlib" not in src:
    src = src.replace("import os\n", "import os\nfrom pathlib import Path\n", 1)

src = src.replace(DASHBOARD_MARKER, API_BLOCK + DASHBOARD_MARKER, 1)
SERVER.write_text(src, encoding="utf-8")
print("OK: event-line API inserted")
