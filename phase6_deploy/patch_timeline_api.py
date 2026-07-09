"""
patch_timeline_api.py
在 server.py 的 /api/reflection/run 路由后插入两条时间线 API：
  GET  /api/timeline/{date}          → 返回整天 JSON
  PATCH /api/timeline/{date}/{hour}   → 手动编辑某小时条目
"""
import sys
from pathlib import Path

SRV = Path("/opt/Ombre-Brain/server.py")
src = SRV.read_text(encoding="utf-8")

if "/api/timeline/" in src:
    print("SKIP: timeline API already present")
    sys.exit(0)

ANCHOR = '@mcp.custom_route("/dashboard", methods=["GET"])'

if ANCHOR not in src:
    print("ERROR: anchor not found")
    sys.exit(1)

TIMELINE_ROUTES = '''\
@mcp.custom_route("/api/timeline/{date}", methods=["GET"])
async def api_timeline_get(request):
    """Return daily timeline JSON for a given date (YYYY-MM-DD)."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    date_str = request.path_params.get("date", "")
    import re
    if not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", date_str):
        return JSONResponse({"error": "invalid date format, use YYYY-MM-DD"}, status_code=400)
    try:
        from daily_timeline import DailyTimeline as _DT
        _dt = _DT(config=config)
        data = _dt.load(date_str)
        return JSONResponse(data)
    except Exception as e:
        logger.warning("Timeline GET failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/timeline/{date}/{hour}", methods=["PATCH"])
async def api_timeline_patch(request):
    """Manually edit one hour entry of the daily timeline."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    date_str = request.path_params.get("date", "")
    hour_str = request.path_params.get("hour", "")
    import re
    if not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", date_str):
        return JSONResponse({"error": "invalid date"}, status_code=400)
    if not re.fullmatch(r"\\d{1,2}", hour_str):
        return JSONResponse({"error": "invalid hour"}, status_code=400)
    hour = int(hour_str)
    if not (0 <= hour <= 23):
        return JSONResponse({"error": "hour must be 0-23"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be object"}, status_code=400)
    try:
        from daily_timeline import DailyTimeline as _DT
        _dt = _DT(config=config)
        updated = _dt.manual_edit(date_str, hour, body)
        return JSONResponse({"ok": True, "hour": hour, "entry": updated})
    except Exception as e:
        logger.warning("Timeline PATCH failed | %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


'''

src = src.replace(ANCHOR, TIMELINE_ROUTES + ANCHOR, 1)
SRV.write_text(src, encoding="utf-8")
print("OK: timeline API routes inserted")
