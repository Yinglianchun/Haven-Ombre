"""
fix_timeline_api.py
修复 server.py 里 timeline API 的方法名：
  _dt.load → _dt.get_full_data
  _dt.manual_edit → _dt.edit_entry
"""
from pathlib import Path

SRV = Path("/opt/Ombre-Brain/server.py")
src = SRV.read_text(encoding="utf-8")

# fix 1: load → get_full_data
OLD1 = "        data = _dt.load(date_str)"
NEW1 = "        data = _dt.get_full_data(date_str)"

# fix 2: manual_edit → edit_entry (with correct signature)
OLD2 = "        updated = _dt.manual_edit(date_str, hour, body)"
NEW2 = """\
        # edit_entry patches one field at a time; apply all fields from body
        updated = {}
        for field, value in body.items():
            if field in ("doing", "chatting", "mood", "reflection"):
                _dt.edit_entry(date_str, hour, field, str(value))
                updated[field] = value
        # return the full hour entry after edits
        refreshed = _dt.get_full_data(date_str)
        updated = refreshed.get("hours", {}).get(str(hour), updated)"""

fixes = [(OLD1, NEW1, "load→get_full_data"), (OLD2, NEW2, "manual_edit→edit_entry")]
for old, new, label in fixes:
    if old not in src:
        print(f"SKIP {label}: not found (already fixed?)")
    else:
        src = src.replace(old, new, 1)
        print(f"OK: {label}")

SRV.write_text(src, encoding="utf-8")
print("server.py written OK")
