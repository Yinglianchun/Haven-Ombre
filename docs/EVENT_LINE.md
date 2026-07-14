# Event Line (phase6)

Reusable cross-day event tracking for Ombre-Brain shadow forks.

## Files
- `event_line.py` — core store + sync into `EVENT_LINE` cache block
- `cache_activity_zone.py` — BEGIN/END block helpers (shared by timeline/alarm/I)
- `phase6_deploy/patch_*.py` — reference patches for gateway / dashboard / API
- `examples/event_line/*.json` — empty template + fictional sample (no personal data)

## Tags
```xml
<event_create title="..." progress="10">...</event_create>
<event_update id="evt_xxx" progress="40">...</event_update>
<event_close id="evt_xxx">...</event_close>
```

Apply patches against your local gateway/server anchors before deploying.
