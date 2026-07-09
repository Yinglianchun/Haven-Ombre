import sqlite3, json

conn = sqlite3.connect('/state/raw_events.sqlite')
cols = [c[1] for c in conn.execute("PRAGMA table_info(raw_events)").fetchall()]

# 拉今日所有数据（UTC 2026-07-02 22:00 起 = 北京时间 7-03 06:00起）
# 其实从 2026-07-02 全天看
rows = conn.execute(
    "SELECT created_at, role, text FROM raw_events "
    "WHERE created_at >= '2026-07-02T22:00' "
    "ORDER BY created_at"
).fetchall()
print(f"=== 今日(北京时间7月3日)对话: {len(rows)} 条 ===")
for ts, role, text in rows:
    # UTC转北京时间(+8)
    from datetime import datetime, timezone, timedelta
    try:
        dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
        bj = dt.astimezone(timezone(timedelta(hours=8)))
        bj_str = bj.strftime('%H:%M')
    except:
        bj_str = ts[11:16]
    t = str(text or "")[:200].replace("\n"," ")
    print(f"  BJ{bj_str} [{role}] {t}")

conn.close()
