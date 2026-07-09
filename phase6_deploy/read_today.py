import sqlite3, json

# raw_events - 今日对话
conn = sqlite3.connect('/state/raw_events.sqlite')
try:
    rows = conn.execute(
        "SELECT timestamp, role, content FROM raw_events WHERE timestamp >= '2026-07-03' ORDER BY timestamp"
    ).fetchall()
except Exception as e:
    rows = []
    print("raw_events error:", e)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print("tables:", tables)
    except:
        pass

print(f"=== raw_events today: {len(rows)} rows ===")
for ts, role, content in rows[:40]:
    c = str(content or "")[:150].replace("\n", " ")
    print(f"  {ts} [{role}] {c}")

conn.close()

# phase5_context_ring - 最近7轮
try:
    with open('/state/phase5_context_ring.json') as f:
        ring = json.load(f)
    print("\n=== phase5_context_ring ===")
    turns = ring.get("turns", [])
    for i, t in enumerate(turns):
        u = str(t.get("user",""))[:100].replace("\n"," ")
        a = str(t.get("assistant",""))[:100].replace("\n"," ")
        ts = t.get("timestamp","")
        print(f"  [{i}] {ts} U:{u}")
        print(f"       A:{a}")
    summary = ring.get("summary","")
    if summary:
        print(f"\n  摘要: {summary[:200]}")
except Exception as e:
    print("ring error:", e)

# conversation_turns sqlite
try:
    conn2 = sqlite3.connect('/state/persona_state.db')
    tables2 = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("\n=== persona_state.db tables:", [t[0] for t in tables2])
    if any('turn' in t[0].lower() or 'conv' in t[0].lower() for t in tables2):
        for tname, in tables2:
            if 'turn' in tname.lower() or 'conv' in tname.lower():
                rows2 = conn2.execute(f"SELECT * FROM {tname} WHERE timestamp >= '2026-07-03' LIMIT 20").fetchall()
                print(f"  {tname}: {len(rows2)} rows today")
                for r in rows2[:5]:
                    print(" ", str(r)[:150])
    conn2.close()
except Exception as e:
    print("persona_state error:", e)
