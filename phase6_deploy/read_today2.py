import sqlite3, json

# raw_events 字段探查
conn = sqlite3.connect('/state/raw_events.sqlite')
cols = conn.execute("PRAGMA table_info(raw_events)").fetchall()
print("=== raw_events columns:", [c[1] for c in cols])
rows = conn.execute(
    "SELECT * FROM raw_events ORDER BY rowid DESC LIMIT 30"
).fetchall()
print(f"Total recent rows: {len(rows)}")
col_names = [c[1] for c in cols]
ts_col = next((c for c in col_names if 'time' in c.lower() or 'date' in c.lower() or 'at' in c.lower()), col_names[0])
print(f"Using timestamp col: {ts_col}")

for row in rows[:20]:
    d = dict(zip(col_names, row))
    ts = d.get(ts_col, "")
    role = d.get("role", d.get("type",""))
    content = str(d.get("content", d.get("text", d.get("body",""))))[:150].replace("\n"," ")
    print(f"  {ts} [{role}] {content}")

conn.close()

# persona_exchange_log
try:
    conn2 = sqlite3.connect('/state/persona_state.db')
    cols2 = conn2.execute("PRAGMA table_info(persona_exchange_log)").fetchall()
    print("\n=== persona_exchange_log columns:", [c[1] for c in cols2])
    rows2 = conn2.execute("SELECT * FROM persona_exchange_log ORDER BY rowid DESC LIMIT 20").fetchall()
    col_names2 = [c[1] for c in cols2]
    for row in rows2[:15]:
        d = dict(zip(col_names2, row))
        print(" ", str(d)[:200])
    conn2.close()
except Exception as e:
    print("exchange_log error:", e)

# phase5_context_ring full turns
try:
    with open('/state/phase5_context_ring.json') as f:
        ring = json.load(f)
    turns = ring.get("turns", [])
    print(f"\n=== context ring turns: {len(turns)} ===")
    for i, t in enumerate(turns):
        ts = t.get("timestamp","")
        u = str(t.get("user",""))[:200].replace("\n"," ")
        a = str(t.get("assistant",""))[:200].replace("\n"," ")
        print(f"  [{i}] {ts}")
        print(f"    U: {u}")
        print(f"    A: {a}")
except Exception as e:
    print("ring error:", e)
