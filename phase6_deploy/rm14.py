import json
p = '/state/timeline_2026-07-03.json'
d = json.load(open(p))
d['hours'].pop('14', None)
open(p, 'w').write(json.dumps(d, ensure_ascii=False, indent=2))
print('done, remaining:', list(d['hours'].keys()))
