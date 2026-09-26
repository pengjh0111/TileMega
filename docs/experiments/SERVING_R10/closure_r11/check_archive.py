#!/usr/bin/env python3
"""Check archived evidence integrity only; never launches an experiment."""
from pathlib import Path
import gzip, hashlib, json

HERE = Path(__file__).resolve().parent
failures = []
def check(label, ok, value):
    print(f'{label} {"PASS" if ok else "FAIL"}: {value}')
    if not ok:
        failures.append(label)

for item in json.loads((HERE / 'compressed_raw.json').read_text()):
    data = gzip.decompress((HERE / item['archived_path']).read_bytes())
    check('SHA256 ' + item['original_path'],
          hashlib.sha256(data).hexdigest() == item['sha256'], item['sha256'])
plans = [json.loads(p.read_text()) for p in (HERE / 'historical_b22/plans').glob('*/result.json')]
complete = [p for p in plans if p['returncode'] == 0]
check('historical plans', len(complete) == 18, f'{len(complete)}/20 completed (b22)')
check('historical budget failure recorded', all(p['seconds'] > 600 for p in complete),
      f'{sum(p["seconds"] > 600 for p in complete)}/18 over 600 s')
for p in (HERE / 'seed_checks').glob('*/mode_check.json'):
    d = json.loads(p.read_text())
    check(p.parent.name + ' mode archive', d['pass'] and d['mismatches'] == 0, d['tokens'])
for p in (HERE / 'seed_checks').glob('*/hf_check.json'):
    d = json.loads(p.read_text())
    check(p.parent.name + ' HF archive', d['pass'] and d['gap_le_0_5_ratio'] >= 0.99 and d['max_gap'] <= 3,
          f'positions={d["positions"]}, max_gap={d["max_gap"]}')
stop = json.loads((HERE / 'stopped_processes.json').read_text())
check('recorded stop', not stop['survivors'], f'{len(stop["terminated"])} terminated; {len(stop["survivors"])} survivors at stop')
print('Archive integrity only; no missing R10 acceptance gate is promoted to PASS.')
raise SystemExit(bool(failures))
