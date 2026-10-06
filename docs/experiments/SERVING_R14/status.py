#!/usr/bin/env python3
"""Read-only queue view, including steps not yet read by the scheduler."""
import argparse,json,os
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefix',default='C');a=p.parse_args()
h=Path(__file__).resolve().parent
state=json.loads((h/'scheduler/state.json').read_text())
steps=[s for q in sorted((h/'queue').glob('queue_*.json')) for s in json.loads(q.read_text())]
pid=int((h/'scheduler/scheduler.pid').read_text())
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
print(f'scheduler PID={pid} alive={alive}; prefix={a.prefix or "all"}')
counts=Counter()
for s in steps:
    if not s['name'].startswith(a.prefix):continue
    r=state.get(s['name'],{});status=r.get('status','pending');counts[status]+=1
    print(f'{s["name"]:32} {status:9} attempts={r.get("attempts",0)} rc={r.get("exit_code","-")}')
print(' '.join(f'{k}={v}' for k,v in sorted(counts.items())))
