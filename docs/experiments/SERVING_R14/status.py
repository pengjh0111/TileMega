#!/usr/bin/env python3
"""Read-only queue view, including steps not yet read by the scheduler."""
import argparse,json,os
from collections import Counter
from pathlib import Path
h=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--prefix');p.add_argument('--queue',type=Path,help='Show only this queue file; relative paths resolve inside the queue directory')
p.add_argument('--queue-dir',type=Path,default=h/'queue');p.add_argument('--state-dir',type=Path,default=h/'scheduler');a=p.parse_args()
state_path=a.state_dir/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
files=[a.queue_dir/a.queue] if a.queue else sorted(a.queue_dir.glob('queue_*.json'))
prefix=a.prefix if a.prefix is not None else ('' if a.queue else 'C')
steps=[s for q in files for s in json.loads(q.read_text())]
pid_path=a.state_dir/'scheduler.pid';pid=int(pid_path.read_text()) if pid_path.exists() else None
try:
    if pid is None:raise ProcessLookupError
    os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
print(f'scheduler PID={pid} alive={alive}; queue={a.queue or "all"}; prefix={prefix or "all"}')
counts=Counter()
for s in steps:
    if not s['name'].startswith(prefix):continue
    r=state.get(s['name'],{});status=r.get('status','pending');counts[status]+=1
    print(f'{s["name"]:32} {status:9} attempts={r.get("attempts",0)} rc={r.get("exit_code","-")}')
print(f'total={sum(counts.values())} '+' '.join(f'{k}={v}' for k,v in sorted(counts.items())))
