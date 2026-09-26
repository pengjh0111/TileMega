#!/usr/bin/env python3
"""Start final EV-1 only after all four corrected plan chains exit."""
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/TileMega')
work = Path('/root/r10_work')
chains = (3206167, 3206175, 3206186, 3206197)
deadline = time.monotonic() + 12 * 60 * 60
while time.monotonic() < deadline:
    live = []
    for pid in chains:
        path = Path(f'/proc/{pid}/cmdline')
        try:
            command = path.read_bytes().replace(b'\0', b' ')
        except FileNotFoundError:
            continue
        if b'run_plan_matrix.py' in command:
            live.append(pid)
    if not live:
        break
    time.sleep(30)
else:
    raise SystemExit('plan matrix exceeded 12 hours')
plans = list((work / 'plans').glob('*/plan.so.plan.json'))
if len(plans) < 20:
    raise SystemExit(f'plan matrix exited with only {len(plans)}/20 plans')
print('20/20 plans exist; checking cross-library symbol isolation', flush=True)
subprocess.run([sys.executable, str(root / 'docs/experiments/SERVING_R10/relink_plan_tables.py')],
               cwd=root, check=True)
print('symbol isolation complete; starting final EV-1', flush=True)
subprocess.run([sys.executable, str(root / 'docs/experiments/SERVING_R10/run_e2e_matrix.py')],
               cwd=root, check=True)
print('EV-1 complete', flush=True)
