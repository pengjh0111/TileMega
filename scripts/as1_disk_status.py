#!/usr/bin/env python3
"""Measure only AS-1 output subtrees and the personal root's filesystem."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/d50064638/junhuip')
WORK = ROOT / 'tilemega-as1'
RUNS = ROOT / 'TileMega/runs/as1'


def main():
    usage = shutil.disk_usage(ROOT)
    import os
    fs = os.statvfs(ROOT)
    report = {'disk_total_bytes': usage.total, 'disk_free_bytes': usage.free,
              'inode_total': fs.f_files, 'inode_available': fs.f_favail,
              'quota': 'unavailable: quota utility not installed',
              'warning': usage.free/usage.total < 0.15 or fs.f_favail/fs.f_files < 0.05,
              'project_usage': {}}
    for p in [RUNS, *(WORK / n for n in ('build', 'cache', 'tmp'))]:
        r = subprocess.run(['du', '-s', '-B1', '--', str(p)], capture_output=True, text=True)
        report['project_usage'][str(p)] = int(r.stdout.split()[0]) if r.returncode == 0 else None
    history = RUNS / 'cleanup_history.jsonl'
    report['last_cleanup'] = json.loads(history.read_text().splitlines()[-1])['time'] if history.exists() else None
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
