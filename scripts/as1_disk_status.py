#!/usr/bin/env python3
"""Measure only AS-1 output subtrees and the personal root's filesystem."""
import json
from pathlib import Path
import shutil
import subprocess
import datetime as dt

ROOT = Path('/home/d50064638/junhuip')
WORK = ROOT / 'tilemega-as1'
RUNS = ROOT / 'TileMega/runs/as1'


def main(project='as1'):
    if project not in {'as1', 'as2'}:
        raise ValueError('Unknown controlled project')
    work = ROOT / ('tilemega-' + project)
    runs = ROOT / ('TileMega/runs/' + project)
    usage = shutil.disk_usage(ROOT)
    import os
    fs = os.statvfs(ROOT)
    report = {'disk_total_bytes': usage.total, 'disk_free_bytes': usage.free,
              'inode_total': fs.f_files, 'inode_available': fs.f_favail,
              'quota': 'unavailable: quota utility not installed',
              'warning': usage.free/usage.total < 0.15 or fs.f_favail/fs.f_files < 0.05,
              'project_usage': {}}
    report['sampled_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    names = ('build', 'cache', 'tmp', 'weights') if project == 'as2' else ('build', 'cache', 'tmp')
    for p in [runs, *(work / n for n in names)]:
        r = subprocess.run(['du', '-s', '-B1', '--', str(p)], capture_output=True, text=True)
        report['project_usage'][str(p)] = int(r.stdout.split()[0]) if r.returncode == 0 else None
    history = runs / 'cleanup_history.jsonl'
    all_work = subprocess.run(['du', '-s', '-B1', '--', str(work)], capture_output=True, text=True)
    report['work_total_bytes_including_protected_dependencies'] = (
        int(all_work.stdout.split()[0]) if all_work.returncode == 0 else None)
    report['last_cleanup'] = json.loads(history.read_text().splitlines()[-1])['time'] if history.exists() else None
    report['generated_bytes'] = sum(n for n in report['project_usage'].values() if n is not None)
    policy = json.loads((ROOT/('TileMega/configs/bench/'+project+'_disk_policy.json')).read_text())
    report['warning'] = (usage.free/usage.total < policy.get('stop_below_free_fraction', 0.15) or
                         fs.f_favail/fs.f_files < policy.get('stop_below_free_inode_fraction', 0.05))
    report['generated_soft_budget_bytes'] = policy['project_generated_soft_limit_bytes']
    report['warning'] |= report['generated_bytes'] > report['generated_soft_budget_bytes']
    samples = runs/'disk_samples.jsonl'
    previous = [json.loads(line) for line in samples.read_text().splitlines()] if samples.exists() else []
    report['sampled_peak_generated_bytes'] = max([report['generated_bytes'], *(s['generated_bytes'] for s in previous)])
    with samples.open('a') as stream:
        stream.write(json.dumps(report)+'\n')
    print(json.dumps(report, indent=2))
    return 75 if report['warning'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
