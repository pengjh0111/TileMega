#!/usr/bin/env python3
"""Append file identities to the AS-1 manifest; retention is the default."""
import argparse
import datetime as dt
import json
from pathlib import Path
from as1_cleanup import DEFAULT_RUN, DEFAULT_WORK, file_stamp, plain_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--build-id')
    parser.add_argument('--completed', action='store_true')
    parser.add_argument('--discardable', action='store_true')
    parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    roots = [DEFAULT_RUN, *(DEFAULT_WORK / n for n in ('build', 'cache', 'tmp'))]
    path = args.path.absolute()
    if path not in roots:
        plain_path(path, roots)
    active = []
    if args.pid:
        value = Path('/proc', str(args.pid), 'stat').read_text()
        active.append(dict(pid=args.pid, start_ticks=int(value[value.rfind(')')+2:].split()[19])))
    files = [path] if path.is_file() else sorted(path.rglob('*'))
    job_cache = {}
    count = 0
    with (DEFAULT_RUN / 'manifest.jsonl').open('a') as manifest:
        for p in files:
            if p.is_symlink() or not p.is_file() or p.name in {'manifest.jsonl', 'cleanup_history.jsonl'}:
                continue
            plain_path(p, roots)
            row = dict(path=str(p), stage=args.stage, category=args.category,
                       keep=not args.discardable, regenerable=args.discardable,
                       completed=args.completed, active_refs=active,
                       created_at=dt.datetime.now(dt.timezone.utc).isoformat(), stamp=file_stamp(p))
            for parent in p.parents:
                if not parent.is_relative_to(DEFAULT_RUN):
                    break
                receipt = parent/'job.json'
                if not receipt.is_file():
                    continue
                if parent not in job_cache:
                    try:
                        job_cache[parent] = json.loads(receipt.read_text())
                    except (OSError, json.JSONDecodeError):
                        job_cache[parent] = {'status': 'unknown'}
                job = job_cache[parent]
                if job.get('status') not in {'passed', 'failed', 'timeout', 'interrupted', 'launch_error'}:
                    row['completed'] = False
                    if job.get('pid'):
                        try:
                            value = Path('/proc',str(int(job['pid'])),'stat').read_text()
                            row['active_refs'] = [*active,dict(pid=int(job['pid']),
                                start_ticks=int(value[value.rfind(')')+2:].split()[19]))]
                        except (OSError, ValueError):
                            row['keep'] = True
                elif job.get('status') != 'passed':
                    row.update(keep=True, regenerable=False, category='failure')
                break
            if args.build_id:
                row['build_id'] = args.build_id
            manifest.write(json.dumps(row)+'\n')
            count += 1
    print(json.dumps({'registered': count, 'path': str(path), 'retained': not args.discardable}))


if __name__ == '__main__':
    main()
