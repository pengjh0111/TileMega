#!/usr/bin/env python3
"""Run independent official asset downloads as one CPU scheduler job."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def read_jobs(queue):
    jobs, destinations = [], set()
    for path in sorted(queue.glob('queue_*.json')):
        for step in json.loads(path.read_text()):
            command = step['command']
            if step.get('gpu') or len(command) < 2 or \
                    Path(command[1]).resolve() != HERE / 'fetch_asset.py':
                raise ValueError('only independent CPU fetch_asset jobs can run concurrently')
            destination = Path(command[command.index('--out') + 1]).resolve()
            if destination in destinations or '/' in step['name']:
                raise ValueError('duplicate destination or invalid asset name')
            destinations.add(destination)
            jobs.append(dict(name=step['name'], command=command,
                             destination=str(destination), cwd=step.get('cwd')))
    if not jobs:
        raise ValueError('empty asset queue')
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    jobs = read_jobs(args.queue_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    def run(job):
        with (args.out / (job['name'] + '.log')).open('w') as stream:
            code = subprocess.run(job['command'], cwd=job['cwd'], stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
        result = dict(name=job['name'], destination=job['destination'], exit_code=code)
        identity = Path(job['destination'] + '.download.json')
        if code == 0:
            result['identity'] = json.loads(identity.read_text())
        return result

    result = dict(evidence='verified', passed=False, workers=args.workers, assets=[])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            record = future.result()
            result['assets'].append(record)
            temporary = args.out / 'result.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(args.out / 'result.json')
            print(json.dumps(dict(event='asset_complete', name=record['name'],
                                  exit_code=record['exit_code'])), flush=True)
    result['passed'] = all(record['exit_code'] == 0 for record in result['assets'])
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
