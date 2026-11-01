"""Run five fresh probe processes inside an already guarded GPU group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.getenv('TILEMEGA_GPU_LOCK_HELD') != '1' or not os.getenv('TILEMEGA_DEVICE_GROUP'):
        raise RuntimeError('probe runner requires mx_gpu_group_job.sh')
    args.out.mkdir(parents=True, exist_ok=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command = [str(args.exe.resolve()), *arguments]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    identity = dict(source=str(args.source), source_sha256=sha(args.source),
                    binary=str(args.exe), binary_sha256=sha(args.exe),
                    source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    gpu_indices=list(map(int, os.environ['TILEMEGA_DEVICE_GROUP'].split(','))),
                    command=command)
    (args.out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    okay = 0
    for fresh in range(5):
        out = args.out / f'process{fresh}'
        out.mkdir(exist_ok=False)
        start = time.monotonic()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = time.monotonic() - start
        (out / 'stdout.jsonl').write_text(result.stdout)
        (out / 'stderr.txt').write_text(result.stderr)
        records = []
        for line in result.stdout.splitlines():
            if line.startswith('{'):
                records.append(json.loads(line))
        passed = result.returncode == 0 and records and all(r.get('pass', True) for r in records)
        okay += bool(passed)
        record = dict(identity, fresh_process=fresh, returncode=result.returncode,
                      elapsed_seconds=duration, records=records, passed=bool(passed))
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(fresh_process=fresh, returncode=result.returncode,
                              elapsed_seconds=duration, records=records)), flush=True)
    if sha(args.source) != identity['source_sha256'] or sha(args.exe) != identity['binary_sha256']:
        raise RuntimeError('probe source/binary identity changed')
    (args.out / 'summary.json').write_text(json.dumps(dict(passed_processes=okay, fresh_processes=5), indent=2) + '\n')
    return 0 if okay == 5 else 2


if __name__ == '__main__':
    raise SystemExit(main())
