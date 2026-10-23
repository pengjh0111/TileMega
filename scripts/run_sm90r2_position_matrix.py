#!/usr/bin/env python3
"""Run an immutable native position matrix inside one guarded GPU lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from sm90d1_common import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('a scheduler/guard GPU lease is required')
    if os.environ.get('TILEMEGA_DEVICE_INDEX') == '6':
        raise ValueError('GPU6 is excluded')
    manifest = read(a.manifest)
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in manifest['cases']:
        binary = a.manifest.parent / case['name']
        if case['exit_code'] or sha(binary) != case['binary_sha256']:
            raise ValueError('position program changed or failed to build')
        log = a.out / (case['name'] + '.log')
        with log.open('w') as stream:
            result = subprocess.run([str(binary.resolve())], stdout=stream, stderr=subprocess.STDOUT)
        observations = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        summary = next((r for r in observations if 'pass' in r), {})
        rows.append(dict(name=case['name'], case=case['case'], binary_sha256=sha(binary),
            exit_code=result.returncode, summary=summary,
            exclusions=[r for r in observations if r.get('excluded')]))
    passed = all(not r['exit_code'] and r['summary'].get('pass') for r in rows)
    write(a.out / 'receipt.json', dict(pass_=passed, manifest_sha256=sha(a.manifest),
        fresh_program_processes=len(rows), programs=rows, purpose='development correctness only'))
    print(json.dumps(dict(pass_=passed, programs=len(rows),
        position_cases=sum(r['summary'].get('position_cases', 0) for r in rows))))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
