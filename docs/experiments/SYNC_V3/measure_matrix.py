#!/usr/bin/env python3
"""Rotate mechanisms, placement and probe arms together in fresh processes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ('full', 'nofence', 'nowait', 'neither', 'l1nosync')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.manifest.read_text())
    session = str(time.time_ns())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'session.json').write_text(json.dumps(dict(session=session,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        configurations=specs, rounds=25), indent=2)+'\n')
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            pairs = [(name, p, arm) for name in specs for p in (0, 5) for arm in ARMS]
            for round_ in range(25):
                for order in range(len(pairs)):
                    name, placement, arm = pairs[(round_ + order) % len(pairs)]
                    source = (REPO / specs[name]).resolve()
                    key = f'{model}_p{placement}_{arm}'
                    build_meta = json.loads((source / 'log' / f'{key}.build.json').read_text())
                    binary = source / 'bin' / key
                    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
                    if digest != build_meta['binary_sha256']:
                        raise ValueError(f'binary changed after compilation: {binary}')
                    folder = out / name / 'paired' / f'{model}_s{seq}_p{placement}' / arm
                    folder.mkdir(parents=True, exist_ok=True)
                    log = folder / f'r{round_}.log'
                    if log.exists():
                        raise ValueError(f'refusing to overwrite prior session: {log}')
                    cmd = [str(binary), str(REPO / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3')]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    output = result.stdout + result.stderr
                    log.write_text(output)
                    log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,
                        exit_code=result.returncode, session=session, round=round_, order=order,
                        configuration=name, combinations=len(pairs), time_ns=time.time_ns(),
                        build=build_meta, binary_sha256=digest))+'\n')
                    if len(re.findall(r'^E2E_TIME ', output, re.M)) != 1:
                        raise ValueError(f'missing timing: {log}')
                    if arm == 'full' and (result.returncode or 'RESULT status=PASS' not in output):
                        raise ValueError(f'full correctness failed: {log}')
                print(f'MATRIX {model} s{seq} round={round_+1}/25 configurations={len(specs)}', flush=True)
    (out / 'status.txt').write_text('PASS\n')


if __name__ == '__main__':
    main()
