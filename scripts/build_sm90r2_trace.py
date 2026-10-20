#!/usr/bin/env python3
"""Add stage-boundary timestamps to an immutable SM90R2 library."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    original, output = a.input.resolve(), a.out.resolve()
    if not output.is_relative_to(ROOT / 'runs/sm90r2') or output == original:
        p.error('diagnostic outputs belong under runs/sm90r2')
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.build.identity import generate, verify
    identity = verify(original)
    if identity['trace'] or output.exists():
        raise ValueError('a new trace output and an uninstrumented input are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.cu', '.plan.json', '.source.json'):
        shutil.copy2(str(original) + suffix, str(output) + suffix)
    command = shlex.split(Path(str(original) + '.build_command.txt').read_text())
    if command.count(str(original)) != 1 or command.count(str(original) + '.cu') != 1:
        raise ValueError('ambiguous build command')
    command = [str(output) if x == str(original) else str(output) + '.cu'
               if x == str(original) + '.cu' else x for x in command]
    command[1:1] = ['-lineinfo', '-DTILEMEGA_TRACE_STAGE=1', '-DTILEMEGA_TRACE_STEP=1']
    Path(str(output) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    with Path(str(output) + '.ptxas.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = generate(output, identity['source']['root'], str(output) + '.source.json')
    if not result['trace']:
        raise ValueError('trace identity was not recorded')
    record = dict(reference_artifact_id=identity['artifact_id'], trace_artifact_id=result['artifact_id'],
                  performance_eligible=False, method='same generated program; stage and step timestamps only')
    Path(str(output) + '.diagnostic.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
