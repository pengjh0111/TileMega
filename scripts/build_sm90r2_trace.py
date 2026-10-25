#!/usr/bin/env python3
"""Add stage-boundary timestamps to an immutable SM90R2 library."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native-phases', action='store_true')
    a = p.parse_args()
    original, output = a.input.resolve(), a.out.resolve()
    runs = (ROOT / 'runs').resolve()
    if not any(output.is_relative_to(runs / r) for r in ('sm90r2', 'sm90r3')) or output == original:
        p.error('diagnostic outputs belong under runs/sm90r2 or runs/sm90r3')
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.build.identity import generate, verify, validate_snapshot_inputs
    identity = verify(original)
    validate_snapshot_inputs(identity['source']['root'], identity['source'])
    if identity['trace'] or output.exists():
        raise ValueError('a new trace output and an uninstrumented input are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.cu', '.plan.json', '.source.json'):
        shutil.copy2(str(original) + suffix, str(output) + suffix)
    # Worktree paths reach runs/ through a symlink; compare resolved paths.
    command = [str(Path(x).resolve()) if x.endswith(('.so', '.cu')) else x
               for x in shlex.split(Path(str(original) + '.build_command.txt').read_text())]
    sources = [x for x in command if x.endswith('.cu')]
    # Some R1 libraries were compiled from a since-removed staging copy of the
    # recorded source; the copy beside the library must then match its identity.
    if command.count(str(original)) != 1 or len(sources) != 1 or (
            sources[0] != str(original) + '.cu' and
            sha(str(original) + '.cu') != identity['generated_source_sha256']):
        raise ValueError('ambiguous build command')
    command = [str(output) if x == str(original) else str(output) + '.cu'
               if x == sources[0] else x for x in command]
    command[1:1] = ['-lineinfo', '-DTILEMEGA_TRACE_STAGE=1', '-DTILEMEGA_TRACE_STEP=1']
    if a.native_phases:
        command.insert(1, '-DTILEMEGA_TRACE_NATIVE_PHASES=1')
    Path(str(output) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    with Path(str(output) + '.ptxas.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = generate(output, identity['source']['root'], str(output) + '.source.json')
    if not result['trace']:
        raise ValueError('trace identity was not recorded')
    record = dict(reference_artifact_id=identity['artifact_id'], trace_artifact_id=result['artifact_id'],
                  performance_eligible=False, native_phases=a.native_phases,
                  method='same generated program; boundary timestamps and optional native phase probes')
    Path(str(output) + '.diagnostic.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
