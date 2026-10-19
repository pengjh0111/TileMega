#!/usr/bin/env python3
"""Derive a diagnostic clone without changing generated plan or geometry."""
import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from sm90d1_common import ROOT, RUN, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--define', action='append', default=[])
    p.add_argument('--lineinfo', action='store_true')
    a = p.parse_args()
    original, output = a.input.resolve(), a.out.resolve()
    if not output.is_relative_to(RUN) or output == original or output.exists():
        p.error('choose a fresh output under runs/sm90d1')
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.build.identity import generate, verify
    identity = verify(original)
    if identity['trace']:
        raise ValueError('clone must start with an uninstrumented artifact')
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.cu', '.plan.json', '.source.json'):
        shutil.copy2(str(original) + suffix, str(output) + suffix)
    command = shlex.split(Path(str(original) + '.build_command.txt').read_text())
    if command.count(str(original)) != 1 or command.count(str(original) + '.cu') != 1:
        raise ValueError('ambiguous build recipe')
    command = [str(output) if v == str(original) else str(output) + '.cu'
               if v == str(original) + '.cu' else v for v in command]
    command[1:1] = (['-lineinfo'] if a.lineinfo else []) + ['-D' + v for v in a.define]
    Path(str(output) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    with Path(str(output) + '.ptxas.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = generate(output, identity['source']['root'], str(output) + '.source.json')
    write(str(output) + '.diagnostic.json', dict(parent=identity['artifact_id'],
          artifact_id=result['artifact_id'], defines=a.define, lineinfo=a.lineinfo,
          trace=result['trace'], builder_sha256=sha(__file__),
          method='identical source snapshot, generated CUDA and plan; added compilation flags only'))


if __name__ == '__main__':
    main()
