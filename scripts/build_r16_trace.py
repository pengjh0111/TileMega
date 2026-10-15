#!/usr/bin/env python3
"""Derive an explicitly ineligible diagnostic binary from a frozen plan."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original, output = args.input.resolve(), args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root/'runs/r16') or output == original:
        parser.error('diagnostic output must be a separate artifact under runs/r16')
    sys.path.insert(0, str(root/'python'))
    from tilemega.build.identity import generate, verify
    identity = verify(original)
    if identity['trace']:
        raise ValueError('input must be an uninstrumented reference artifact')
    if output.exists() or Path(str(output)+'.identity.json').exists():
        raise FileExistsError('retain prior diagnostic artifacts; choose a new output')
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.cu', '.plan.json', '.source.json'):
        shutil.copy2(str(original)+suffix, str(output)+suffix)
    command = shlex.split(Path(str(original)+'.build_command.txt').read_text())
    if command.count(str(original)) != 1 or command.count(str(original)+'.cu') != 1:
        raise ValueError('build command must identify exactly one input and output')
    command = [str(output) if word == str(original) else
               str(output)+'.cu' if word == str(original)+'.cu' else word
               for word in command]
    command[1:1] = ['-lineinfo', '-DTILEMEGA_TRACE_TASK=1',
                    '-DTILEMEGA_TRACE_STAGE=1', '-DTILEMEGA_TRACE_STEP=1',
                    '-DTILEMEGA_TRACE_V2=1']
    Path(str(output)+'.build_command.txt').write_text(shlex.join(command)+'\n')
    with Path(str(output)+'.ptxas.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = generate(output, identity['source']['root'], str(output)+'.source.json')
    if not result['trace']:
        raise ValueError('diagnostic artifact is missing its trace identity')
    record = dict(evidence='verified', reference_artifact_id=identity['artifact_id'],
                  trace_artifact_id=result['artifact_id'], performance_eligible=False,
                  method='same source snapshot, generated plan and geometry; trace macros only',
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(str(output)+'.diagnostic.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
