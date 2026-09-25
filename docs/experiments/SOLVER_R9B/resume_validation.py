#!/usr/bin/env python3
"""Resume an interrupted CPU sample audit without replacing completed rows.

Usage: resume_validation.py llama|qwen3 [--colocate]
Stop the old child first; this runner retains its exit record and partial dump.
The C++ driver replays RNG draws, then evaluates only the missing sample IDs.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', choices=['llama', 'qwen3'])
p.add_argument('--colocate', action='store_true')
a = p.parse_args()
E = pathlib.Path(__file__).resolve().parent
ROOT = E.parents[2]
out = E / ('validation_colocated' if a.colocate else 'validation') / a.model
assert (out / 'exit.json').exists(), 'old runner must have exited'
assert json.loads((out / 'exit.json').read_text())['exit'] != 0, 'audit already completed'
assert not (out / 'resume.command.json').exists(), 'refuse to overwrite resume evidence'
raw = (out / 'samples.tsv').read_bytes()
count = len(list(csv.DictReader(raw.decode().splitlines(), delimiter='\t')))
original = json.loads((out / 'command.json').read_text())
binary = pathlib.Path('/root/r9b_work') / ('resume-' + out.parent.name + '-' + a.model)
shutil.copy2(ROOT / 'build-portable/tools/tilemega-flow-validation', binary)
command = [str(binary), *original['command'][1:], '--resume']
partial = out / ('interrupted_sample_' + str(count))
for path in out.glob(f'sample_{count}.*'):
    partial.mkdir(exist_ok=True)
    shutil.move(str(path), partial / path.name)
shutil.copy2(out / 'exit.json', out / 'interrupted.exit.json')
record = dict(command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
              prior_samples=count, prior_bytes=len(raw), prior_sha256=hashlib.sha256(raw).hexdigest())
(out / 'resume.command.json').write_text(json.dumps(record, indent=2) + '\n')
start = time.monotonic()
with (out / 'resume.log').open('x') as log:
    code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
record.update(exit=code, wall_seconds=time.monotonic() - start,
              preserved_prefix=(out / 'samples.tsv').read_bytes()[:len(raw)] == raw)
(out / 'resume.exit.json').write_text(json.dumps(record, indent=2) + '\n')
if not record['preserved_prefix']:
    raise RuntimeError('completed sample prefix changed')
(out / 'exit.json').write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(code)
