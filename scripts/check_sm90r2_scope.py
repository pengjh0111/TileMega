#!/usr/bin/env python3
"""Generate the four fixed non-decode controls and compare their source bytes."""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/sm90r2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--compare', type=Path)
    a = p.parse_args()
    compiler = a.compiler.resolve()
    manifest = json.loads((ROOT / 'configs/serving/sm90r2/scope_inputs/manifest.json').read_text())
    directory = RUN / 'scope' / a.tag
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / 'inputs'
    inputs.mkdir()

    def unpack(name):
        path = ROOT / name
        if path.suffix != '.xz':
            return name
        data = path.read_bytes()
        raw = lzma.decompress(data)
        if name in manifest['inputs']:
            expected = manifest['inputs'][name]
            if sha(data) != expected['compressed_sha256'] or sha(raw) != expected['raw_sha256']:
                raise ValueError(f'input identity changed: {name}')
        destination = inputs / (sha(raw)[:12] + '-' + path.name[:-3])
        destination.write_bytes(raw)
        return str(destination)

    rows = []
    previous = json.loads(a.compare.read_text()) if a.compare else None
    for control in manifest['controls']:
        output = directory / (control['name'] + '.cu')
        command = [str(compiler), 'compile', unpack(control['graph']), str(output)]
        command += [unpack(x) for x in control['options']]
        with output.with_suffix('.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        row = dict(name=control['name'], command=command, returncode=result.returncode)
        if result.returncode == 0:
            row.update(source=str(output), sha256=sha(output.read_bytes()))
            if previous:
                baseline = next(x for x in previous['controls'] if x['name'] == row['name'])
                row['identical'] = output.read_bytes() == Path(baseline['source']).read_bytes()
        rows.append(row)
        print(json.dumps(row), flush=True)
    receipt = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   compiler_sha256=sha(compiler.read_bytes()), controls=rows)
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if any(x['returncode'] or x.get('identical') is False for x in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
