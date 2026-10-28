#!/usr/bin/env python3
"""Solve one dense cell with the native solver and P-SEL (prompt §4 L0-4)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_inputs import extract, model_dir

TARGET = ROOT / 'configs/targets/sm_90_h800.json'


def export(model, directory):
    return extract(model, 'decode', directory / 'decode.export.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--prefill', type=Path, help='defaults to the REF-R2 prefill of this cell')
    p.add_argument('--compiler', type=Path, default=ROOT / 'build/tools/tilemega')
    p.add_argument('--concurrency', default='serial', choices=('serial', 'trial', 'verified'))
    p.add_argument('--solve-root', default='runs/sm90r3/solve')
    a = p.parse_args()
    cell = f'{a.model}-B{a.batch}'
    directory = ROOT / a.solve_root / a.tag / cell
    directory.mkdir(parents=True, exist_ok=False)
    prefill = (a.prefill or ROOT / f'runs/sm90r2/build/closeout-dense-fixed/{cell}/prefill.so').resolve()
    graph = export(a.model, directory)
    measure = shlex.join([PYTHON, str(ROOT / 'scripts/sm90r3_select.py'), '--model', a.model,
                          '--batch', str(a.batch), '--prefill', str(prefill), '--gpu', str(a.gpu),
                          '--name', f'sel-{a.tag}-{cell}', '--concurrency', a.concurrency])
    binary = directory / 'decode.so'
    command = [str(a.compiler.resolve()), 'compile', str(graph), str(binary), '--solve', str(TARGET),
               '--serving', 'decode', '--emit', 'serving', '--batch', str(a.batch),
               '--runtime-target', str(TARGET), '--selection', 'measure', '--measure-top', '4',
               '--measure-cmd', measure, '--nvcc-split-compile', '2']
    write(directory / 'solve-command.json', dict(command=command, compiler_sha256=sha(a.compiler),
                                                prefill=str(prefill), prefill_sha256=sha(prefill)))
    with (directory / 'solve.log').open('w') as log:
        status = subprocess.call(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                 stdout=log, stderr=subprocess.STDOUT)
    if status:
        print(json.dumps(dict(cell=cell, status='solve failed', log=str(directory / 'solve.log'))))
        return status
    abi = subprocess.call([PYTHON, str(ROOT / 'scripts/check_sm90r2_native_abi.py'), '--prefill', str(prefill),
                           '--decode', str(binary), '--model-config',
                           str(model_dir(a.model) / 'config.json')],
                          cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'python')),
                          stdout=(directory / 'host_abi.log').open('w'), stderr=subprocess.STDOUT)
    identity = read(str(binary) + '.identity.json')
    receipt = read(directory / 'decode.so.search/receipt.json')
    binding = dict(cell=cell, tag=a.tag, decode=str(binary), decode_sha256=sha(binary), prefill=str(prefill),
                   prefill_sha256=sha(prefill), head=identity['source']['head'], host_abi_status=abi,
                   coordinates=str(binary) + '.tmexec.json', winner=receipt['winner'],
                   predicted_ns=[r['prediction']['step_ns'] for r in receipt['ranked']])
    write(directory / 'binding.json', binding)
    print(json.dumps(binding))
    return abi


if __name__ == '__main__':
    sys.exit(main())
