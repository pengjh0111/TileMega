#!/usr/bin/env python3
"""Build one native decode cell at fixed tmexec coordinates (SM90R4 P-PAIR arms)."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_inputs import extract, model_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--coordinates', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--prefill', type=Path, required=True, help='prefill for the CPU-only ABI check')
    p.add_argument('--compiler', type=Path, default=ROOT / 'build/tools/tilemega')
    a = p.parse_args()
    directory = ROOT / 'runs/sm90r4/build' / a.tag / f'{a.model}-B{a.batch}'
    directory.mkdir(parents=True, exist_ok=False)
    graph, binary = directory / 'decode.export.json', directory / 'decode.so'
    extract(a.model, 'decode', graph)
    target = ROOT / 'configs/targets/sm_90_h800.json'
    command = [str(a.compiler.resolve()), 'compile', str(graph), str(binary),
               '--tmexec-config', str(a.coordinates.resolve()), '--serving', 'decode', '--emit', 'serving',
               '--batch', str(a.batch), '--runtime-target', str(target), '--nvcc-split-compile', '2']
    with (directory / 'decode.log').open('w') as log:
        status = subprocess.call(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                 stdout=log, stderr=subprocess.STDOUT)
    if status:
        print(json.dumps(dict(status='failed', log=str(directory / 'decode.log'))), flush=True)
        return status
    status = subprocess.call([PYTHON, str(ROOT / 'scripts/check_sm90r2_native_abi.py'), '--prefill',
                              str(a.prefill.resolve()), '--decode', str(binary), '--model-config',
                              str(model_dir(a.model) / 'config.json')],
                             cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'python')),
                             stdout=(directory / 'host_abi.log').open('w'), stderr=subprocess.STDOUT)
    identity = read(str(binary) + '.identity.json')
    binding = dict(model=a.model, batch=a.batch, tag=a.tag, decode=str(binary), decode_sha256=sha(binary),
                   prefill=str(a.prefill.resolve()), coordinates=str(a.coordinates.resolve()),
                   coordinates_sha256=sha(a.coordinates), head=identity['source']['head'],
                   generated_source_sha256=identity.get('generated_source_sha256'), host_abi_status=status,
                   command=command, compiler_sha256=sha(a.compiler))
    write(directory / 'binding.json', binding)
    print(json.dumps(binding), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
