#!/usr/bin/env python3
"""Build fixed native tmexec development controls from repository inputs."""
import argparse
import json
import lzma
import os
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, PYTHON, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--coordinates', type=Path, required=True)
    p.add_argument('--prefill-library', type=Path,
                   help='explicit rebuilt prefill for the CPU-only ABI check')
    a = p.parse_args()
    c = read(a.config)
    packed = ROOT / f"configs/serving/sm90d1/inputs/{c['model']}/decode.export.json.xz"
    expected = read(ROOT / 'configs/serving/sm90d1/inputs/manifest.json')[str(packed.relative_to(ROOT))]
    if sha(packed) != expected['compressed_sha256']:
        raise ValueError('repository export input changed')
    directory = ROOT / 'runs/sm90r2/build' / a.tag / f"{c['model']}-B{c['batch']}"
    directory.mkdir(parents=True, exist_ok=True)
    graph, binary = directory / 'decode.export.json', directory / 'decode.so'
    receipt = directory / 'decode.build.json'
    if receipt.exists() or binary.exists():
        raise FileExistsError('retain prior build evidence; choose another tag')
    graph.write_bytes(lzma.decompress(packed.read_bytes()))
    if sha(graph) != expected['raw_sha256']:
        raise ValueError('decompressed export input changed')
    command = [str(a.compiler.resolve()), 'compile', str(graph), str(binary),
               '--tmexec-config', str(a.coordinates.resolve()), '--serving', 'decode',
               '--emit', 'serving', '--batch', str(c['batch']), '--runtime-target',
               str(ROOT / 'configs/targets/sm_90_h800.json'), '--nvcc-split-compile', '2']
    command = ['/usr/bin/python3', str(ROOT / 'scripts/run_sm90r1_build.py'),
               '--receipt', str(receipt), '--lock', str(directory / 'decode.lock'),
               '--timeout', '5400', '--max-gib', '48', '--', *command]
    with (directory / 'decode.log').open('w') as log:
        status = subprocess.call(command, cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                 stdout=log, stderr=subprocess.STDOUT)
    if status:
        print(json.dumps(dict(status='failed', receipt=str(receipt))), flush=True)
        return status
    reference = ROOT / 'runs/sm90d1/build/reference2' / directory.name
    prefill = a.prefill_library.resolve() if a.prefill_library else reference / 'prefill.so'
    command = [PYTHON, str(ROOT / 'scripts/check_sm90r2_native_abi.py'),
               '--prefill', str(prefill), '--decode', str(binary),
               '--model-config', str(ROOT / c['model_dir'] / 'config.json')]
    with (directory / 'decode.host_abi.json').open('w') as log:
        status = subprocess.call(command, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'python')),
            stdout=log, stderr=subprocess.STDOUT)
    if status:
        print(json.dumps(dict(status='host ABI check failed', directory=str(directory))), flush=True)
        return status
    identity = read(str(binary) + '.identity.json')
    binding = dict(config=str(a.config.resolve()), config_sha256=sha(a.config),
                   coordinates=str(a.coordinates.resolve()), coordinates_sha256=sha(a.coordinates),
                   tag=a.tag, phase='decode', binary=str(binary), binary_sha256=sha(binary),
                   head=identity['source']['head'], artifact_id=identity['artifact_id'],
                   native_origin='fixed', performance_eligible=not identity['trace'])
    write(directory / 'decode.binding.json', binding)
    print(json.dumps(binding), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
