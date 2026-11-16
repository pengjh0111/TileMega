#!/usr/bin/env python3
"""Guarded occupancy queries on the linked cubins of native finalists."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sm90d1_common import ROOT, sha, write
import sm90r5_dispatch as dispatch


def worker(prepared, out):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('occupancy queries require the physical-card guard')
    rows = []
    for candidate in json.loads(prepared.read_text()):
        run = subprocess.run(candidate['command'], capture_output=True, text=True)
        folder = Path(candidate['directory'])
        (folder / 'stdout.txt').write_text(run.stdout)
        (folder / 'stderr.txt').write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(f'occupancy query failed: {folder}')
        values = json.loads(run.stdout)
        if values['registers'] != candidate['linked_registers']:
            raise ValueError('CUDA driver register allocation differs from the linked identity')
        if values['sms'] * candidate['requested_residency'] != values['grid']:
            raise ValueError('query device differs from the compiled target grid')
        values.update({k: candidate[k] for k in ['rank', 'binary_sha256', 'identity_sha256',
            'cubin_sha256', 'ptxas_sha256', 'requested_residency', 'linked_registers', 'ptxas_registers']})
        values.update(gpu=int(os.environ['TILEMEGA_DEVICE_INDEX']),
                      effective_residency=min(2, values['supported_residency']), evidence='verified')
        rows.append(values)
        write(folder / 'result.json', values)
    write(out, dict(candidates=rows, timing=False, evidence='verified CUDA driver occupancy'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', default='auto')
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        worker(a.worker, a.out)
        return 0
    if not a.candidates:
        p.error('--candidates is required')
    if a.gpu not in ('auto', '0', '1', '2', '3', '4', '5', '7'):
        p.error('--gpu must be auto or one of 0-5,7')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe = out / 'probe'
    command = ['c++', '-std=c++17', '-O2', '-I/usr/local/cuda/include',
        str(ROOT / 'scripts/sm90r6_occupancy_probe.cpp'), '-L/usr/local/cuda/lib64/stubs',
        '-lcuda', '-o', str(probe)]
    write(out / 'build-command.json', command)
    with (out / 'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    prepared = []
    for row in json.loads(a.candidates.read_text()):
        if row['status'] != 'compiled':
            continue
        binary = Path(row['binary']).resolve()
        identity_path = Path(str(binary) + '.identity.json')
        identity = json.loads(identity_path.read_text())
        manifest = json.loads(Path(str(binary) + '.plan.json').read_text())
        if sha(binary) != row['binary_sha256'] or sha(binary) != identity['binary_sha256']:
            raise ValueError('candidate binary differs from its compiled identity')
        kernels = [name for name in identity['resources'] if 'sm90_dense_native_kernel' in name]
        if len(kernels) != 1:
            raise ValueError('native occupancy requires one linked native kernel')
        directory = out / f'candidate-{row["rank"]}'
        directory.mkdir()
        with (directory / 'extract.log').open('w') as log:
            subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--extract-elf', 'all', str(binary)],
                cwd=directory, check=True, stdout=log, stderr=subprocess.STDOUT)
        cubins = list(directory.glob('*.sm_90a.cubin'))
        if len(cubins) != 1:
            raise ValueError('expected one linked sm_90a cubin')
        coordinate = manifest['native_dense']
        prepared.append(dict(rank=row['rank'], directory=str(directory),
            binary_sha256=sha(binary), identity_sha256=sha(identity_path), cubin_sha256=sha(cubins[0]),
            ptxas_sha256=sha(Path(str(binary) + '.ptxas.log')),
            linked_registers=identity['resources'][kernels[0]]['registers'],
            ptxas_registers=identity['ptxas_resources'][kernels[0]]['registers'],
            requested_residency=coordinate['residency'],
            command=[str(probe), str(cubins[0]), kernels[0], str(identity['shared_memory_bytes']),
                     '160', str(manifest['grid']), str(coordinate['cluster'])]))
    if not prepared:
        raise ValueError('no compiled native candidates to query')
    write(out / 'prepared.json', prepared)
    collect = [sys.executable, str(Path(__file__).resolve()), '--worker', str(out / 'prepared.json'),
               '--out', str(out / 'result.json')]
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') == '1':
        subprocess.run(collect, check=True)
        return 0
    gpu, claim = dispatch.claim('trial', 'r6-native-occupancy',
                                only=None if a.gpu == 'auto' else int(a.gpu))
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
            TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_TIMING='0',
            TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)), TILEMEGA_GPU_PID_MODE='namespace_limited',
            TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', PYTHONPATH=str(ROOT / 'python'))
        for key in ['TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE']:
            env.pop(key, None)
        command = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(dispatch.POLICY),
            '--out', str(out / 'guard'), '--needs-free-mib', '2048', '--timeout-s', '300', '--',
            'numactl', f'--cpunodebind={dispatch.numa(gpu)}', *collect]
        write(out / 'command.json', dict(command=command, gpu=gpu))
        with (out / 'run.log').open('w') as log:
            code = subprocess.call(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            return code
    finally:
        claim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
