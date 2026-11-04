#!/usr/bin/env python3
"""Compile the unchanged finalists of a retained offline Algorithm 1 receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sm90d1_common import ROOT, sha, write


def key(coordinates):
    return json.dumps(coordinates, sort_keys=True, separators=(',', ':'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--target', type=Path, default=ROOT / 'configs/targets/sm_90_h800.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split-compile', type=int, default=4)
    a = p.parse_args()
    receipt = json.loads(a.receipt.read_text())
    request = receipt['input']
    for field in ('seed', 'seed_source', 'export'):
        if sha(request[field]) != request[field+'_sha256']:
            raise ValueError('offline input changed: '+field)
    evaluated = {key(c['coordinates']) for c in receipt['evaluated']}
    finalists = receipt['finalists']
    if len(finalists) != 4 or len({key(c['coordinates']) for c in finalists}) != 4:
        raise ValueError('expected four distinct retained finalists')
    if any(key(c['coordinates']) not in evaluated for c in finalists):
        raise ValueError('finalist is absent from the retained E')
    a.out = a.out.resolve();a.out.mkdir(parents=True, exist_ok=False)
    identity = dict(receipt=str(a.receipt.resolve()), receipt_sha256=sha(a.receipt),
        compiler=str(a.compiler.resolve()), compiler_sha256=sha(a.compiler),
        target=str(a.target.resolve()), target_sha256=sha(a.target), input=request,
        scope='fixed builds of retained offline Algorithm 1 finalists; no new search or measured selection')
    write(a.out/'source.json', identity)
    candidates = []
    for rank, candidate in enumerate(finalists):
        directory = a.out / f'candidate-{rank}';directory.mkdir()
        config = directory/'coordinates.json';write(config, candidate['coordinates'])
        binary = directory/'decode.so'
        command = [str(a.compiler.resolve()), 'compile', request['export'], str(binary),
            '--tmexec-config', str(config), '--serving', 'decode', '--emit', 'serving',
            '--batch', str(request['batch']), '--runtime-target', str(a.target.resolve()),
            '--nvcc-split-compile', str(a.split_compile)]
        write(directory/'command.json', command)
        with (directory/'compile.log').open('w') as log:
            status = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        outcome_path = Path(str(binary)+'.compile-result.json')
        outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
        row = dict(rank=rank, coordinates=str(config), pool_ns=candidate['pool_ns'],
            predicted_ns=candidate['replay_ns'], structural_class=candidate['structural_class'],
            compile_result=str(outcome_path), returncode=status)
        if status:
            if outcome.get('status') != 'resource_rejected':
                write(directory/'failure.json', row)
                raise RuntimeError('unexpected compile failure; see '+str(directory/'compile.log'))
            row['status'] = 'resource_rejected'
        else:
            if outcome.get('status') != 'compiled' or outcome.get('inputs_unchanged') is not True:
                raise ValueError('compiled finalist lacks stable build-input evidence')
            built = json.loads(Path(str(binary)+'.identity.json').read_text())
            if built['binary_sha256'] != sha(binary):
                raise ValueError('compiled binary differs from its identity')
            row.update(status='compiled', binary=str(binary), binary_sha256=sha(binary))
        candidates.append(row);write(a.out/'candidates.json', candidates)
        print(json.dumps(dict(rank=rank, status=row['status'], cell=request['id'])), flush=True)
    if not any(row['status'] == 'compiled' for row in candidates):
        raise ValueError('all finalists were rejected by resource compilation')
    while True:
        directory = Path(tempfile.mkdtemp(prefix='occupancy-', dir=a.out))/'probe'
        command = [sys.executable, str(ROOT/'scripts/sm90r6_probe_native.py'),
                   '--candidates', str(a.out/'candidates.json'), '--out', str(directory)]
        with (directory.parent/'run.log').open('w') as log:
            status = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(directory.parent/'returncode.json', dict(returncode=status))
        if status == 75:
            continue
        if status:
            raise RuntimeError('occupancy query failed; see '+str(directory.parent/'run.log'))
        occupancy = json.loads((directory/'result.json').read_text())
        for resource in occupancy['candidates']:
            row = candidates[resource['rank']]
            if resource['binary_sha256'] != row['binary_sha256']:
                raise ValueError('occupancy evidence does not bind the retained finalist')
            row['occupancy'] = resource
            if resource['effective_residency'] != resource['requested_residency']:
                row['status'] = 'requires_occupancy_rescore'
        write(a.out/'candidates.json', candidates)
        needs_rescore = any(row['status'] == 'requires_occupancy_rescore' for row in candidates)
        write(a.out/'result.json', dict(candidates=candidates, occupancy=str(directory/'result.json'),
            status='requires_occupancy_rescore' if needs_rescore else 'ready_for_P_SEL',
            selection='not_measured', source=identity))
        return 3 if needs_rescore else 0


if __name__ == '__main__':
    raise SystemExit(main())
