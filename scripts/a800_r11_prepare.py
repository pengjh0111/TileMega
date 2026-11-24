#!/usr/bin/env python3
"""Prepare and bind four linked finalists for an A800 R11 cell on the login node."""
import argparse
import os
from pathlib import Path

from a100_baselines import BATCHES, sha
from a800_native import artifact
from a800_r10_prepare import MODELS
from a800_r10_release import SOLVE, invoke
from a800_r11_gate import read
from tilemega.bench.scheduler import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    for option in ('cli', 'export', 'target', 'control', 'seed', 'out'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    parser.add_argument('--cpu-budget-s', type=int, default=5400)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('prepare on the login node with CUDA devices hidden')
    control = read(args.control)
    decode, identity, manifest = artifact(Path(control['decode']))
    if control['decode_sha256'] != identity['binary_sha256']:
        raise ValueError('the frozen control artifact changed')
    for field in ('target', 'prefill'):
        if control[field] != 'prompt' and sha(control[field]) != control[f'{field}_sha256']:
            raise ValueError(f'the frozen control {field} changed')

    folder = args.out.resolve()
    output = folder/'prepared.so'
    prepared = Path(f'{output}.search')/'prepared.json'
    abi = prepared.parent/'weight-abi.json'
    scope = dict(model=args.model, batch=args.batch, top=4,
                 **{name: sha(getattr(args, name)) for name in ('cli', 'export', 'target', 'control', 'seed')})
    folder.mkdir(parents=True, exist_ok=True)
    request = folder/'preparation.json'
    if request.exists() and read(request) != scope:
        raise ValueError('start a new preparation for new inputs')
    if not request.exists():
        write_json(request, scope)
    if not abi.exists():
        invoke(SOLVE, 'prepare', ['--model', args.model, '--batch', str(args.batch),
              '--cli', str(args.cli), '--export', str(args.export), '--target', str(args.target),
              '--prefill', control['prefill'], '--seed-coordinates', str(args.seed), '--out', str(output),
              '--top', '4', '--compile-jobs', '1', '--search-budget-ms', '600000',
              '--timeout-s', str(args.cpu_budget_s)], args.cpu_budget_s+30)
    receipt = read(prepared)['receipt']
    finalists = receipt['pipeline']['finalists']
    linked = receipt['candidates']
    packing = {row['rank']: row for row in read(abi)['candidates']}
    if len(finalists) != 4 or [row['rank'] for row in linked] != list(range(4)):
        raise ValueError('R11 requires four distinct pipeline finalists')
    if len({row['structure'] for row in finalists}) < 2:
        raise ValueError('R11 finalist pool needs two structural classes')
    records = []
    for candidate, finalist in zip(linked, finalists):
        rank = candidate['rank']
        if candidate['status'] != 'compiled':
            raise ValueError(f'finalist {rank+1} needs a linked resource correction before queueing')
        binary = Path(candidate['binary'])
        proof = packing[rank]
        if (candidate['binary_sha256'] != proof['binary_sha256']
                or sha(binary) != proof['binary_sha256']
                or sha(proof['host_preparation']) != proof['host_preparation_sha256']):
            raise ValueError(f'finalist {rank+1} differs from its measured-input packing ABI')
        records.append(dict(rank=rank+1, structure=finalist['structure'],
            coordinates=finalist['coordinates'], predicted_ns=candidate['predicted_ns'],
            binary=str(binary), binary_sha256=proof['binary_sha256'],
            host_preparation=proof['host_preparation'],
            host_preparation_sha256=proof['host_preparation_sha256']))
    if len({row['binary_sha256'] for row in records}) != 4:
        raise ValueError('R11 requires four distinct linked candidates')
    incumbent = manifest['native_dense']
    new = sum(row['coordinates'] != incumbent for row in records)
    if new < 2:
        raise ValueError('R11 requires at least two new searched candidates')
    result = dict(model=args.model, batch=args.batch, control=str(args.control.resolve()),
        control_sha256=scope['control'], target=str(args.target.resolve()), target_sha256=scope['target'],
        prepared=str(prepared), prepared_sha256=sha(prepared), new_candidates=new,
        candidates=records, evaluated=receipt['evaluated'])
    binding = folder/'finalists.json'
    if binding.exists() and read(binding) != result:
        raise ValueError('retain the original finalist identities')
    write_json(binding, result)


if __name__ == '__main__':
    main()
