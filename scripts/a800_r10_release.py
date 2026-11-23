#!/usr/bin/env python3
"""Prepare, measure and finalize the normal pipeline shortlist before promotion."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import BATCHES, ROOT, inputs, sha
from a800_baselines import RUN
from a800_native import artifact
from a800_r10_prepare import MODELS, read
from a800_shortlist_abi import freeze_host_preparation
from tilemega.bench.scheduler import write_json


SOLVE = ROOT/'scripts/a800_solve.py'
PAIR = ROOT/'scripts/a800_pair_native.py'


def timeout(model):
    return 1080 if model == 'qwen3_30b_a3b' else 570


def invoke(script, mode, options, bound, receipt=None):
    command = [sys.executable, str(script), mode, *options]
    if receipt is not None:
        if receipt.exists():
            if subprocess.check_output(command, cwd=ROOT, timeout=bound) != receipt.read_bytes():
                raise ValueError(f'queued inputs changed since {receipt}')
            return
        evidence = subprocess.check_output(command, cwd=ROOT, timeout=bound)
        pending = receipt.with_suffix('.tmp')
        pending.write_bytes(evidence)
        pending.replace(receipt)
    else:
        subprocess.run(command, cwd=ROOT, timeout=bound, check=True)


def bind_shortlist(folder, options, control, prepared):
    candidates = read(prepared)['receipt']['candidates']
    if sum(row['status'] == 'compiled' for row in candidates) < 2:
        raise ValueError('normal measured selection needs two prepared finalists')
    model, batch = control['model'], control['batch']
    suite_budget = 2*timeout(model)+480
    measure_output = folder/'measured.so'
    request = [*options, '--out', str(measure_output), '--timeout-s', str(suite_budget),
               '--prepared-shortlist', str(prepared), '--defer-finalize']
    receipt = folder/'selection.plan.json'
    invoke(SOLVE, 'plan', request, 120, receipt)
    spec = dict(model=model, batch=batch, control=control, solve=options,
                shortlist=str(prepared), select=request, receipt_sha256=sha(receipt),
                select_budget_s=suite_budget, selection_results=str(folder/'measured.so.search'))
    metadata = folder/'selection.binding.json'
    if metadata.exists() and read(metadata) != spec:
        raise ValueError('retain the original R10 selection binding')
    write_json(metadata, spec)


def prepare(args):
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('prepare the normal shortlist on the login node with CUDA devices hidden')
    cell = f'{args.model}/B{args.batch}'
    selection = read(args.selection)['cells'][cell]
    seed = Path(selection['seed'])
    if sha(seed) != selection['seed_sha256']:
        raise ValueError('the measured seed changed before shortlist preparation')
    control = read(args.inventory)['cells'][cell]
    if (sha(control['timing_source']) != control['timing_sha256']
            or (control['model'], control['batch']) != (args.model, args.batch)):
        raise ValueError('the archived R9 control changed')
    args.out.mkdir(parents=True, exist_ok=True)
    options = ['--model', args.model, '--batch', str(args.batch), '--export', str(args.export),
               '--target', control['target'], '--prefill', control['prefill'], '--cli', str(args.cli),
               '--compile-jobs', '1', '--top', '2',
               '--search-budget-ms', '600000', '--seed-coordinates', str(seed)]
    search = args.out/'prepared.so.search'
    cpu_budget = 3000
    if not (search/'weight-abi.json').exists():
        invoke(SOLVE, 'prepare', [*options, '--out', str(args.out/'prepared.so'),
                                 '--timeout-s', str(cpu_budget)], cpu_budget+30)
    bind_shortlist(args.out, options, control, search/'prepared.json')


def measure(folder, binding):
    plan = folder/'selection.plan.json'
    if sha(plan) != binding['receipt_sha256']:
        raise ValueError('the queued shortlist plan changed')
    job = os.environ['SLURM_JOB_ID']
    run = RUN/'search'/job
    archive = folder/f'selection-{job}'
    if run.exists() or archive.exists():
        raise ValueError('retain the previous guard evidence for this allocation')
    try:
        invoke(SOLVE, 'run', [*binding['select'], '--submission-plan', str(plan)],
               binding['select_budget_s']+30)
    finally:
        if run.exists():
            run.rename(archive)


def guarded_selection(folder, job, exit_code):
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('process GPU selection receipts on the login node')
    if not job.isdecimal():
        raise ValueError('provide the numeric allocation ID recorded by Slurm')
    guard = folder/f'selection-{job}'
    result = read(guard/'guard_result.json')
    device = read(guard/'job/physical-device-identity.json')
    plan = read(folder/'selection.plan.json')
    observed = read(guard/'planned.json')
    acceptance = 'strict' if exit_code == 0 else 'rejected'
    if (result['code'] != exit_code or result['acceptance'] != acceptance
            or result['reason'] != f'child exit {exit_code}' or not result['ownership_verified']
            or device['compute_capability'] != [8, 0]
            or observed.pop('device')['uuid'] != device['nvml_uuid'] or observed != plan):
        raise ValueError('selection did not pass its frozen single-A800 guard')
    return device


def reprepare(folder, binding, job, output):
    device = guarded_selection(folder, job, 75)
    proof = Path(binding['selection_results'])/'needs-reprepare.json'
    receipt = read(proof)
    original = Path(binding['shortlist'])
    if (receipt['prepared_sha256'] != sha(original) or receipt['status'] != 'needs_reprepare'
            or not receipt['corrections']):
        raise ValueError('occupancy corrections do not bind the guarded shortlist')
    output.mkdir(parents=True, exist_ok=True)
    prepared = output/'prepared.so.search/prepared.json'
    if not prepared.with_name('weight-abi.json').exists():
        invoke(SOLVE, 'reprepare', [*binding['solve'], '--out', str(output/'prepared.so'),
                                    '--prepared-shortlist', str(original), '--reprepare-receipt', str(proof),
                                    '--timeout-s', '1800'], 1830)
    bind_shortlist(output, binding['solve'], binding['control'], prepared)
    write_json(output/'occupancy-binding.json', dict(job=job, gpu_uuid=device['nvml_uuid'],
                                                     correction_sha256=sha(proof)))


def finalize(folder, binding, job):
    device = guarded_selection(folder, job, 0)
    search = Path(binding['selection_results'])
    proof = search/'ready-to-finalize.json'
    prepared = Path(binding['shortlist'])
    selection = read(proof)
    timing = selection['receipt']['measurement']
    if (selection['prepared_sha256'] != sha(prepared) or timing.get('split_rank_measurement')
            or timing['gpu_uuid'] != device['nvml_uuid'] or timing.get('independent_allocations')
            or timing.get('prior_measurements')
            or timing['checkpoint_sha256'] != binding['control']['checkpoint_sha256']
            or len(timing['arms']) < 2 or any(row['processes'] != 1 for row in timing['arms'].values())):
        raise ValueError('normal finalists did not use one shared guarded allocation and checkpoint')
    library = folder/'winner.so'
    if not library.exists():
        invoke(SOLVE, 'finalize', [*binding['solve'], '--out', str(library), '--prepared-shortlist', str(prepared),
                                 '--finalize-receipt', str(proof), '--timeout-s', '1800'], 1830)
    _, identity, manifest = artifact(library)
    rebuild = read(folder/'winner.so.rebuild/comparison.json')
    if (manifest['native_origin'] != 'solver' or manifest['native_selection']['selection'] != 'measure'
            or not rebuild['pass'] or rebuild['reference_sha256'] != timing['winner_binary_sha256']
            or rebuild['candidate_sha256'] != identity['binary_sha256']):
        raise ValueError('formal promotion requires the normal measured solver artifact')
    control = binding['control']
    if sha(control['timing_source']) != control['timing_sha256']:
        raise ValueError('the archived control timing changed before formal comparison')
    raw = read(Path(read(control['timing_source'])['evidence'])/f"B{binding['batch']}/measurements.json")
    baseline = Path(raw['configuration']['decode_binary'])
    if sha(baseline) != control['decode_sha256']:
        raise ValueError('formal R9 control does not match its published binary')
    prefill = control['prefill']
    prefix = () if prefill == 'prompt' else (Path(prefill),)
    config, _ = inputs(binding['model'])
    checkpoint = read(config/'config.json')
    hosts = [folder/'control-host.json', folder/'trial-host.json']
    for path, decode in zip(hosts, (baseline, library)):
        if not path.exists():
            freeze_host_preparation(checkpoint, binding['batch'], (*prefix, decode), path)
    pair = ['--model', binding['model'], '--batch', str(binding['batch']), '--target', control['target'],
            '--prefill', prefill, '--control', str(baseline), '--trial', str(library),
            '--control-preparation', str(hosts[0]), '--trial-preparation', str(hosts[1]),
            '--expected-checkpoint-sha256', control['checkpoint_sha256'], '--formal-trial',
            '--fresh-correctness', '--process-timeout-s', str(timeout(binding['model']))]
    budget = 7*(timeout(binding['model'])+111)+180
    receipt = folder/'formal.plan.json'
    invoke(PAIR, 'plan', [*pair, '--budget-s', str(budget)], 180, receipt)
    write_json(folder/'formal.binding.json', dict(pair=pair, budget_s=budget, plan_sha256=sha(receipt),
               solve_job=job, solve_proof_sha256=sha(proof)))


def formal(folder):
    binding = read(folder/'formal.binding.json')
    receipt = folder/'formal.plan.json'
    if sha(receipt) != binding['plan_sha256']:
        raise ValueError('formal promotion plan changed after preparation')
    budget = binding['budget_s']
    options = [*binding['pair'], '--budget-s', str(budget)]
    invoke(PAIR, 'plan', options, 180, receipt)
    job = os.environ['SLURM_JOB_ID']
    run = RUN/'optimization/decode-pairs'/job
    try:
        invoke(PAIR, 'run', options, budget+20)
    finally:
        if run.exists():
            run.rename(folder/f'formal-{job}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    make = sub.add_parser('prepare')
    make.add_argument('--model', choices=MODELS, required=True)
    make.add_argument('--batch', choices=BATCHES, type=int, required=True)
    for name in ('selection', 'inventory', 'export', 'cli', 'out'):
        make.add_argument(f'--{name}', type=Path, required=True)
    for mode in ('measure', 'reprepare', 'finalize', 'formal'):
        run = sub.add_parser(mode)
        run.add_argument('--folder', type=Path, required=True)
        if mode in ('reprepare', 'finalize'):
            run.add_argument('--job', required=True)
        if mode == 'reprepare':
            run.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'formal':
        formal(args.folder)
    else:
        binding = read(args.folder/'selection.binding.json')
        if args.mode == 'measure':
            measure(args.folder, binding)
        elif args.mode == 'reprepare':
            reprepare(args.folder, binding, args.job, args.out)
        else:
            finalize(args.folder, binding, args.job)


if __name__ == '__main__':
    main()
