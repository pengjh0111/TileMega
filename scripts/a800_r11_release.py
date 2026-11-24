#!/usr/bin/env python3
"""Queue one guarded, same-GPU finalist comparison per R11 cell."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import ROOT, sha
from a800_r10_release import PAIR
from a800_r11_gate import read
from tilemega.bench.scheduler import write_json


def clean_head():
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('use the clean checkout that owns this R11 campaign')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def plan(folder, fresh_correctness):
    final = folder/'finalists.json'
    manifest = read(final)
    if (sha(manifest['control']) != manifest['control_sha256'] or sha(manifest['prepared']) != manifest['prepared_sha256']
            or sha(manifest['target']) != manifest['target_sha256']):
        raise ValueError('the frozen control or shortlist changed')
    control = read(manifest['control'])
    model, batch = manifest['model'], manifest['batch']
    limit = (1080 if model == 'qwen3_30b_a3b' else
             570 if model in ('olmoe_1b_7b', 'mistral_7b_v0_3', 'llama3_1_8b') else 420)
    candidates = manifest['candidates']
    checks = 5 if fresh_correctness else 1
    budget = (1 + len(candidates)*(checks+1))*(limit+65) + 150
    args = ['--model', model, '--batch', str(batch), '--control', control['decode'],
            '--prefill', control['prefill'], '--target', control['target'], '--trial-target', manifest['target'],
            '--expected-checkpoint-sha256', control['checkpoint_sha256'], '--numeric-gate',
            '--budget-s', str(budget), '--process-timeout-s', str(limit)]
    if fresh_correctness:
        args.append('--fresh-correctness')
    if control.get('host_preparation'):
        args.extend(('--control-preparation', control['host_preparation']))
    for row in candidates:
        rank, binary = row['rank'], row['binary']
        if sha(binary) != row['binary_sha256'] or sha(row['host_preparation']) != row['host_preparation_sha256']:
            raise ValueError('a finalist changed after preparation')
        if rank == 1:
            args.extend(('--trial', binary, '--trial-preparation', row['host_preparation']))
        else:
            args.extend(('--additional-trial', f'rank-{rank}={binary}',
                         '--arm-preparation', f'rank-{rank}={row["host_preparation"]}'))
        if row.get('target'):
            label = 'trial' if rank == 1 else f'rank-{rank}'
            if sha(row['target']) != row['target_sha256']:
                raise ValueError('a finalist calibration changed after preparation')
            args.extend(('--arm-target', f'{label}={row["target"]}'))
    route_arms = ['trial' if row['rank'] == 1 else f'rank-{row["rank"]}'
                  for row in candidates if row.get('route_check')]
    if route_arms:
        args.extend(('--routing-export', manifest['routing_export']))
        for label in route_arms:
            args.extend(('--routing-arm', label))
    head = clean_head()
    proof = subprocess.check_output([sys.executable, str(PAIR), 'plan', *args], cwd=ROOT, text=True, timeout=180)
    guard_plan = folder/'guard-inputs.jsonl'
    if guard_plan.exists() and guard_plan.read_text() != proof:
        raise ValueError('the guarded pair inputs changed')
    guard_plan.write_text(proof)
    value = dict(model=model, batch=batch, source=head, args=args, budget_s=budget,
        finalists=str(final), finalists_sha256=sha(final), guard_inputs=str(guard_plan),
        guard_inputs_sha256=sha(guard_plan))
    path = folder/'finalists.plan.json'
    if path.exists() and read(path) != value:
        raise ValueError('retain the original queued comparison')
    write_json(path, value)
    return path


def run(ledger, expected):
    if (os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800' or sha(ledger) != expected):
        raise ValueError('run the frozen ledger on the single-A800 allocation')
    row = read(ledger)['cells'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    if sha(row['plan']) != row['sha256']:
        raise ValueError('the queued cell plan changed')
    spec = read(row['plan'])
    if (spec['source'] != clean_head() or sha(spec['finalists']) != spec['finalists_sha256']
            or sha(spec['guard_inputs']) != spec['guard_inputs_sha256']):
        raise ValueError('the queued compiler artifacts or source changed')
    return subprocess.run([sys.executable, str(PAIR), 'run', *spec['args']], cwd=ROOT,
                          timeout=spec['budget_s']+20).returncode


def submit(stem, plans, parallel, launcher, host_root):
    if os.environ.get('SLURM_JOB_ID') or parallel < 1:
        raise ValueError('submit from the login node with a positive concurrency cap')
    if len({(read(path)['model'], read(path)['batch']) for path in plans}) != len(plans):
        raise ValueError('each model and batch may appear only once in the array')
    stem = stem.resolve()
    plans = [path.resolve(strict=True) for path in plans]

    def container(path):
        return str(Path('/work')/path.relative_to(host_root)) if host_root else str(path)

    ledger = Path(f'{stem}.rounds.json')
    intent = Path(f'{stem}.single.intent')
    ticket = Path(f'{stem}.id')
    value = dict(cells=[dict(plan=container(path), sha256=sha(path)) for path in plans])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists() and read(ledger) != value:
        raise ValueError('retain the original array ledger')
    if not ledger.exists():
        write_json(ledger, value)
    wall_minutes = (max(read(path)['budget_s'] for path in plans)+179)//60
    command = ['sbatch', '--parsable', '--partition=gpu_a800', '--gres=gpu:1',
               f'--array=0-{len(plans)-1}%{parallel}', f'--time={wall_minutes}',
               f'--output={stem}-%A_%a.out', *launcher, container(ledger), sha(ledger)]
    previous = dict(command=command, ledger_sha256=sha(ledger))
    if intent.exists() and read(intent) != previous:
        raise ValueError('retain the original Slurm intent')
    if not intent.exists():
        write_json(intent, previous)
    if ticket.exists():
        return ticket.read_text().strip()
    job = subprocess.check_output(command, text=True, timeout=45).split(';', 1)[0].strip()
    if not job.isdecimal():
        raise ValueError(f'invalid parsable Slurm response {job!r}')
    with ticket.open('x') as stream:
        stream.write(job+'\n')
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    freeze = modes.add_parser('plan')
    freeze.add_argument('folder', type=Path)
    freeze.add_argument('--fresh-correctness', action='store_true')
    launch = modes.add_parser('submit')
    launch.add_argument('--stem', type=Path, required=True)
    launch.add_argument('--parallel', type=int, required=True)
    launch.add_argument('--launcher', action='append', required=True,
                        help='scheduler script and optional fixed arguments, in order')
    launch.add_argument('--host-root', type=Path,
                        help='host directory mounted as /work by the scheduler launcher')
    launch.add_argument('plans', type=Path, nargs='+')
    execute = modes.add_parser('run')
    execute.add_argument('ledger', type=Path)
    execute.add_argument('sha256')
    args = parser.parse_args()
    if args.mode == 'run':
        return run(args.ledger, args.sha256)
    if args.mode == 'plan':
        print(plan(args.folder.resolve(), args.fresh_correctness))
    else:
        print(submit(args.stem, args.plans, args.parallel, args.launcher,
                     args.host_root.resolve(strict=True) if args.host_root else None))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
