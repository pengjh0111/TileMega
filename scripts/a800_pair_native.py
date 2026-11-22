#!/usr/bin/env python3
"""Compare two SM80 artifacts with a shared opposite phase and target."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from a100_baselines import BATCHES, MODELS, ROOT, sha
from a800_baselines import RUN
from a800_native import artifact
from tilemega.bench.scheduler import write_json


def timed_requests(result):
    runs = {(row['N'], row['run']): float(row['wall_seconds'])
            for row in result['generation_runs'] if not row['warmup']}
    if (set(runs) != {(length, repeat) for length in (1, 1024) for repeat in (1, 2, 3)}
            or any(not math.isfinite(value) or value <= 0 for value in runs.values())):
        raise ValueError('expected three timed requests at each frozen decode length')
    return [[runs[length, repeat] for repeat in (1, 2, 3)] for length in (1, 1024)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', choices=BATCHES, required=True, type=int)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--control-preparation', type=Path)
    parser.add_argument('--trial-preparation', type=Path)
    shared = parser.add_mutually_exclusive_group(required=True)
    shared.add_argument('--decode', type=Path)
    shared.add_argument('--prefill', type=Path)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--trial-target', type=Path,
                        help='an updated calibration for the same physical hardware class')
    parser.add_argument('--formal-trial', action='store_true',
                        help='independently time a solver winner after five correctness processes')
    parser.add_argument('--budget-s', type=int, default=810)
    parser.add_argument('--process-timeout-s', type=int, default=125)
    args = parser.parse_args()
    if not 0 < args.process_timeout_s < args.budget_s - 111:
        parser.error('leave time for admission and cleanup within the pair budget')

    phase = 'prefill' if args.decode is not None else 'decode'
    shared_phase = 'decode' if phase == 'prefill' else 'prefill'
    shared_library = getattr(args, shared_phase)
    prompt_mode = shared_phase == 'prefill' and str(shared_library) == 'prompt'
    if prompt_mode and args.model != 'olmoe_1b_7b':
        parser.error('only OLMoE uses the decode library for prompt processing')
    if args.formal_trial and phase != 'decode':
        parser.error('formal trial timing requires a solver-selected decode artifact')
    options = ['--model', args.model, '--batch', str(args.batch),
               f'--{shared_phase}', str(shared_library),
               '--timeout-s', str(args.process_timeout_s)]
    stages = {'control': f'tune-{phase}',
              'trial': 'candidate-timing' if args.formal_trial else f'tune-{phase}'}
    targets = {'control': args.target, 'trial': args.trial_target or args.target}
    if args.trial_target:
        control, trial = (json.loads(target.read_text()) for target in targets.values())
        fields = ('name', 'memory_mib', 'capability', 'mig', 'driver')
        if (control['arch_tag'] != trial['arch_tag'] or control['resources'] != trial['resources']
                or any(control['device_provenance'][key] != trial['device_provenance'][key]
                       for key in fields)):
            parser.error('a paired trial must preserve the calibrated device class')
    libraries = {'control': artifact(args.control)[1]['binary_sha256'],
                 'trial': artifact(args.trial)[1]['binary_sha256']}
    if libraries['control'] == libraries['trial']:
        parser.error('the two variant libraries must be distinct')
    runner = [sys.executable, str(ROOT/'scripts/a800_native.py')]
    preparations = {}
    for label in libraries:
        path = getattr(args, f'{label}_preparation')
        preparations[label] = ['--host-preparation', str(path.resolve(strict=True))] if path else []
    if args.mode == 'plan':
        for label in libraries:
            library = getattr(args, label)
            subprocess.run([*runner, 'plan', *options, '--stage', stages[label], '--round', '0', f'--{phase}',
                            str(library), '--target', str(targets[label]), '--budget-s',
                            str(args.budget_s), *preparations[label]], check=True)
        return 0

    job = int(os.environ['SLURM_JOB_ID'])
    if os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800':
        parser.error('paired tuning requires the ordinary single-GPU allocation')
    deadline = time.monotonic() + args.budget_s
    if args.formal_trial:
        pair = dict(decode_sha256=libraries['trial'], target_sha256=sha(targets['trial']),
                    prefill_sha256='prompt' if prompt_mode else sha(shared_library))
        checks = [json.loads(path.read_text()) for path in (RUN/'native').glob(
            f'*/{args.model}/B{args.batch}/*/correctness-*.json')]
        accepted = [r for r in checks if r['pass'] and all(r[k] == v for k, v in pair.items())]
        if (len({r['round'] for r in accepted}) < 5 or len({r['tokens_sha256'] for r in accepted}) != 1
                or not any(r['hf_pass'] for r in accepted) or any(r['job'] == str(job) for r in accepted)):
            raise ValueError('formal paired timing needs five prior independent correctness processes and full HF')
    out = RUN/'optimization'/f'{phase}-pairs'/str(job)
    out.mkdir(parents=True, exist_ok=False)
    order = ('control', 'trial')
    write_json(out/'planned.json', dict(protocol_version='a800-single-v2', kind='paired_tuning',
        model=args.model, batch=args.batch, order=order,
        phase=phase, variant_sha256=libraries, shared_sha256='prompt' if prompt_mode else sha(shared_library),
        target_sha256=sha(args.target), targets_sha256={arm:sha(target) for arm,target in targets.items()}))
    rounds = []
    for position, label in enumerate(order):
        round_index = job * 6 + position
        remaining = int(deadline - time.monotonic())
        stage = stages[label]
        subprocess.run([*runner, 'run', *options, '--stage', stage, '--round', str(round_index),
                        f'--{phase}', str(getattr(args, label)), '--target', str(targets[label]),
                        '--budget-s', str(remaining),
                        *preparations[label]],
                       check=True, timeout=remaining)
        receipts = (RUN/'native').glob(f'*/{args.model}/B{args.batch}/*/{stage}-{round_index}.json')
        receipt, = [json.loads(path.read_text()) for path in receipts]
        result_path = Path(receipt['evidence'])/f'B{args.batch}'/'measurements.json'
        result = json.loads(result_path.read_text())
        if (not receipt['pass'] or receipt['job'] != str(job) or result['acceptance'] != 'strict'
                or result['configuration'][f'{phase}_sha256'] != libraries[label]
                or result['configuration'][f'{shared_phase}_sha256'] != (
                    'prompt mode' if prompt_mode else sha(shared_library))):
            raise ValueError('paired comparison lacks an accepted artifact-bound process')
        metric = {name: float(result[name]) for name in
                  ('ttft_seconds', 'e2e_seconds', 'tpot_seconds', 'output_tokens_per_second')}
        if any(not math.isfinite(value) or value <= 0 for value in metric.values()):
            raise ValueError('paired comparison has invalid request metrics')
        short, long = timed_requests(result)
        individual = [(end - start)/1023 for start, end in zip(short, long)]
        measured = (statistics.median(long)-statistics.median(short))/1023
        if min(individual) <= 0 or not math.isclose(measured, metric['tpot_seconds'], rel_tol=1e-7):
            raise ValueError('timing receipt differs from the frozen TPOT protocol')
        rounds.append(dict(arm=label, round=round_index, gpu_uuid=receipt['device_uuid'],
            tokens_sha256=receipt['tokens_sha256'], checkpoint_sha256=result['checkpoint']['checkpoint_sha256'],
            config_sha256=result['model_config_sha256'], prompt_sha256=result['prompt_ids_sha256'],
            n1_seconds=short, n1024_seconds=long, tpot_by_repeat_seconds=individual,
            measurement_sha256=sha(result_path), **metric))
        write_json(out/'rounds.json', rounds)
    if any(len({r[field] for r in rounds}) != 1
           for field in ('gpu_uuid', 'tokens_sha256', 'checkpoint_sha256',
                         'config_sha256', 'prompt_sha256')):
        raise ValueError('paired comparison changed GPU, output tokens, or weights')
    medians = {label: {name: statistics.median(r[name] for r in rounds if r['arm'] == label)
                       for name in metric} for label in libraries}
    control, trial = rounds
    paired_tpot = [old/new for old,new in zip(control['tpot_by_repeat_seconds'],
                                             trial['tpot_by_repeat_seconds'])]
    write_json(out/'result.json', dict(protocol_version='a800-single-v2',kind='paired_tuning',
        formal_timing=args.formal_trial, rounds=len(rounds), medians=medians,
        paired_tpot_ratios=paired_tpot,
        clears_tpot_screen=all(ratio>1 for ratio in paired_tpot)
            and trial['tpot_seconds'] < .995 * control['tpot_seconds'],
        paired_e2e_ratios=[group['control']['e2e_seconds']/group['trial']['e2e_seconds']
                           for group in ({r['arm']: r for r in rounds[i:i+2]} for i in range(0, len(rounds), 2))]))
    print(json.dumps(medians), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
