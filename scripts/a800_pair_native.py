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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', choices=BATCHES, required=True, type=int)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--trial', type=Path, required=True)
    shared = parser.add_mutually_exclusive_group(required=True)
    shared.add_argument('--decode', type=Path)
    shared.add_argument('--prefill', type=Path)
    parser.add_argument('--target', type=Path, required=True)
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
    stage = f'tune-{phase}'
    options = ['--model', args.model, '--batch', str(args.batch),
               '--target', str(args.target), f'--{shared_phase}', str(shared_library),
               '--stage', stage, '--timeout-s', str(args.process_timeout_s)]
    libraries = {'control': artifact(args.control)[1]['binary_sha256'],
                 'trial': artifact(args.trial)[1]['binary_sha256']}
    if libraries['control'] == libraries['trial']:
        parser.error('the two variant libraries must be distinct')
    runner = [sys.executable, str(ROOT/'scripts/a800_native.py')]
    if args.mode == 'plan':
        for library in (args.control, args.trial):
            subprocess.run([*runner, 'plan', *options, '--round', '0', f'--{phase}',
                            str(library), '--budget-s', str(args.budget_s)], check=True)
        return 0

    job = int(os.environ['SLURM_JOB_ID'])
    if os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800':
        parser.error('paired tuning requires the ordinary single-GPU allocation')
    deadline = time.monotonic() + args.budget_s
    out = RUN/'optimization'/f'{phase}-pairs'/str(job)
    out.mkdir(parents=True, exist_ok=False)
    order = ('control', 'trial')
    write_json(out/'planned.json', dict(model=args.model, batch=args.batch, order=order,
        phase=phase, variant_sha256=libraries, shared_sha256='prompt' if prompt_mode else sha(shared_library), target_sha256=sha(args.target)))
    rounds = []
    for position, label in enumerate(order):
        round_index = job * 6 + position
        remaining = int(deadline - time.monotonic())
        subprocess.run([*runner, 'run', *options, '--round', str(round_index),
                        f'--{phase}', str(getattr(args, label)), '--budget-s', str(remaining)],
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
        rounds.append(dict(arm=label, round=round_index, gpu_uuid=receipt['device_uuid'],
            tokens_sha256=receipt['tokens_sha256'], checkpoint_sha256=result['checkpoint']['checkpoint_sha256'],
            measurement_sha256=sha(result_path), **metric))
        write_json(out/'rounds.json', rounds)
    if any(len({r[field] for r in rounds}) != 1
           for field in ('gpu_uuid', 'tokens_sha256', 'checkpoint_sha256')):
        raise ValueError('paired comparison changed GPU, output tokens, or weights')
    medians = {label: {name: statistics.median(r[name] for r in rounds if r['arm'] == label)
                       for name in metric} for label in libraries}
    write_json(out/'result.json', dict(formal_timing=False, rounds=len(rounds), medians=medians,
        paired_e2e_ratios=[group['control']['e2e_seconds']/group['trial']['e2e_seconds']
                           for group in ({r['arm']: r for r in rounds[i:i+2]} for i in range(0, len(rounds), 2))]))
    print(json.dumps(medians), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
