#!/usr/bin/env python3
"""Compare two SM80 artifacts with a shared opposite phase and target."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

from a100_baselines import BATCHES, MODELS, ROOT, sha
from a800_baselines import A800_PARTITIONS, RUN
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
    parser.add_argument('--additional-trial', action='append', default=[], metavar='LABEL=LIBRARY')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--control-preparation', type=Path)
    parser.add_argument('--trial-preparation', type=Path)
    parser.add_argument('--arm-preparation', action='append', default=[], metavar='LABEL=RECEIPT')
    parser.add_argument('--private-stream', action='store_true')
    parser.add_argument('--routing-export', type=Path,
                        help='verify identical MoE routes before measuring fixed router coordinates')
    parser.add_argument('--routing-arm', action='append', default=[],
                        help='verify ordered routes for this named R11 arm using the control trajectory')
    shared = parser.add_mutually_exclusive_group(required=True)
    shared.add_argument('--decode', type=Path)
    shared.add_argument('--prefill', type=Path)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--trial-target', type=Path,
                        help='an updated calibration for the same physical hardware class')
    parser.add_argument('--arm-target', action='append', default=[], metavar='LABEL=TARGET',
                        help='bind a finalist built against a different A800 calibration')
    parser.add_argument('--formal-trial', action='store_true',
                        help='independently time a solver winner after five correctness processes')
    parser.add_argument('--fresh-correctness', action='store_true',
                        help='run five new correctness processes in this allocation before timing')
    parser.add_argument('--numeric-gate', action='store_true',
                        help='require C1/C2 on each trial before comparative timing')
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
    variants = {'control': args.control, 'trial': args.trial}
    for spec in args.additional_trial:
        label, equal, path = spec.partition('=')
        if not equal or not path or not re.fullmatch(r'[a-z0-9_-]+', label) or label in variants:
            parser.error('additional trials require distinct labels and existing libraries')
        variants[label] = Path(path)
    if len(variants) > 16 or (len(variants) > 2 and not args.numeric_gate and
                             (args.formal_trial or args.trial_target or args.trial_preparation)):
        parser.error('multi-arm trials share one target and use independent screening')
    if args.fresh_correctness and not (args.formal_trial or args.numeric_gate):
        parser.error('fresh correctness belongs to a formal paired trial')
    if args.numeric_gate and (phase != 'decode' or args.formal_trial):
        parser.error('the numeric gate is an independent paired decode protocol')
    if args.formal_trial and phase != 'decode':
        parser.error('formal trial timing requires a solver-selected decode artifact')
    if args.routing_export and (phase != 'decode' or args.formal_trial or
                                args.model not in ('olmoe_1b_7b', 'qwen3_30b_a3b')):
        parser.error('route bit comparison is a fixed MoE decode experiment')
    if args.routing_arm and (not args.numeric_gate or not args.routing_export
                            or not set(args.routing_arm) <= variants.keys()-{'control'}):
        parser.error('routing-arm requires the numeric gate, export, and declared trial labels')
    route_arms = {'control', *(args.routing_arm or (variants.keys()-{'control'}))} if args.routing_export else set()
    options = ['--model', args.model, '--batch', str(args.batch),
               f'--{shared_phase}', str(shared_library),
               '--timeout-s', str(args.process_timeout_s)]
    if args.private_stream:
        options.append('--private-stream')
    if args.expected_checkpoint_sha256:
        options.extend(('--expected-checkpoint-sha256', args.expected_checkpoint_sha256))
    stages = {'control': f'tune-{phase}',
              'trial': 'candidate-timing' if args.formal_trial else f'tune-{phase}'}
    stages.update((label, f'tune-{phase}') for label in variants if label not in stages)
    targets = {label: args.trial_target or args.target for label in variants}
    targets['control'] = args.target
    overrides = set()
    for specification in args.arm_target:
        label, equal, path = specification.partition('=')
        if (not args.numeric_gate or not equal or label not in variants or label == 'control'
                or not path or label in overrides):
            parser.error('arm-target requires a unique declared numeric-gate trial')
        targets[label] = Path(path)
        overrides.add(label)
    if set(targets.values()) != {args.target}:
        control = json.loads(args.target.read_text())
        fields = ('name', 'memory_mib', 'capability', 'mig', 'driver')
        for target in set(targets.values()) - {args.target}:
            trial = json.loads(target.read_text())
            if (control['arch_tag'] != trial['arch_tag'] or control['resources'] != trial['resources']
                    or any(control['device_provenance'][key] != trial['device_provenance'][key]
                           for key in fields)):
                parser.error('a paired trial must preserve the calibrated device class')
    libraries = {label: artifact(path)[1]['binary_sha256'] for label, path in variants.items()}
    if len(set(libraries.values())) != len(variants):
        parser.error('the variant libraries must be distinct')
    runner = [sys.executable, str(ROOT/'scripts/a800_native.py')]
    preparations = {}
    for label in libraries:
        path = getattr(args, f'{label}_preparation', None)
        preparations[label] = ['--host-preparation', str(path.resolve(strict=True))] if path else []
    for spec in args.arm_preparation:
        label,equal,path = spec.partition('=')
        if not args.numeric_gate or not equal or label not in preparations or preparations[label]:
            parser.error('provide one host preparation for a declared numeric-gate arm')
        preparations[label] = ['--host-preparation',str(Path(path).resolve(strict=True))]
    def correctness(index, budget, run, teacher_forced, label='trial', route_tokens=None):
        command = [*runner, 'run' if run else 'plan', *options, '--stage', 'correctness',
                   '--round', str(index), '--decode', str(variants[label]),
                   '--target', str(targets[label]), '--budget-s', str(budget), *preparations[label]]
        if teacher_forced:
            command.append('--hf-reference')
        if args.numeric_gate:
            command.append('--numeric-gate')
        if label in route_arms:
            command.extend(('--routing-export', str(args.routing_export.resolve(strict=True))))
        if route_tokens:
            command.extend(('--routing-tokens', str(route_tokens)))
        subprocess.run(command, check=True, timeout=budget)

    if args.mode == 'plan':
        if args.fresh_correctness and not args.numeric_gate:
            for round_index in range(1_000_008, 1_000_013):
                correctness(round_index, args.budget_s, False, round_index == 1_000_008)
        if args.numeric_gate or args.routing_export:
            for offset, label in enumerate(variants):
                if label != 'control' or args.routing_export:
                    correctness(1_000_016 + offset, args.budget_s, False, label != 'control', label)
        for label in libraries:
            library = variants[label]
            subprocess.run([*runner, 'plan', *options, '--stage', stages[label], '--round', '0', f'--{phase}',
                            str(library), '--target', str(targets[label]), '--budget-s',
                            str(args.budget_s), *preparations[label]], check=True)
        return 0

    job = int(os.environ['SLURM_JOB_ID'])
    partition = os.environ.get('SLURM_JOB_PARTITION')
    if partition not in A800_PARTITIONS:
        parser.error('paired tuning requires the ordinary single-GPU allocation')
    deadline = time.monotonic() + args.budget_s
    if args.fresh_correctness and not args.numeric_gate:
        for repeat in range(5):
            correctness(job * 16 + 8 + repeat, max(1, int(deadline - time.monotonic())),
                        True, repeat == 0)
    if args.formal_trial:
        pair = dict(decode_sha256=libraries['trial'], target_sha256=sha(targets['trial']),
                    prefill_sha256='prompt' if prompt_mode else sha(shared_library))
        checks = [json.loads(path.read_text()) for path in (RUN/'native').glob(
            f'*/{args.model}/B{args.batch}/*/correctness-*.json')]
        accepted = [r for r in checks if r['pass'] and all(r[k] == v for k, v in pair.items())
                    and (not args.fresh_correctness or str(r['job']) == str(job))]
        if (len({r['round'] for r in accepted}) < 5 or len({r['tokens_sha256'] for r in accepted}) != 1
                or not any(r['hf_pass'] for r in accepted)
                or (not args.fresh_correctness and any(str(r['job']) == str(job) for r in accepted))):
            raise ValueError('formal paired timing needs five independent correctness processes and full HF')
    out = RUN/'optimization'/f'{phase}-pairs'/str(job)
    out.mkdir(parents=True, exist_ok=False)
    order = tuple(variants)
    plan = dict(protocol_version='a800-single-v2', kind='paired_tuning',
        model=args.model, batch=args.batch, order=order, partition=partition,
        phase=phase, variant_sha256=libraries, shared_sha256='prompt' if prompt_mode else sha(shared_library),
        target_sha256=sha(args.target), targets_sha256={arm:sha(target) for arm,target in targets.items()})
    if args.numeric_gate:
        plan.update(numeric_gate='sm90-c1-c2-v1', correctness_processes=5 if args.fresh_correctness else 1)
        if route_arms:
            plan.update(routing_arms=sorted(route_arms), routing_export_sha256=sha(args.routing_export))
    write_json(out/'planned.json', plan)
    rounds = []
    rejected = {}
    numeric_receipts = {}
    numeric_proofs = {}
    if args.numeric_gate:
        from a800_r11_gate import INVARIANTS, verify

        expected_routes = None
        routing_tokens = None
        for position,label in enumerate(order):
            if label == 'control' and not args.routing_export:
                continue
            count = 1 if label == 'control' else plan['correctness_processes']
            proofs, reasons = [], []
            for offset in range(count):
                repeat = job*128+32+position*5+offset
                try:
                    trajectory = routing_tokens if label in route_arms else None
                    correctness(repeat,max(1,int(deadline-time.monotonic())),True,True,label,trajectory)
                    [path] = list((RUN/'native').glob(f'*/{args.model}/B{args.batch}/*/correctness-{repeat}.json'))
                    proof = verify(path,job=job,model=args.model,batch=args.batch,identity={
                        'decode_sha256': libraries[label], 'prefill_sha256': plan['shared_sha256'],
                        'target_sha256': plan['targets_sha256'][label]})
                    if label in route_arms:
                        route = proof['receipt']
                        actual = route['router_indices_sha256'], route['routing_trajectory_sha256']
                        if label == 'control':
                            expected_routes = actual
                            routing_tokens = Path(route['evidence'])/'reference_tokens.json'
                        elif actual != expected_routes:
                            raise ValueError('router trial changed ordered expert indices')
                    proofs.append(proof)
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    reasons.append({'round':repeat,'reason':str(error)})
            numeric_receipts[label] = dict(required=count,passed=len(proofs),
                proofs=[{key:row[key] for key in ('path','sha256')} for row in proofs],rejected=reasons)
            numeric_proofs[label] = proofs
            if (len(proofs) != count or len({row['pid'] for row in proofs}) != count
                    or len({row['receipt']['tokens_sha256'] for row in proofs}) != 1):
                rejected[label] = {'status':'rejected','fields':['numeric_gate']}
            write_json(out/'numeric-gate.json', numeric_receipts)
            if label == 'control' and label in rejected:
                raise ValueError('router control failed the numeric gate')
    elif args.routing_export:
        expected = None
        route_receipts = {}
        for offset, label in enumerate(order):
            repeat = job * 32 + 16 + offset
            try:
                correctness(repeat, max(1, int(deadline-time.monotonic())), True,
                            label != 'control', label)
                lookup = (RUN/'native').glob(f'*/{args.model}/B{args.batch}/*/correctness-{repeat}.json')
                path, = [item for item in lookup if json.loads(item.read_text())['job'] == str(job)]
                observed = json.loads(path.read_text())
                if (not observed['pass'] or observed['decode_sha256'] != libraries[label] or
                        label != 'control' and not observed['hf_pass']):
                    raise ValueError('router correctness is missing bound artifact or teacher-forced proof')
                fingerprint = (observed['tokens_sha256'], observed['router_indices_sha256'],
                               observed['device_uuid'], observed['routing_export_sha256'])
                if label == 'control':
                    expected = fingerprint
                elif fingerprint != expected:
                    raise ValueError('router trial changed ordered expert indices or greedy tokens')
                route_receipts[label] = {'path': str(path), 'sha256': sha(path)}
            except (ValueError, subprocess.SubprocessError) as error:
                if label == 'control':
                    raise
                rejected[label] = {'status': 'rejected', 'reason': str(error), 'round': repeat}
        write_json(out/'routing.json', {'passed': route_receipts, 'rejected': rejected})
        if not route_receipts.keys() - {'control'}:
            return 1
    for position, label in enumerate(order):
        if label in rejected:
            continue
        round_index = job * (6 if len(variants) == 2 else 16) + position
        remaining = int(deadline - time.monotonic())
        stage = stages[label]
        command = [*runner, 'run', *options, '--stage', stage, '--round', str(round_index),
                   f'--{phase}', str(variants[label]), '--target', str(targets[label]),
                   '--budget-s', str(remaining), *preparations[label]]
        completed = subprocess.run(command, timeout=max(1, remaining))
        if completed.returncode:
            if label == 'control' or (len(variants) == 2 and not args.numeric_gate):
                completed.check_returncode()
            rejected[label] = {'status': 'rejected', 'runner_exit': completed.returncode,
                               'round': round_index}
            write_json(out/'rejected.json', rejected)
            continue
        receipts = (RUN/'native').glob(f'*/{args.model}/B{args.batch}/*/{stage}-{round_index}.json')
        receipt, = [json.loads(path.read_text()) for path in receipts]
        result_path = Path(receipt['evidence'])/f'B{args.batch}'/'measurements.json'
        result = json.loads(result_path.read_text())
        if (not receipt['pass'] or receipt['job'] != str(job) or result['acceptance'] != 'strict'
                or sha(Path(receipt['evidence'])/'reference_tokens.json') != receipt['tokens_sha256']
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
    invariant_fields = ('gpu_uuid', 'tokens_sha256', 'checkpoint_sha256',
                        'config_sha256', 'prompt_sha256')
    if args.numeric_gate:
        invariant_fields = INVARIANTS
    if len(variants) > 2 or args.routing_export or args.numeric_gate:
        control = rounds[0]
        comparisons = {}
        for arm in rounds[1:]:
            label = arm['arm']
            differences = [field for field in invariant_fields if arm[field] != control[field]]
            if args.numeric_gate:
                if any(any(proof[field] != arm[field] for field in invariant_fields)
                       or proof['receipt']['tokens_sha256'] != arm['tokens_sha256']
                       for proof in numeric_proofs[label]):
                    differences.append('same_arm_correctness')
            if differences:
                rejected[label] = {'status': 'rejected', 'fields': differences, 'round': arm['round']}
                continue
            ratio = control['tpot_seconds'] / arm['tpot_seconds']
            comparisons[label] = {'ratio': ratio, 'tpot_seconds': arm['tpot_seconds'],
                                  'clears_tpot_screen': arm['tpot_seconds'] < .995 * control['tpot_seconds']}
        write_json(out/'result.json', dict(protocol_version='a800-single-v2', kind='paired_tuning',
                    control_seconds=control['tpot_seconds'], arms=comparisons, rejected=rejected))
        return 0
    if any(len({r[field] for r in rounds}) != 1 for field in invariant_fields):
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
