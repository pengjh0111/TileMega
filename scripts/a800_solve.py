#!/usr/bin/env python3
"""Run CTG pipeline search, linked occupancy and measured selection under Slurm."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from a100_baselines import BATCHES, MODELS, ROOT, sha
from a800_baselines import ENVS, RUN, calibrated_device, device_policy
from a800_native import artifact
from tilemega.bench.scheduler import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'prepare', 'run'))
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--batch', required=True, type=int, choices=BATCHES)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--cli', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pin-theta', type=Path)
    parser.add_argument('--seed-coordinates', type=Path)
    parser.add_argument('--submission-plan', type=Path)
    parser.add_argument('--prepared-shortlist', type=Path)
    parser.add_argument('--search-budget-ms', type=int, default=600000)
    parser.add_argument('--compile-jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--top', type=int, choices=range(2, 7), default=2)
    parser.add_argument('--timeout-s', type=int, default=720)
    args = parser.parse_args()
    source, target, cli = (path.resolve(strict=True) for path in (args.export, args.target, args.cli))
    output = args.out.resolve()
    if args.timeout_s <= 0 or args.search_budget_ms <= 0 or output.suffix != '.so' or output.exists():
        parser.error('provide a positive process budget and a new .so output')
    spec = json.loads(target.read_text())
    fingerprint = spec['device_provenance']
    if (spec['arch_tag'] != 'sm_80' or 'A800' not in fingerprint['name']
            or spec.get('sm80_native_constants', {}).get('gpu_uuid') != fingerprint['uuid']
            or not os.access(cli, os.X_OK)):
        parser.error('provide an executable compiler and a calibrated A800 target')
    if args.prefill != 'prompt':
        library, identity, _ = artifact(Path(args.prefill))
        prefill, prefill_sha = str(library), identity['binary_sha256']
    elif args.model == 'olmoe_1b_7b':
        prefill = prefill_sha = 'prompt'
    else:
        parser.error('this model requires an independent prefill library')

    py = str(ENVS/'tilemega/bin/python')
    select = shlex.join((py, str(ROOT/'scripts/a800_select.py'), '--model', args.model,
                         '--batch', str(args.batch), '--prefill', prefill, '--target', str(target)))
    command = [str(cli), 'compile', str(source), str(output), '--solve', str(target),
               '--serving', 'decode', '--emit', 'serving', '--batch', str(args.batch),
               '--tp', '1', '--runtime-target', str(target), '--capacity', '1088',
               '--prompt-len', '64', '--past-range', '64:1086', '--search', 'pipeline',
               '--selection', 'measure', '--measure-top', str(args.top), '--measure-cmd', select,
               '--nvcc-split-compile', str(args.compile_jobs), '--search-budget-ms', str(args.search_budget_ms)]
    if args.pin_theta:
        pinned = args.pin_theta.resolve(strict=True)
        command.extend(('--pin-theta', str(pinned)))
    if args.seed_coordinates:
        seed = args.seed_coordinates.resolve(strict=True)
        command.extend(('--seed-coordinates', str(seed)))
    if args.mode == 'prepare':
        if os.environ.get('SLURM_JOB_ID') or args.prepared_shortlist or args.submission_plan:
            parser.error('prepare a fresh shortlist on the login node')
        command.extend(('--prepare-shortlist', '1'))
        result = subprocess.run(['timeout', '--kill-after=15s', str(args.timeout_s), *command], cwd=ROOT)
        if result.returncode:
            return result.returncode
        from a100_baselines import inputs
        from a800_shortlist_abi import check_shortlist
        path = Path(f'{output}.search')
        config = json.loads((inputs(args.model)[0]/'config.json').read_text())
        report = check_shortlist(prefill, path/'prepared.json', config, args.batch)
        write_json(path/'weight-abi.json', dict(prefill_sha256=prefill_sha, candidates=report))
        return 0
    if args.prepared_shortlist:
        prepared = args.prepared_shortlist.resolve(strict=True)
        command.extend(('--prepared-shortlist', str(prepared)))
    heartbeat_timeout_s = min(540, args.timeout_s)
    selector = ROOT/'scripts/a800_select.py'
    receipt = dict(protocol_version='a800-single-v2', kind='selection', model=args.model,
                   batch=args.batch, prefill_sha256=prefill_sha, select_sha256=sha(selector),
                   source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   command=command, target=str(target), target_sha256=sha(target),
                   export_sha256=sha(source), cli_sha256=sha(cli), calibration_uuid=fingerprint['uuid'],
                   timeout_s=args.timeout_s, heartbeat_timeout_s=heartbeat_timeout_s)
    prior = os.environ.get('TILEMEGA_PSEL_PRIOR')
    if prior:
        previous = Path(prior).resolve(strict=True)
        receipt['prior_selection'] = str(previous)
        receipt['prior_selection_sha256'] = sha(previous)
    if args.pin_theta:
        receipt['pin_theta_sha256'] = sha(pinned)
    if args.seed_coordinates:
        receipt['seed_coordinates_sha256'] = sha(seed)
    if args.prepared_shortlist:
        receipt['prepared_shortlist_sha256'] = sha(prepared)
    if args.mode == 'plan':
        print(json.dumps(receipt, indent=2))
        return 0

    if args.submission_plan and json.loads(args.submission_plan.read_text()) != receipt:
        raise ValueError('queued search artifacts, source, or configuration changed')
    policy = device_policy()
    dev = calibrated_device(policy, spec)
    out = RUN/'search'/os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    policy_path = out/'guard-policy.json'
    write_json(policy_path, policy)
    write_json(out/'planned.json', dict(receipt, device=dev))
    env = dict(os.environ, TILEMEGA_DEVICE_INDEX=str(dev['index']),
               TILEMEGA_GPU_LOCK=policy['gpu_lock_path'], TILEMEGA_NUMA_NODE=dev['numa_node'],
               TILEMEGA_TIMING='1', TILEMEGA_GPU_PID_MODE='strict', OMP_NUM_THREADS='2',
               MAX_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2')
    wrapper = [sys.executable, str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', str(out/'job'), '--guard-policy', str(policy_path),
               '--expected-uuid', dev['uuid'], '--expected-pci', dev['pci'],
               '--numa-node', dev['numa_node'], '--allow-no-measurements', '--']
    guarded = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(policy_path),
               '--out', str(out), '--needs-free-mib', '16384',
               '--timeout-s', str(args.timeout_s), '--heartbeat-timeout-s', str(heartbeat_timeout_s),
               '--', 'taskset', '--cpu-list', dev['cpus'],
               *wrapper, *command]
    return subprocess.run(guarded, cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
