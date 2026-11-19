#!/usr/bin/env python3
"""Plan baseline cells on login and execute a finite single-GPU Slurm cohort."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from a100_baselines import BATCHES, ENGINES, MODELS, ROOT, inspect_inputs, prepare
from run_r16_with_device_identity import probe_cuda
from tilemega.bench.gpu_guard import host_proc, smi
from tilemega.bench.scheduler import main as schedule, write_json

RUN = ROOT / 'runs/a800'
INVENTORY = ROOT.parent / 'TileMega-a100/runs/a100/inventory'
ENVS = ROOT.parent / 'envs'
PROBE = 'a800_device_identity'


def selection(args):
    filters = [args.model or MODELS, args.engine or ENGINES, args.batch or BATCHES,
               args.round_index or range(3)]
    return [f'{model}_{engine}_b{batch:02d}_r{repeat}'
            for model in filters[0] for engine in filters[1] for batch in filters[2]
            for repeat in filters[3]]


def device_policy():
    job, mask = os.getenv('SLURM_JOB_ID', ''), os.getenv('CUDA_VISIBLE_DEVICES', '')
    if (not job.isdecimal() or os.getenv('SLURM_JOB_PARTITION') != 'gpu_a800'
            or not mask or ',' in mask or mask in ('-1', 'NoDevFiles')):
        raise RuntimeError('requires an existing single-GPU gpu_a800 Slurm allocation')
    host_proc()
    cuda = probe_cuda()
    if cuda['cuda_visible_count'] != 1 or cuda['compute_capability'] != [8, 0]:
        raise RuntimeError('the Slurm allocation must expose exactly one SM80 CUDA device')
    physical = 'GPU-' + cuda['cuda_uuid']
    index, nvml_uuid = (field.strip() for field in
        smi('-i', physical, '--query-gpu=index,uuid', '--format=csv,noheader').strip().split(','))
    if nvml_uuid != physical or not index.isdecimal():
        raise RuntimeError('CUDA ordinal and NVML device could not be matched')
    return dict(allowed_gpu_indices=[int(index)], gpu_uuid=physical, cuda_visible_devices=mask,
                gpu_lock_path=str(RUN/'locks'/f'{physical}.lock'), strict_host_owners=True, samples=31,
                interval_s=1, max_util_pct=2, max_hidden_mib=1024, hidden_violations=1,
                max_idle_wait_s=180)


def capacity_receipts(run, selected, jobs, inputs, policy):
    """Only skip when BF16 weights alone exceed the measured card capacity."""
    physical_bytes = inputs['device']['memory_mib'] * 2**20
    selection = set(selected)
    receipts = {step['name']: dict(status='OOM', device_uuid=policy['gpu_uuid'],
                reason='BF16 weight data alone exceeds the physical device capacity',
                checkpoint_bytes=inputs['models'][step['model']]['checkpoint_bytes'],
                device_bytes=physical_bytes)
                for step in jobs if step['name'] in selection
                and inputs['models'][step['model']]['checkpoint_bytes'] > physical_bytes}
    for name, record in receipts.items():
        write_json(run/f'{name}.oom.json', record)
    return [name for name in selected if name not in receipts]


def run_cohort(args, wanted):
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=ROOT, text=True, timeout=10).strip():
        raise RuntimeError('commit tracked changes before launching a measured cohort')
    run = RUN / 'cohorts' / args.cohort
    run.mkdir(parents=True, exist_ok=True)
    with (run/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = device_policy()
        allocation = run/'allocations'/os.environ['SLURM_JOB_ID']
        allocation.mkdir(parents=True, exist_ok=True)
        policy_path = allocation/'guard-policy.json'
        write_json(policy_path, policy)
        pinned = run/'device.json'
        if pinned.exists() and json.loads(pinned.read_text())['uuid'] != policy['gpu_uuid']:
            raise RuntimeError('allocation received a different UUID from the saved cohort')
        jobs = prepare(policy_path, run, INVENTORY, ENVS, 'A800',
                       args.process_timeout_s, args.heartbeat_timeout_s)
        inputs = json.loads((run/'baseline_inputs.json').read_text())
        if not pinned.exists():
            write_json(pinned, inputs['device'])
        write_json(allocation/'baseline_inputs.json', inputs)
        wanted = capacity_receipts(run, wanted, jobs, inputs, policy)
        options = ['--queue-dir', str(run/'queue'), '--out', str(run/'baseline'),
                   '--policy', str(policy_path), '--deadline-hours', str(args.budget_s/3600),
                   '--release-on-blocked', '--resume-timing']
        write_json(allocation/'selected.json', [PROBE, *wanted])
        for name in [PROBE, *wanted]:
            options.extend(('--only', name))
        os.environ.update(TILEMEGA_DEVICE_INDEX=str(policy['allowed_gpu_indices'][0]),
                          TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
        os.environ.pop('TILEMEGA_DEVICE_GROUP', None)
        schedule(options)
        state = json.loads((run/'baseline/state.json').read_text())
        selected = {name: state[name]['status'] for name in [PROBE, *wanted]}
        print(json.dumps(dict(cohort=args.cohort, device=policy['gpu_uuid'], status=selected)), flush=True)
        return 0 if all(status == 'done' for status in selected.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--model', choices=MODELS, action='append')
    parser.add_argument('--engine', choices=ENGINES, action='append')
    parser.add_argument('--batch', type=int, choices=BATCHES, action='append')
    parser.add_argument('--round', dest='round_index', type=int, choices=range(3), action='append')
    parser.add_argument('--budget-s', type=int, default=840)
    parser.add_argument('--process-timeout-s', type=int, default=600)
    parser.add_argument('--heartbeat-timeout-s', type=int, default=300)
    args = parser.parse_args()
    if not args.cohort or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.cohort):
        parser.error('cohort must contain only ASCII letters, digits, hyphens and underscores')
    if not 0 < args.heartbeat_timeout_s <= args.process_timeout_s < args.budget_s:
        parser.error('expected 0 < heartbeat timeout <= process timeout < budget')
    wanted = [] if args.probe_only else selection(args)
    models = inspect_inputs(INVENTORY)
    if args.mode == 'plan':
        print(json.dumps(dict(cohort=args.cohort, manifest=models, probe=PROBE, steps=wanted,
                              budget_s=args.budget_s, maximum_single_process_s=args.process_timeout_s),
                         indent=2))
        return 0
    for suite in ENGINES.values():
        if not os.access(ENVS/suite/'bin/python', os.X_OK):
            raise RuntimeError(f'workspace is missing the {suite} Python interpreter')
    return run_cohort(args, wanted)


if __name__ == '__main__':
    raise SystemExit(main())
