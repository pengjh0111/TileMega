#!/usr/bin/env python3
"""Plan baseline cells on login and execute a finite single-GPU Slurm cohort."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from a100_baselines import BATCHES, ENGINES, MODELS, ROOT, device_environment, inspect_inputs, prepare, sha
from run_r16_with_device_identity import probe_cuda
from tilemega.bench.gpu_guard import host_proc, smi
from tilemega.bench.scheduler import main as schedule, write_json

RUN = ROOT / 'runs/a800'
INVENTORY = ROOT.parent / 'TileMega-a100/runs/a100/inventory'
ENVS = ROOT.parent / 'envs'
PROBE = 'a800_device_identity'
CUDA_RESOURCES = dict(num_sms='sm_count', l2_bytes='l2_bytes',
    max_smem_per_sm='shared_memory_per_sm', max_dynamic_smem_per_cta='shared_memory_per_block_optin',
    regs_per_sm='registers_per_sm', max_threads_per_sm='max_threads_per_sm', warp_size='warp_size')


def hardware_class(device):
    return tuple(device[key] for key in ('name', 'memory_mib', 'capability', 'mig'))


def calibrated_device(policy, target):
    device = device_environment(policy, 'A800')
    cuda = probe_cuda()
    provenance = target['device_provenance']
    if (hardware_class(device) != hardware_class(provenance)
            or any(cuda[attribute] != target['resources'][field]
                   for field, attribute in CUDA_RESOURCES.items())):
        raise RuntimeError(f"A800 hardware differs from the calibration profile: {device['name']}")
    return device


def selection(args):
    filters = [args.model or MODELS, args.engine or ENGINES, args.batch or BATCHES,
               args.round_index or range(3)]
    return [f'{model}_{engine}_b{batch:02d}_r{repeat}'
            for model in filters[0] for engine in filters[1] for batch in filters[2]
            for repeat in filters[3]]


def mpk_process_budget(budget, requested, policy):
    # The scheduler admits each GPU step only if its full guard allowance and
    # 20s of scheduling margin fit. A new cohort runs two guarded steps.
    guard_allowance = policy['samples'] * policy['interval_s'] + 60 + 20
    probe_timeout = 120
    reserve = probe_timeout + 2 * (guard_allowance + policy['own_child_settle_s']) + 10
    available = min(requested, int(budget - reserve))
    if available <= 0:
        raise ValueError('MPK allocation cannot fit the device probe and both GPU guards')
    return available


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
                max_idle_wait_s=180, own_child_settle_s=2, keep_host_pid_visible=True,
                confirm_hidden_memory=True, confirm_retired_owners=True)


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
        device = device_environment(policy, 'A800')
        if args.expected_target:
            expected = json.loads(args.expected_target.read_text())['device_provenance']
            if hardware_class(device) != hardware_class(expected):
                raise RuntimeError('Slurm assigned a different A800 hardware class')
        first_device = run/'device.json'
        if first_device.exists() and hardware_class(json.loads(first_device.read_text())) != hardware_class(device):
            raise RuntimeError('choose a separate cohort for a different A800 hardware class')
        allocation = run/'allocations'/os.environ['SLURM_JOB_ID']
        allocation.mkdir(parents=True, exist_ok=True)
        policy_path = allocation/'guard-policy.json'
        write_json(policy_path, policy)
        expected_checkpoint = (args.model[0], args.expected_checkpoint_sha256) if args.expected_checkpoint_sha256 else None
        if args.engine == ['mpk']:
            args.process_timeout_s = mpk_process_budget(args.budget_s, args.process_timeout_s, policy)
            args.heartbeat_timeout_s = min(args.heartbeat_timeout_s, args.process_timeout_s)
            print(f'MPK process budget {args.process_timeout_s}s after both guard allowances', flush=True)
        jobs = prepare(policy_path, run, INVENTORY, ENVS, 'A800',
                       args.process_timeout_s, args.heartbeat_timeout_s, args.seed_suite, expected_checkpoint,
                       engines={'mpk': 'mpk'} if args.engine == ['mpk'] else None)
        inputs = json.loads((run/'baseline_inputs.json').read_text())
        if not first_device.exists():
            write_json(first_device, inputs['device'])
        write_json(allocation/'baseline_inputs.json', inputs)
        if args.engine == ['mpk']:
            preflight = json.loads((RUN/'mpk/preflight.json').read_text())
            serving = ROOT/'scripts/a800_mpk_serving.py'
            if (sha(serving) != preflight['serving_sha256'] or
                    sha(ROOT/'python/tilemega/serving/mpk_baseline.py') != preflight['harness_sha256']):
                raise RuntimeError('MPK harness changed since source preflight')
            fingerprint = {k: preflight[k] for k in ('revision', 'diff_sha256', 'extension_sha256',
                           'harness_sha256', 'serving_sha256', 'attention_sha256', 'target_sha256')}
            write_json(allocation/'mpk-inputs.json', dict(fingerprint, guard_policy_sha256=sha(policy_path)))
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
        state_path = run/'baseline/state.json'
        if args.retry_failed and state_path.is_file():
            state = json.loads(state_path.read_text())
            if any(row['status'] == 'running' for row in state.values()):
                raise RuntimeError('reconcile the interrupted process before retrying')
            for name in [PROBE, *wanted]:
                if state.get(name, {}).get('status') in ('failed', 'skipped'):
                    state[name].update(status='pending', retry_at=0)
            write_json(state_path, state)
        schedule(options)
        state = json.loads(state_path.read_text())
        selected = {name: state[name]['status'] for name in [PROBE, *wanted]}
        print(json.dumps(dict(cohort=args.cohort, device=policy['gpu_uuid'], status=selected)), flush=True)
        return 0 if all(status == 'done' for status in selected.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--expected-target', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--seed-suite', choices=('vllm', 'tilemega'), default='vllm')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--model', choices=MODELS, action='append')
    parser.add_argument('--engine', choices=(*ENGINES, 'mpk'), action='append')
    parser.add_argument('--batch', type=int, choices=BATCHES, action='append')
    parser.add_argument('--round', dest='round_index', type=int, choices=range(3), action='append')
    parser.add_argument('--budget-s', type=int, default=840)
    parser.add_argument('--process-timeout-s', type=int, default=600)
    parser.add_argument('--heartbeat-timeout-s', type=int, default=300)
    args = parser.parse_args()
    if not args.cohort or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.cohort):
        parser.error('cohort must contain only ASCII letters, digits, hyphens and underscores')
    if not 0 < args.heartbeat_timeout_s <= args.process_timeout_s or args.budget_s <= 0:
        parser.error('expected positive budget and 0 < heartbeat timeout <= process timeout')
    if args.process_timeout_s >= args.budget_s and not (args.mode == 'run' and args.engine == ['mpk']):
        parser.error('process timeout must fit the allocation')
    if args.expected_checkpoint_sha256 and len(args.model or []) != 1:
        parser.error('an explicit checkpoint hash requires one selected model')
    if args.seed_suite != 'vllm' and (args.engine != ['torchcompile'] or not args.expected_checkpoint_sha256):
        parser.error('use the alternate seed suite only for a TorchCompile cell with a verified hash')
    if 'mpk' in (args.engine or []) and args.engine != ['mpk']:
        parser.error('use a dedicated cohort for MPK')
    wanted = [] if args.probe_only else selection(args)
    models = inspect_inputs(INVENTORY)
    if args.mode == 'plan':
        print(json.dumps(dict(cohort=args.cohort, manifest=models, probe=PROBE, steps=wanted,
                              budget_s=args.budget_s, maximum_single_process_s=args.process_timeout_s),
                         indent=2))
        return 0
    required_suites = ENGINES.values() if args.seed_suite == 'vllm' else (args.seed_suite,)
    if args.engine == ['mpk']:
        required_suites = ('tilemega', args.seed_suite, 'mpk')
    for suite in required_suites:
        if not os.access(ENVS/suite/'bin/python', os.X_OK):
            raise RuntimeError(f'workspace is missing the {suite} Python interpreter')
    return run_cohort(args, wanted)


if __name__ == '__main__':
    raise SystemExit(main())
