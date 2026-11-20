#!/usr/bin/env python3
"""Measure the SM80 native cost model under a bounded one-card allocation."""
import argparse
import json
import os
from pathlib import Path
import sys

from a100_baselines import ROOT, device_environment, sha
from a800_baselines import RUN, device_policy
from tilemega.bench.scheduler import main as schedule, write_json

SECTIONS = {
    'all': [], 'base': ['--base-only'], 'gates': ['--gates-only'],
    'epilogue': ['--epilogue-only'], 'attention': ['--attention-only'],
    'attention-fill': ['--attention-fill-only'],
    'moe': ['--moe-task-only'], 'olmoe': ['--e64-only'],
    'q1': ['--q1-only'], 'moe-merge': ['--moe-merge-only'], 'mainloop': [],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('plan', 'run'))
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--artifact', type=Path, default=RUN/'build/native-constants-sm80')
    p.add_argument('--resource-audit', type=Path)
    p.add_argument('--section', choices=SECTIONS, default='base')
    p.add_argument('--repetitions', type=int, choices=(1, 5), default=5)
    p.add_argument('--budget-s', type=int, default=720)
    p.add_argument('--timeout-s', type=int, default=90)
    args = p.parse_args()
    if not 0 < args.timeout_s < args.budget_s:
        p.error('require 0 < per-process timeout < allocation budget')
    target = args.target.resolve()
    spec = json.loads(target.read_text())
    wanted = spec['device_provenance']
    artifact = args.artifact.resolve()
    if spec['arch_tag'] != 'sm_80' or 'A800' not in wanted.get('name', '') or not wanted.get('probe_sha256'):
        p.error('the target must come from a guarded A800 resource probe')
    if not os.access(artifact, os.X_OK):
        p.error('compile the SM80 native constants probe on the login node first')
    artifact_hash = sha(artifact)
    command_tail = [str(artifact), *SECTIONS[args.section]]
    if args.section == 'mainloop':
        if not args.resource_audit:
            p.error('supply the linked mainloop resource audit')
        audit = json.loads(args.resource_audit.read_text())
        if audit['binary_sha256'] != artifact_hash or not audit.get('compatible_cases'):
            p.error('resource audit does not name compatible linked mainloop entries')
        command_tail += ['--cases', audit['compatible_cases']]
    if args.mode == 'plan':
        print(json.dumps(dict(target=str(target), device=wanted['uuid'], command=command_tail,
                             artifact_sha256=artifact_hash, count=args.repetitions,
                             timeout_s=args.timeout_s, budget_s=args.budget_s), indent=2))
        return 0

    policy = device_policy()
    device = device_environment(policy, 'A800')
    for key in ('uuid', 'name', 'pci', 'driver', 'capability', 'mig', 'memory_mib'):
        if device[key] != wanted[key]:
            raise RuntimeError(f'A800 calibration target mismatch for {key}')
    out = RUN/'calibration'/os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    policy_path = out/'guard-policy.json'
    write_json(policy_path, policy)
    env = dict(TILEMEGA_DEVICE_INDEX=str(device['index']), TILEMEGA_NUMA_NODE=device['numa_node'],
               TILEMEGA_GPU_PID_MODE='strict', OMP_NUM_THREADS='2', TMPDIR=str(RUN/'tmp'))
    (RUN/'tmp').mkdir(parents=True, exist_ok=True)
    os.environ.update(TILEMEGA_DEVICE_INDEX=env['TILEMEGA_DEVICE_INDEX'],
                      TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
    wrapper = ROOT/'scripts/run_r16_with_device_identity.py'
    command = ['taskset', '--cpu-list', device['cpus'], sys.executable, str(wrapper), '--out',
               '{attempt_dir}/job', '--guard-policy', str(policy_path), '--expected-uuid', device['uuid'],
               '--expected-pci', device['pci'], '--numa-node', device['numa_node'],
               '--allow-no-measurements', '--', *command_tail]
    jobs = [dict(name=f'sm80-{args.section}-f{i}', gpu=True, timing=True, env=env, command=command,
                 timeout_s=args.timeout_s, needs_free_mib=8192, cwd=str(ROOT),
                 artifact_sha256=artifact_hash, target_sha256=sha(target), expected_uuid=device['uuid'])
            for i in range(args.repetitions)]
    queue = out/'queue'
    queue.mkdir()
    write_json(queue/'queue_calibration.json', jobs)
    results = out/'results'
    schedule(['--queue-dir', str(queue), '--out', str(results), '--policy', str(policy_path),
              '--deadline-hours', str(args.budget_s/3600), '--release-on-blocked', '--stop-on-failure'])
    state = json.loads((results/'state.json').read_text())
    counts = {job['name']: state[job['name']]['status'] for job in jobs}
    print(json.dumps(dict(jobs=counts, after_sha256=sha(artifact))), flush=True)
    return 0 if all(status == 'done' for status in counts.values()) and sha(artifact) == artifact_hash else 1


if __name__ == '__main__':
    raise SystemExit(main())
