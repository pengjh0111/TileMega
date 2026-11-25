#!/usr/bin/env python3
"""Measure a frozen batch of SM80 service cases under the A800 GPU guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import ROOT, sha
from a800_baselines import RUN, calibrated_device, device_policy
from tilemega.bench.scheduler import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--identity-sha256', required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--target-sha256', required=True)
    p.add_argument('--run-label', help='unique evidence folder for a probe in a shared allocation')
    p.add_argument('--sample', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.run_label and (Path(a.run_label).name != a.run_label or a.run_label in ('.', '..')):
        p.error('run label must be a local directory name')
    identity_path = a.binary.with_suffix('.identity.json')
    identity = json.loads(identity_path.read_text())
    if (sha(identity_path) != a.identity_sha256 or sha(a.binary) != identity['binary_sha256']
            or sha(a.target) != a.target_sha256
            or identity['contract'] not in ('sm80-dense-gemm-service-v1', 'sm80-fused-expert-service-v1',
                                           'sm80-attention-fresh-service-v1', 'sm80-dense-width-scan-v1',
                                           'sm80-dense-role-scan-v1')):
        raise ValueError('the queued service inputs changed')
    cases = identity['cases']
    if not 1 <= len(cases) <= 9 or len(cases) != len(set(cases)):
        raise ValueError('submit one to nine unique service cases')
    if a.sample:
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
            raise ValueError('service probes require exclusive admission under the guard')
        for case in cases:
            if Path(case).name != case:
                raise ValueError('invalid service case label')
            result = subprocess.run([str(a.binary), case], capture_output=True, text=True, timeout=35)
            if result.returncode:
                raise RuntimeError(f'service case {case}: {result.stderr.strip()}')
            write_json(a.sample/f'{case}.json', json.loads(result.stdout))
            print(case, result.stdout.strip(), flush=True)
        return 0

    policy = device_policy()
    device = calibrated_device(policy, json.loads(a.target.read_text()))
    out = RUN/'calibration'/'r2-supplement-service'/os.environ['SLURM_JOB_ID']
    if a.run_label:
        out /= a.run_label
    out.mkdir(parents=True, exist_ok=False)
    policy_path = out/'policy.json'
    write_json(policy_path, policy)
    write_json(out/'artifact.json', identity)
    write_json(out/'frozen.json', dict(identity_sha256=a.identity_sha256, target_sha256=a.target_sha256))
    env = dict(os.environ, TILEMEGA_GPU_PID_MODE='strict', TILEMEGA_TIMING='0',
               TILEMEGA_DEVICE_INDEX=str(device['index']), TILEMEGA_NUMA_NODE=device['numa_node'],
               TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
    wrapper = [sys.executable, str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', str(out/'job'), '--guard-policy', str(policy_path),
               '--expected-uuid', device['uuid'], '--expected-pci', device['pci'],
               '--numa-node', device['numa_node'], '--allow-no-measurements', '--']
    command = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(policy_path),
               '--out', str(out/'guard'), '--needs-free-mib', '1024', '--timeout-s', '355', '--',
               'taskset', '--cpu-list', device['cpus'], *wrapper,
               sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--sample', str(out)]
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
