#!/usr/bin/env python3
"""Run an isolated native stage ledger with the frozen A800 input and strict guard."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import BATCHES, MODELS, ROOT, inputs, inspect_inputs, sha
from a800_baselines import ENVS, INVENTORY, RUN, calibrated_device, device_policy
from a800_native import artifact
from tilemega.bench.scheduler import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', choices=BATCHES, required=True, type=int)
    parser.add_argument('--prefill', required=True, type=Path)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--independent-arms', action='store_true')
    parser.add_argument('--submission-plan', type=Path)
    parser.add_argument('--timeout-s', type=int, default=330)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error('trace deadline must be positive')

    prefill = artifact(args.prefill)[0]
    reference, original, manifest = artifact(args.reference)
    trace = args.trace.resolve(strict=True)
    identity = json.loads(Path(f'{trace}.identity.json').read_text())
    if (not identity['trace'] or identity['performance_eligible']
            or identity['arch'] != 'sm_80' or identity['nvcc_arch'] != 'sm_80'
            or identity['binary_sha256'] != sha(trace)
            or identity['manifest_sha256'] != sha(Path(f'{trace}.plan.json'))
            or identity['generated_source_sha256'] != original['generated_source_sha256']
            or identity['target_sha256'] != original['target_sha256']
            or original['target_sha256'] != sha(args.target)
            or manifest['runtime_batch'] != args.batch):
        parser.error('the trace and reference must share one SM80 native plan and target')
    planned = dict(model=args.model, batch=args.batch, reference=sha(reference), trace=sha(trace),
                   prefill=sha(prefill), target=sha(args.target), timeout_s=args.timeout_s,
                   independent_arms=args.independent_arms)
    if args.mode == 'plan':
        print(json.dumps(planned, indent=2))
        return 0
    if args.submission_plan and json.loads(args.submission_plan.read_text()) != planned:
        parser.error('trace inputs differ from the submission plan')
    policy = device_policy()
    device = calibrated_device(policy, json.loads(args.target.read_text()))
    folder = RUN/'optimization'/'decode-traces'/os.environ['SLURM_JOB_ID']
    folder.mkdir(parents=True, exist_ok=False)
    policy_path = folder/'guard-policy.json'
    write_json(policy_path, policy)
    write_json(folder/'planned.json', dict(planned, device=device))
    config, prompt = inputs(args.model)
    checkpoint = inspect_inputs(INVENTORY)[args.model]['checkpoint_bytes']
    child = [str(ENVS/'vllm/bin/python'), '-m', 'tilemega.serving.random_checkpoint',
             '--config', str(config), '--out', str(folder/'checkpoint'),
             '--seed', '0', '--max-gib', str(math.ceil(checkpoint/2**30)), '--',
             sys.executable, str(ROOT/'scripts/sm90r4_ledger.py'), 'run',
             '--batch', str(args.batch), '--prefill', str(prefill), '--decode', str(trace),
             '--reference', str(reference), '--prompt-ids', str(prompt), '--out', str(folder/'ledger')]
    if args.independent_arms:
        child.append('--independent-arms')
    wrapper = [sys.executable, str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', str(folder/'job'), '--guard-policy', str(policy_path),
               '--expected-uuid', device['uuid'], '--expected-pci', device['pci'],
               '--numa-node', device['numa_node'], '--allow-no-measurements', '--', *child]
    env = dict(os.environ, TILEMEGA_GPU_PID_MODE='strict', TILEMEGA_TIMING='0',
               TILEMEGA_DEVICE_INDEX=str(device['index']), TILEMEGA_NUMA_NODE=device['numa_node'],
               TILEMEGA_GPU_LOCK=policy['gpu_lock_path'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(policy_path),
               '--out', str(folder/'guard'), '--needs-free-mib', str(math.ceil(checkpoint/2**20) + 8192),
               '--timeout-s', str(args.timeout_s), '--', 'taskset', '--cpu-list', device['cpus'], *wrapper]
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
