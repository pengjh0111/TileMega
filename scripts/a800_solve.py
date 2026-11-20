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
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--batch', required=True, type=int, choices=BATCHES)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--cli', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pin-theta', type=Path)
    parser.add_argument('--seed-coordinates', type=Path)
    parser.add_argument('--top', type=int, choices=range(2, 7), default=2)
    parser.add_argument('--timeout-s', type=int, default=720)
    args = parser.parse_args()
    source, target, cli = (path.resolve(strict=True) for path in (args.export, args.target, args.cli))
    output = args.out.resolve()
    if args.timeout_s <= 0 or output.suffix != '.so' or output.exists():
        parser.error('provide a positive process budget and a new .so output')
    spec = json.loads(target.read_text())
    fingerprint = spec['device_provenance']
    if (spec['arch_tag'] != 'sm_80' or 'A800' not in fingerprint['name']
            or spec.get('sm80_native_constants', {}).get('gpu_uuid') != fingerprint['uuid']
            or not os.access(cli, os.X_OK)):
        parser.error('provide an executable compiler and a calibrated A800 target')
    if args.prefill != 'prompt':
        prefill = str(artifact(Path(args.prefill))[0])
    elif args.model == 'olmoe_1b_7b':
        prefill = 'prompt'
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
               '--nvcc-split-compile', '2']
    if args.pin_theta:
        pinned = args.pin_theta.resolve(strict=True)
        command.extend(('--pin-theta', str(pinned)))
    if args.seed_coordinates:
        seed = args.seed_coordinates.resolve(strict=True)
        command.extend(('--seed-coordinates', str(seed)))
    heartbeat_timeout_s = min(540, args.timeout_s)
    receipt = dict(command=command, target=str(target), target_sha256=sha(target),
                   export_sha256=sha(source), cli_sha256=sha(cli), calibration_uuid=fingerprint['uuid'],
                   timeout_s=args.timeout_s, heartbeat_timeout_s=heartbeat_timeout_s)
    if args.pin_theta:
        receipt['pin_theta_sha256'] = sha(pinned)
    if args.seed_coordinates:
        receipt['seed_coordinates_sha256'] = sha(seed)
    if args.mode == 'plan':
        print(json.dumps(receipt, indent=2))
        return 0

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
