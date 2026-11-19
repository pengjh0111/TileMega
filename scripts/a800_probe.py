#!/usr/bin/env python3
"""Record A800 resources from a guarded CUDA/NVML probe and seed an empty target."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from a100_baselines import ROOT, device_environment, sha
from a800_baselines import RUN, device_policy
from tilemega.bench.scheduler import write_json


def write_target(output, device, identity, receipt):
    target = json.loads((ROOT/'configs/targets/sm_80.json').read_text())
    target['resources'].update(num_sms=identity['sm_count'], l2_bytes=identity['l2_bytes'],
        max_smem_per_sm=identity['shared_memory_per_sm'],
        max_dynamic_smem_per_cta=identity['shared_memory_per_block_optin'],
        regs_per_sm=identity['registers_per_sm'], max_threads_per_sm=identity['max_threads_per_sm'],
        warp_size=identity['warp_size'])
    target['device_provenance'] = dict(**device, cluster=os.environ.get('SLURM_CLUSTER_NAME', ''),
        node=socket.gethostname(), slurm_job_id=os.environ['SLURM_JOB_ID'],
        probe_sha256=sha(receipt), resources_source='guarded CUDA driver and NVML query',
        native_calibration_status='pending')
    write_json(output, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        template = json.loads((ROOT/'configs/targets/sm_80.json').read_text())
        assert template['arch_tag'] == 'sm_80' and not template['calibration']['calibrated']
        print('Will bind CUDA ordinal 0 to NVML, collect 31 idle samples, and write an uncalibrated target.')
        return 0
    policy = device_policy()
    device = device_environment(policy, 'A800')
    out = RUN/'probes'/os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    policy_path = out/'guard-policy.json'
    write_json(policy_path, policy)
    write_json(out/'nvml-device.json', device)
    env = dict(os.environ, TILEMEGA_DEVICE_INDEX=str(device['index']),
               TILEMEGA_GPU_LOCK=policy['gpu_lock_path'], TILEMEGA_TIMING='0',
               TILEMEGA_NUMA_NODE=device['numa_node'])
    env.pop('TILEMEGA_DEVICE_GROUP', None)
    wrapper = ROOT/'scripts/run_r16_with_device_identity.py'
    command = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(policy_path),
               '--out', str(out), '--needs-free-mib', '1024', '--timeout-s', '45', '--',
               sys.executable, str(wrapper), '--out', str(out), '--guard-policy', str(policy_path),
               '--expected-uuid', device['uuid'], '--expected-pci', device['pci'],
               '--numa-node', device['numa_node'], '--allow-no-measurements', '--', '/usr/bin/true']
    result = subprocess.run(command, env=env)
    if result.returncode:
        return result.returncode
    receipt = out/'physical-device-identity.json'
    write_target(out/'sm_80_a800.json', device, json.loads(receipt.read_text()), receipt)
    print(f"Guarded A800 resource receipt: {out/'sm_80_a800.json'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
