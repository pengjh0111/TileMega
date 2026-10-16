#!/usr/bin/env python3
"""Run an unchanged harness under its guard and bind receipts to a CUDA UUID."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_guard(policy, index):
    for value in (policy, json.loads(Path(policy['gpu_allowlist_file']).read_text())
                  if policy.get('gpu_allowlist_file') else {}):
        if value.get('allowed_gpu_indices') is not None and index not in value['allowed_gpu_indices']:
            raise ValueError('device is excluded by the active user GPU restriction')
    expected = f'/tmp/tilemega_gpu{index}.lock'
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != str(index) or
            os.environ.get('TILEMEGA_DEVICE_INDEX') != str(index) or
            os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or
            os.environ.get('TILEMEGA_GPU_LOCK') != expected):
        raise ValueError('matching explicit device variables and the per-card outer guard are required')
    with open(expected, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(handle, fcntl.LOCK_UN)
        raise ValueError('outer guard does not hold the physical GPU lock')


def probe_cuda():
    # Driver queries create no CUDA context; the separate probe also exits
    # before the official harness reserves its model and KV-cache memory.
    lib = ctypes.CDLL('libcuda.so.1')
    def check(code):
        if code:
            raise RuntimeError(f'CUDA driver identity query failed: {code}')
    check(lib.cuInit(0))
    device = ctypes.c_int()
    check(lib.cuDeviceGet(ctypes.byref(device), 0))
    raw_uuid = (ctypes.c_ubyte * 16)()
    check(lib.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), device))
    bus = ctypes.create_string_buffer(32)
    check(lib.cuDeviceGetPCIBusId(bus, 32, device))
    memory = ctypes.c_size_t()
    check(lib.cuDeviceTotalMem_v2(ctypes.byref(memory), device))
    sms = ctypes.c_int()
    check(lib.cuDeviceGetAttribute(ctypes.byref(sms), 16, device))
    pci = bus.value.decode()
    node_path = Path('/sys/bus/pci/devices') / pci.lower() / 'numa_node'
    return dict(cuda_uuid=str(uuid.UUID(bytes=bytes(raw_uuid))), cuda_pci_bus_id=pci,
                numa_node=int(node_path.read_text()), sm_count=sms.value,
                total_memory=memory.value, probe_pid=os.getpid(),
                cuda_driver_queries_without_context=True)


def annotate(out, receipt, allow_empty=False):
    identity = json.loads(receipt.read_text())
    paths = sorted((out / 'measurements').rglob('measurements.json'))
    if not paths and not allow_empty:
        raise ValueError('successful harness produced no measurements.json')
    for path in paths:
        value = json.loads(path.read_text())
        if value.get('physical_device_identity') not in (None, identity):
            raise ValueError('measurement already has another physical device identity')
        value.update(physical_device_identity=identity,
                     physical_device_identity_file=str(receipt.resolve()),
                     physical_device_identity_sha256=sha(receipt))
        path.write_text(json.dumps(value, indent=2) + '\n')
    return [str(path) for path in paths]


def main():
    if sys.argv[1:] == ['--probe']:
        policy = Path(os.environ['TILEMEGA_DEVICE_PROBE_POLICY'])
        require_guard(json.loads(policy.read_text()), int(os.environ['TILEMEGA_DEVICE_INDEX']))
        print(json.dumps(probe_cuda()))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-uuid', required=True)
    parser.add_argument('--expected-pci', required=True)
    parser.add_argument('--numa-node', type=int, required=True)
    parser.add_argument('--guard-policy', type=Path, default=Path('configs/bench/guard_policy.json'))
    parser.add_argument('--allow-no-measurements', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('supply -- followed by the unchanged harness command')
    index = int(os.environ['TILEMEGA_DEVICE_INDEX'])
    require_guard(json.loads(args.guard_policy.read_text()), index)
    probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe'],
                           text=True, capture_output=True,
                           env=dict(os.environ, TILEMEGA_DEVICE_PROBE_POLICY=str(args.guard_policy.resolve())))
    if probe.returncode:
        raise RuntimeError('CUDA identity probe failed: ' + probe.stderr[-2000:])
    identity = json.loads(probe.stdout)
    nvml = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    if (identity['cuda_uuid'] != nvml.removeprefix('GPU-') or
            identity['cuda_uuid'] != args.expected_uuid.removeprefix('GPU-') or
            identity['cuda_pci_bus_id'].lower() != args.expected_pci.lower() or
            identity['numa_node'] != args.numa_node):
        raise ValueError('CUDA/NVML UUID, PCI or NUMA differs from the predeclared device')
    identity.update(created=time.time(), gpu_index=index, nvml_uuid=nvml,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        probe_exit_code=probe.returncode, wrapper_pid=os.getpid(), wrapper_sha256=sha(__file__),
        outer_guard_holds_percard_lock=True, guard_policy_sha256=sha(args.guard_policy))
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / 'physical-device-identity.json'
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    result = subprocess.run(command)
    paths = annotate(args.out, receipt, args.allow_no_measurements) if result.returncode == 0 else []
    (args.out / 'physical-device-command.json').write_text(json.dumps(dict(
        command=command, actual_returncode=result.returncode, measurements=paths), indent=2) + '\n')
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
