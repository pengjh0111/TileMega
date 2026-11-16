#!/usr/bin/env python3
"""Require empty process boundaries and a stable singleton during each probe."""
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import dnn1_constants as constants
import dnn1_device as device
ORIGINAL_RUN = constants.run


def validate_window(window):
    owners = {tuple(sorted(sample['owners'])) for sample in window['samples'] if sample['owners']}
    if window['errors'] or len(owners) != 1 or any(len(group) != 1 for group in owners):
        raise RuntimeError('probe process window lacks one stable observed owner')


def strict_run(args):
    spec = constants.read(args.spec)
    if not spec.get('strict_processes') or spec['strict_entry_sha256'] != constants.sha(__file__):
        raise ValueError('strict process entry identity changed or was not requested')
    args.out.mkdir(parents=True, exist_ok=True)
    boundaries, windows = [], []
    real_gpu, real_popen = device.gpu, subprocess.Popen
    binaries = {row['path'] for row in constants.read(spec['build'])['binaries'].values()}

    def boundary(gpu):
        value = real_gpu(gpu)
        boundaries.append(value)
        constants.write(args.out/'strict_boundaries.json', boundaries)
        if value['owners'] or value['used_mib'] > 1:
            raise RuntimeError('probe boundary is not empty at <=1MiB')
        return value

    class Probe(real_popen):
        def communicate(self, *args_, **kwargs):
            window = dict(container_pid=self.pid, started=time.time(), samples=[], errors=[])
            def monitor():
                while self.poll() is None:
                    try:
                        window['samples'].append(real_gpu(args.gpu))
                    except Exception as error:
                        window['errors'].append(str(error))
                    time.sleep(.1)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            result = super().communicate(*args_, **kwargs)
            window['child_completed'] = time.time()
            thread.join()
            windows.append(window)
            constants.write(args.out/'strict_windows.json', windows)
            validate_window(window)
            return result

    def spawn(command, *args_, **kwargs):
        return (Probe if str(command[0]) in binaries else real_popen)(command, *args_, **kwargs)

    device.gpu, subprocess.Popen = boundary, spawn
    try:
        ORIGINAL_RUN(args)
        constants.write(args.out/'strict_acceptance.json', dict(
            status='sampled_boundaries_and_windows_pass', strict_entry_sha256=constants.sha(__file__),
            boundary_count=len(boundaries), binary_processes=len(windows),
            live_samples=sum(len(row['samples']) for row in windows),
            boundaries_sha256=constants.sha(args.out/'strict_boundaries.json'),
            windows_sha256=constants.sha(args.out/'strict_windows.json'),
            namespace_ownership='inferred singleton between empty boundaries; host/container PID mapping unverified',
            probe_binary_changed=False))
    finally:
        device.gpu, subprocess.Popen = real_gpu, real_popen


if __name__ == '__main__':
    # Keep the ordinary collector's fingerprint and receipt format.
    constants.run = strict_run
    try:
        raise SystemExit(constants.main())
    finally:
        constants.run = ORIGINAL_RUN
