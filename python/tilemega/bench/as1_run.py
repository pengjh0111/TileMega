#!/usr/bin/env python3
"""Run one project job on one visible NPU; observe other jobs without excluding them."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path('/home/d50064638/junhuip')
WORK = ROOT / 'tilemega-as1'
RUNS = ROOT / 'TileMega/runs/as1'


def contained(path, roots):
    path = path.resolve()
    return any(path.is_relative_to(root.resolve()) and path != root.resolve() for root in roots)


def snapshot():
    try:
        result = subprocess.run(['npu-smi', 'info'], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=15, check=False)
        return {'time': dt.datetime.now(dt.timezone.utc).isoformat(),
                'exit_code': result.returncode, 'output': result.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'time': dt.datetime.now(dt.timezone.utc).isoformat(), 'error': str(exc)}


def devices_from_snapshot(state):
    devices = {}
    current = None
    for line in state.get('output', '').splitlines():
        match = re.match(r'\|\s*(\d+)\s+\|\s*Ascend950PR\s*\|', line)
        if match:
            current = int(match[1])
        elif current is not None:
            cells = line.strip().strip('|').split('|')
            if len(cells) == 4 and re.search(r'\d+\s*/\s*\d+', cells[-1]):
                used, total = map(int, re.findall(r'(\d+)\s*/\s*(\d+)', cells[-1])[-1])
                if total > 0:
                    devices[current] = {'used_mib': used, 'total_mib': total,
                                        'free_mib': total-used}
                current = None
    return devices


def child_env(out, physical):
    env = os.environ.copy()
    paths = {
        'TMPDIR': WORK / 'tmp', 'XDG_CACHE_HOME': WORK / 'cache/xdg',
        'PIP_CACHE_DIR': WORK / 'cache/pip', 'CCACHE_DIR': WORK / 'cache/ccache',
        'TORCH_EXTENSIONS_DIR': WORK / 'cache/torch_extensions',
        'TRITON_CACHE_DIR': WORK / 'cache/triton',
        'TORCHINDUCTOR_CACHE_DIR': WORK / 'cache/torchinductor',
        'HF_HOME': WORK / 'cache/huggingface',
        'PYTHONPYCACHEPREFIX': WORK / 'cache/pycache',
        'ASCEND_CACHE_PATH': WORK / 'cache/cann',
        'ASCEND_WORK_PATH': out / 'cann',
        'ASCEND_PROCESS_LOG_PATH': out / 'cann/plog',
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update({'ASCEND_RT_VISIBLE_DEVICES': str(physical), 'AS1_LOGICAL_DEVICE_ID': '0',
                'AS1_PHYSICAL_DEVICE_ID': str(physical), 'PYTHONNOUSERSITE': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    return env


def stop_owned_group(proc):
    # start_new_session below makes this process the leader of our own group.
    # No UID-based process lookup, external PID list, or device reset is used.
    for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        # Also terminate any descendants still in this job's process group.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-device', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--sample-s', type=float, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.timeout_s <= 900 or args.sample_s < 1:
        parser.error('A command, timeout in (0,900], and sample interval >=1 are required')
    out = args.out.absolute()
    if not contained(out, [RUNS, WORK / 'logs']):
        parser.error('Output must be beneath project runs/as1 or tilemega-as1/logs')
    # Never overwrite evidence or reuse another job's directory.
    out.mkdir(parents=True, exist_ok=False)
    before = snapshot()
    devices = devices_from_snapshot(before)
    if args.physical_device == 'auto':
        if not devices:
            parser.error('Cannot choose device automatically; specify inventoried physical ID')
        physical = max(devices, key=lambda n: devices[n]['free_mib'])
        reason = 'Largest observed free HBM; external jobs are observation only'
    else:
        physical = int(args.physical_device)
        reason = 'Explicit inventoried physical device selection'
    (out / 'before.json').write_text(json.dumps(before, indent=2)+'\n')
    started = time.monotonic()
    record = {'physical_device_id': physical, 'logical_device_id_expected': 0,
              'mapping_evidence': 'stated; validate with ACL mapping probe',
              'selection_reason': reason, 'observed_hbm': devices.get(physical),
              'command': command, 'device_locked': False, 'timeout_s': args.timeout_s,
              'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'starting'}
    external_pids = [int(pid) for dev, pid in re.findall(
        r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|', before.get('output', ''), re.MULTILINE)
        if int(dev) == physical]
    record['external_pids_observed_before'] = external_pids
    record['other_processes_present_before'] = bool(external_pids) if 'output' in before else None
    record['timing_status'] = 'contended' if external_pids else 'load_not_fully_characterized'
    record_path = out / 'job.json'
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    proc = None
    exit_code = 125
    try:
        with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
            proc = subprocess.Popen(command, cwd=out, env=child_env(out, physical),
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(pid=proc.pid, process_group=proc.pid, status='running')
            record_path.write_text(json.dumps(record, indent=2)+'\n')
            with (out / 'samples.jsonl').open('x') as samples:
                while True:
                    remaining = args.timeout_s-(time.monotonic()-started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, args.timeout_s)
                    try:
                        exit_code = proc.wait(timeout=min(args.sample_s, remaining))
                        record['status'] = 'passed' if exit_code == 0 else 'failed'
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-started >= args.timeout_s:
                            raise
                        samples.write(json.dumps(snapshot())+'\n')
                        samples.flush()
    except subprocess.TimeoutExpired:
        record.update(status='timeout', device_work_may_outlive_host=True)
        exit_code = 124
        if proc is not None:
            stop_owned_group(proc)
    except KeyboardInterrupt:
        record['status'] = 'interrupted'
        exit_code = 130
        if proc is not None:
            stop_owned_group(proc)
    except OSError as exc:
        record.update(status='launch_error', error=str(exc))
    finally:
        record.update(exit_code=exit_code, elapsed_s=time.monotonic()-started,
                      finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        if proc is not None:
            record['child_exit_code'] = proc.poll()
        (out / 'after.json').write_text(json.dumps(snapshot(), indent=2)+'\n')
        record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'job': str(out), **record}, ensure_ascii=False))
    return exit_code if exit_code >= 0 else 128-exit_code


if __name__ == '__main__':
    sys.exit(main())
