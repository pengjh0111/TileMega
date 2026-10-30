#!/usr/bin/env python3
"""Run one project job on one visible NPU; observe other jobs without excluding them."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/home/d50064638/junhuip')
WORK = ROOT / 'tilemega-as1'
RUNS = ROOT / 'TileMega/runs/as1'
# In-memory ownership records only; never a device admission lock.
_pending_telemetry = []


def process_identity(pid):
    raw = Path('/proc', str(pid), 'stat').read_text()
    fields = raw[raw.rfind(')')+2:].split()
    return {'pid': pid, 'start_ticks': int(fields[19]),
            'process_group': int(fields[2]), 'session': int(fields[3])}


def contained(path, roots):
    path = path.resolve()
    return any(path.is_relative_to(root.resolve()) and path != root.resolve() for root in roots)


def observed_processes(output):
    result={}
    for _,pid in re.findall(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|',output,re.MULTILINE):
        try:result[pid]=process_identity(int(pid))
        except (FileNotFoundError,ProcessLookupError):result[pid]=None
    return result


def snapshot(timeout_s=30):
    if not 0 < timeout_s <= 60:
        raise ValueError('Management query timeout must be in (0,60] seconds')
    # The toolkit prepends different copies of libc_sec/libmmpa. Management
    # uses the installed driver's libraries; compute children keep their CANN env.
    env = os.environ.copy()
    driver_libs = [str(p) for p in (Path('/usr/local/Ascend/driver/lib64/driver'),
                                  Path('/usr/local/Ascend/driver/lib64/common')) if p.is_dir()]
    if driver_libs:
        env['LD_LIBRARY_PATH'] = ':'.join(driver_libs + ([env['LD_LIBRARY_PATH']]
                                                        if env.get('LD_LIBRARY_PATH') else []))
    command = [shutil.which('npu-smi') or 'npu-smi', 'info']
    stamp = {'time': dt.datetime.now(dt.timezone.utc).isoformat(),
             'command': command, 'driver_library_precedence': driver_libs,
             'query_timeout_s': timeout_s}
    for proc, identity in list(_pending_telemetry):
        if proc.poll() is None:
            return {**stamp, 'error': 'Previous own telemetry query has not exited',
                    'query_identity': identity, 'query_may_remain_in_kernel_wait': True}
        _pending_telemetry.remove((proc, identity))
    try:
        proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True, env=env)
        identity = process_identity(proc.pid)
        try:
            output, _ = proc.communicate(timeout=timeout_s)
            return {**stamp, 'exit_code': proc.returncode, 'output': output,
                    'query_identity': identity,'process_identities':observed_processes(output)}
        except subprocess.TimeoutExpired:
            # Popen's context manager/run() can wait indefinitely after killing
            # an uninterruptible management query. Reaping also needs a bound.
            if proc.poll() is None and process_identity(proc.pid) == identity:
                proc.kill()
            try:
                output, _ = proc.communicate(timeout=1)
                return {**stamp, 'error': f'Own telemetry query exceeded{timeout_s:g}s',
                        'exit_code': proc.returncode, 'output': output,
                        'query_identity': identity, 'query_timed_out': True}
            except subprocess.TimeoutExpired:
                _pending_telemetry.append((proc, identity))
                if proc.stdout is not None:
                    proc.stdout.close()
                return {**stamp, 'error': f'Own telemetry query exceeded{timeout_s:g}s and bounded reaping',
                        'query_identity': identity, 'query_timed_out': True,
                        'query_may_remain_in_kernel_wait': True}
    except OSError as exc:
        return {**stamp, 'error': str(exc)}


def devices_from_snapshot(state):
    if state.get('exit_code') != 0:
        return {}
    devices = {}
    current = None
    health = None
    for line in state.get('output', '').splitlines():
        match = re.match(r'\|\s*(\d+)\s+\|\s*Ascend950PR\s*\|\s*([^|]+)\|', line)
        if match:
            current = int(match[1]);health=match[2].strip()
        elif current is not None:
            cells = line.strip().strip('|').split('|')
            if len(cells) == 4 and re.search(r'\d+\s*/\s*\d+', cells[-1]):
                used, total = map(int, re.findall(r'(\d+)\s*/\s*(\d+)', cells[-1])[-1])
                if total > 0:
                    pci = re.search(r'\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b', line)
                    utilization=(cells[1].strip() if cells[1].strip().isdigit() else
                                 re.match(r'\s*(\d+)\s+\d+\s*/',cells[-1]))
                    if hasattr(utilization,'group'):utilization=utilization.group(1)
                    devices[current] = {'used_mib': used, 'total_mib': total,
                                        'free_mib': total-used,
                                        'pci_bus_id': pci.group(0).lower() if pci else None,
                                        'health':health,
                                        'utilization_percent':int(utilization) if utilization else None}
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
    if WORK.name=='tilemega-as2':
        venv=Path(env.get('VIRTUAL_ENV','/')).resolve()
        if venv in {WORK/'envs/vllm-ascend',WORK/'envs/sglang-npu'}:
            # Different TorchNPU stacks must not reuse each other's extensions.
            for name in ('TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
                paths[name]/=venv.name
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update({'ASCEND_RT_VISIBLE_DEVICES': str(physical), 'AS1_LOGICAL_DEVICE_ID': '0',
                'AS1_PHYSICAL_DEVICE_ID': str(physical), 'PYTHONNOUSERSITE': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if WORK.name == 'tilemega-as2':
        env.update(AS2_LOGICAL_DEVICE_ID='0', AS2_PHYSICAL_DEVICE_ID=str(physical))
    return env


def stop_owned_group(proc):
    # start_new_session below makes this process the leader of our own group.
    # No UID-based process lookup, external PID list, or device reset is used.
    owned = []
    def remember(pid):
        try:
            identity = process_identity(pid)
            if identity['session'] != proc.pid:
                return
            owned.append(identity)
            children = Path('/proc', str(pid), 'task', str(pid), 'children').read_text().split()
            for child in children:
                remember(int(child))
        except (FileNotFoundError, ProcessLookupError):
            pass
    remember(proc.pid)
    if proc.poll() is None:
        if not owned or owned[0]['pid'] != proc.pid or owned[0]['process_group'] != proc.pid:
            raise RuntimeError('Cannot verify the owned process group')
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            remember(proc.pid)
    # A reaped leader can have a reused PID; verify each surviving descendant.
    for identity in reversed(owned):
        try:
            if process_identity(identity['pid']) == identity:
                os.kill(identity['pid'], signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
    proc.wait(timeout=5)


def main(project='as1', command_preflight=None):
    global WORK, RUNS
    if project not in {'as1', 'as2'}:
        raise ValueError('Unknown controlled project')
    WORK = ROOT / ('tilemega-' + project)
    RUNS = ROOT / ('TileMega/runs/' + project)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-device', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--sample-s', type=float, default=15)
    parser.add_argument('--telemetry-timeout-s', type=float, default=30)
    parser.add_argument('--required-free-mib', type=int, default=0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or not 0 < args.timeout_s <= 900 or args.sample_s < 1 or
            args.required_free_mib < 0 or not 0 < args.telemetry_timeout_s <= 60):
        parser.error('Require a command, job timeout in (0,900], telemetry timeout in (0,60], and sample interval >=1')
    out = args.out.absolute()
    if not contained(out, [RUNS, WORK / 'logs']):
        parser.error('Output must be beneath the selected project runs or logs')
    # Never overwrite evidence or reuse another job's directory.
    out.mkdir(parents=True, exist_ok=False)
    before = snapshot(args.telemetry_timeout_s)
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
              'telemetry_timeout_s': args.telemetry_timeout_s,
              'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'starting'}
    record['pci_bus_id'] = devices.get(physical, {}).get('pci_bus_id')
    telemetry_ok = before.get('exit_code') == 0 and physical in devices
    record['telemetry_status'] = 'available' if telemetry_ok else 'unavailable'
    external_pids = [int(pid) for dev, pid in re.findall(
        r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|', before.get('output', ''), re.MULTILINE)
        if int(dev) == physical]
    record['external_pids_observed_before'] = external_pids
    record['other_processes_present_before'] = bool(external_pids) if telemetry_ok else None
    record['timing_status'] = 'contended' if external_pids else 'load_not_fully_characterized'
    mapping = RUNS / 'inventory' / ('acl_device_' + str(physical) + '.json')
    if mapping.is_file():
        verified_map = json.loads(mapping.read_text())
        if (verified_map.get('physical_device_id') == physical and
                verified_map.get('logical_device_id') == 0 and verified_map.get('visible_count') == 1):
            current_pci = record['pci_bus_id']
            bound_pci = verified_map.get('pci_bus_id')
            mapping_evidence = ('verified' if telemetry_ok and current_pci and bound_pci == current_pci
                                else 'stated; cached ACL mapping is not bound to the current PCI identity')
            record.update(mapping_evidence=mapping_evidence, mapping_source=str(mapping))
    executable = Path(command[0])
    if executable.is_file():
        with executable.open('rb') as binary:
            record['executable_sha256'] = hashlib.file_digest(binary, 'sha256').hexdigest()
    record_path = out / 'job.json'
    record.update(parent_pid=os.getpid(), project=project,
                  manifest_id=str(out.relative_to(ROOT)), required_free_mib=args.required_free_mib)
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    if command_preflight is not None:
        proof = command_preflight(command, physical)
        record['capability_preflight'] = proof
        if proof['status'] == 'blocked':
            record.update(status='blocked-capability', reason=proof['reason'], exit_code=78,
                          elapsed_s=time.monotonic()-started)
            record_path.write_text(json.dumps(record, indent=2)+'\n')
            (out/'after.json').write_text(json.dumps(snapshot(args.telemetry_timeout_s), indent=2)+'\n')
            print(json.dumps(record))
            return 78
    unhealthy=telemetry_ok and devices[physical].get('health')!='OK'
    if unhealthy or args.required_free_mib and (not telemetry_ok or devices[physical]['free_mib'] < args.required_free_mib):
        status = 'blocked-health' if unhealthy else 'blocked-resource' if telemetry_ok else 'telemetry_unavailable'
        reason = 'Selected device health changed before child launch' if unhealthy else 'Insufficient observed free HBM' if telemetry_ok else 'Cannot verify the requested HBM budget from npu-smi'
        record.update(status=status, reason=reason, exit_code=75, elapsed_s=time.monotonic()-started)
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        (out / 'after.json').write_text(json.dumps(snapshot(args.telemetry_timeout_s), indent=2)+'\n')
        print(json.dumps(record))
        return 75
    proc = None
    exit_code = 125
    try:
        with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
            proc = subprocess.Popen(command, cwd=out, env=child_env(out, physical),
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            identity = process_identity(proc.pid)
            if identity['session'] != proc.pid or identity['process_group'] != proc.pid:
                raise RuntimeError('Child did not enter its own session/process group')
            record.update(**identity, status='running')
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
                        remaining = args.timeout_s-(time.monotonic()-started)
                        if remaining <= 0:
                            raise
                        samples.write(json.dumps(snapshot(min(args.telemetry_timeout_s, remaining)))+'\n')
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
        (out / 'after.json').write_text(json.dumps(snapshot(args.telemetry_timeout_s), indent=2)+'\n')
        record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'job': str(out), **record}, ensure_ascii=False))
    return exit_code if exit_code >= 0 else 128-exit_code


if __name__ == '__main__':
    sys.exit(main())
