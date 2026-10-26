#!/usr/bin/env python3
"""Verify that timeout cleanup reaches only this wrapper job's process group."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[3]
runner = repo / 'python/tilemega/bench/as1_run.py'
spec = importlib.util.spec_from_file_location('as1_run', runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parsed = module.devices_from_snapshot({'exit_code': 0, 'output': (repo / 'runs/as1/inventory/npu.txt').read_text()})
if parsed[0]['total_mib'] != 131072 or parsed[1]['total_mib'] != 131072:
    raise AssertionError('HBM telemetry parsed the memory/hugepage columns')
out = repo / 'runs/as1/jobs' / ('runner_timeout_'+str(time.time_ns()))
sentinel = subprocess.Popen(['/usr/bin/sleep', '60'], start_new_session=True)
try:
    child = ('import subprocess,time; from pathlib import Path; '
             'p=subprocess.Popen(["/usr/bin/sleep","60"]); '
             'Path("descendant.pid").write_text(str(p.pid)); time.sleep(60)')
    result = subprocess.run([sys.executable, str(runner), '--physical-device', '0',
                             '--out', str(out), '--timeout-s', '0.7', '--',
                             sys.executable, '-c', child], capture_output=True, text=True, timeout=40)
    if result.returncode != 124 or sentinel.poll() is not None:
        raise AssertionError('Timeout status or unrelated process protection failed')
    pid = int((out / 'descendant.pid').read_text())
    stat_path = Path('/proc', str(pid), 'stat')
    if stat_path.exists():
        value = stat_path.read_text()
        if value[value.rfind(')')+2:].split()[0] != 'Z':
            raise AssertionError('Own descendant survived timeout')
    receipt = {'test': 'T-14', 'status': 'passed', 'evidence': 'verified',
               'job': str(out), 'exit_code': result.returncode,
               'unrelated_sentinel_pid': sentinel.pid, 'sentinel_survived': True,
               'own_descendant_stopped': True, 'device_kernel_launches': 0}
    (out / 'runner_check.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
finally:
    if sentinel.poll() is None:
        os.killpg(sentinel.pid, signal.SIGTERM)
        sentinel.wait(timeout=5)
