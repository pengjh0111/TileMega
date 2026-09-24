#!/usr/bin/env python3
"""Archive a resource-affected attempt and rerun all ten control processes.

Only a mixture of internally equal runs and pre-output OOMs is recoverable.
Alternatively, --reason contention requires ten internally equal runs and
a busy-GPU snapshot in the raw device log. Numeric failures and unexplained
exits stop this script. No automatic retry loop is used. Qwen recovery resumes
only this cell's downstream arms, retaining already launched arms and audits.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time

from gpu_admission import resource_failure
from measure import parse, run

E = pathlib.Path(__file__).resolve().parent


def recoverable(logs):
    if len(logs) != 10:
        raise ValueError('expected the entire original ten-process attempt')
    oom = 0
    for log in logs:
        raw = log.read_text()
        good, timing = parse(raw)
        if good and timing:
            continue
        if not resource_failure(raw):
            raise ValueError('not a pre-output resource failure: ' + str(log))
        oom += 1
    if not oom:
        raise ValueError('no resource failure to recover')
    return oom


def contention_attempt(logs, device):
    if len(logs)!=10 or not all(parse(p.read_text())[0] and parse(p.read_text())[1] for p in logs):
        raise ValueError('contention remeasurement requires ten internally equal original runs')
    import re
    text=device.read_text()
    utilization=re.search(r'Utilization\s+GPU\s+:\s*(\d+)\s*%',text)
    if not utilization or int(utilization[1])<=5:
        raise ValueError('no recorded busy GPU snapshot justifying contention remeasurement')
    return int(utilization[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, choices=[1, 4, 16, 64], required=True)
    ap.add_argument('--model', choices=['llama','qwen3'], default='qwen3')
    ap.add_argument('--reason', choices=['oom','contention'], default='oom')
    ap.add_argument('--fixture', type=pathlib.Path, required=True)
    a = ap.parse_args()
    control = E / 'legacy_r8_domain' / f'{a.model}_s{a.seq}'
    with (control / 'recovery.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = control / 'selected.cu.measurement'
        logs=sorted(old.glob('process_*.log'))
        observed=(recoverable(logs) if a.reason=='oom' else
                  contention_attempt(logs,old/'device_before.log'))
        source = control / 'selected.cu'
        prior = json.loads((control / 'selected.cu.measurement.json').read_text())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if prior['candidates'][0]['source_sha256'] != digest:
            raise ValueError('source changed since the failed attempt')
        archive = control / ('failed_attempts' if a.reason=='oom' else 'excluded_attempts') / f'{a.reason}_{time.time_ns()}'
        archive.mkdir(parents=True)
        files = ['selected.cu.measurement', 'selected.cu.measurement.json',
                 'measure.log', 'measure.json', 'measure.command.json',
                 'downstream_blocked.json']
        for name in files:
            path = control / name
            if path.exists():
                path.rename(archive / name)
        manifest = dict(source_sha256=digest, reason=a.reason,
                        original_ooms=observed if a.reason=='oom' else 0,
                        initial_utilization_percent=observed if a.reason=='contention' else None,
                        archive=str(archive), recovery_started_ns=time.time_ns(),
                        policy='all ten fresh processes; no numeric failure retry')
        (control / 'resource_recovery.json').write_text(json.dumps(manifest, indent=2) + '\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith('TILEMEGA_')}
        code = run(['python3', E / 'measure.py', '--source', source, '--fixture', a.fixture],
                   control / 'measure.log', env)
        if code:
            raise SystemExit(code)
        if a.model!='qwen3':return
        watcher = ['python3', E / 'continue_qwen.py', '--seq', str(a.seq)]
        with (control / 'resume_downstream.log').open('w') as log:
            proc = subprocess.Popen(list(map(str, watcher)), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        (control / 'resume_downstream.command.json').write_text(json.dumps(
            dict(command=list(map(str, watcher)), pid=proc.pid), indent=2) + '\n')


if __name__ == '__main__':
    main()
