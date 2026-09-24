#!/usr/bin/env python3
"""Archive a pre-output OOM attempt and rerun all ten Qwen control processes.

Only a mixture of internally equal runs and pre-output OOMs is recoverable.
Numeric failures and unexplained exits stop this script. No automatic retry
loop is used. On success only this cell's four downstream arms are resumed.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, choices=[1, 4, 16, 64], required=True)
    ap.add_argument('--fixture', type=pathlib.Path, required=True)
    a = ap.parse_args()
    control = E / 'legacy_r8_domain' / f'qwen3_s{a.seq}'
    with (control / 'recovery.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = control / 'selected.cu.measurement'
        oom = recoverable(sorted(old.glob('process_*.log')))
        source = control / 'selected.cu'
        prior = json.loads((control / 'selected.cu.measurement.json').read_text())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if prior['candidates'][0]['source_sha256'] != digest:
            raise ValueError('source changed since the failed attempt')
        archive = control / 'failed_attempts' / f'oom_{time.time_ns()}'
        archive.mkdir(parents=True)
        files = ['selected.cu.measurement', 'selected.cu.measurement.json',
                 'measure.log', 'measure.json', 'measure.command.json',
                 'downstream_blocked.json']
        for name in files:
            path = control / name
            if path.exists():
                path.rename(archive / name)
        manifest = dict(source_sha256=digest, original_ooms=oom,
                        archive=str(archive), recovery_started_ns=time.time_ns(),
                        policy='all ten fresh processes; no numeric failure retry')
        (control / 'resource_recovery.json').write_text(json.dumps(manifest, indent=2) + '\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith('TILEMEGA_')}
        code = run(['python3', E / 'measure.py', '--source', source, '--fixture', a.fixture],
                   control / 'measure.log', env)
        if code:
            raise SystemExit(code)
        watcher = ['python3', E / 'continue_qwen.py', '--seq', str(a.seq)]
        with (control / 'resume_downstream.log').open('w') as log:
            proc = subprocess.Popen(list(map(str, watcher)), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        (control / 'resume_downstream.command.json').write_text(json.dumps(
            dict(command=list(map(str, watcher)), pid=proc.pid), indent=2) + '\n')


if __name__ == '__main__':
    main()
