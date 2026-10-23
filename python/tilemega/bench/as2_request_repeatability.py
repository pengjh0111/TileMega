"""Audit five serial fresh request processes against an admitted short capture.

CPU-only analysis: it never launches, signals, locks or queries a device. This
establishes repeatability/request-reset evidence, not independent model math or
performance. The golden capture needs its own independent-reference result.
"""
import argparse
import json
from pathlib import Path
import struct

from tilemega.bench.as2_models import RUNS, controlled
from tilemega.bench.as2_native import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binding', 'golden', 'golden_numeric', 'out'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--jobs', nargs=5, type=Path, required=True)
    a = p.parse_args()
    for name in ('binding', 'golden', 'golden_numeric', 'out'):
        setattr(a, name, controlled(getattr(a, name), [RUNS]))
    if a.out.exists():
        raise FileExistsError(a.out)
    identity = json.loads((a.binding / 'request.json').read_text())
    golden = json.loads((a.golden / 'result.json').read_text())
    numeric = json.loads(a.golden_numeric.read_text())
    if numeric['status'] != 'pass' or not numeric['actual_single_forward_launch_count']:
        raise ValueError('Golden needs an independently passed numeric/launch comparison')
    if golden['binding_directory'] != str(a.binding) or golden['osl'] != 4:
        raise ValueError('Require the same four-token solved Plan and canonical inputs')
    if (numeric['model_id'] != identity['model_id'] or numeric['batch'] != identity['batch']
            or golden['warmup'] or golden['repetitions'] != 1
            or numeric['rows'][0]['greedy']['native_sha256'] != digest(a.golden / 'request_0/tokens.bin')):
        raise ValueError('Numeric evidence does not identify this golden request')
    for phase, entry in identity['phases'].items():
        if any(digest(a.binding / phase / name) != sha for name, sha in entry['bound_hashes'].items()):
            raise ValueError('Bound Plan changed')
    if any(digest(a.binding / 'inputs' / row['file']) != row['sha256'] for row in identity['input_buffers']):
        raise ValueError('Canonical request input changed')
    payload = {str(f.relative_to(a.golden / 'request_0')): digest(f)
               for f in (a.golden / 'request_0').rglob('*.bin')
               if f.name not in ('trace.bin', 'profile.bin')}
    if not payload or not any('kv_k_history_' in f for f in payload):
        raise ValueError('Golden lacks payload/KV-history evidence')
    receipts = []
    rows = []
    for path in a.jobs:
        path = controlled(path, [RUNS])
        job = json.loads((path / 'job.json').read_text())
        cmd = job['command']
        if (job['status'] != 'passed' or job['exit_code'] or job.get('device_locked') is not False
                or len(cmd) != 8 or cmd[1:3] != ['request', str(a.binding)]
                or cmd[4:] != ['4', '1', '1', '1']):
            raise ValueError('Require five finite, successful diagnostic reset processes')
        native = controlled(Path(cmd[3]), [RUNS])
        result = json.loads((native / 'result.json').read_text())
        if (result['binding_directory'] != str(a.binding) or result['osl'] != 4
                or result['warmup'] != 1 or result['repetitions'] != 1
                or result['host_kernel_launches'] != 8):
            raise ValueError('Request-reset or forward-count contract changed')
        launches = [json.loads(line) for line in (native / 'launches.jsonl').read_text().splitlines()]
        launch_ok = len(launches) == 8 and all(
            r['forward_index'] == i % 4 and r['seq'] == (1 if i % 4 else 64)
            and r['past'] == (63 + i % 4 if i % 4 else 0)
            for i, r in enumerate(launches))
        failures = []
        for request in (-1, 0):
            folder = native / f'request_{request}'
            for name, sha in payload.items():
                file = folder / name
                if not file.is_file() or digest(file) != sha:
                    failures.append(dict(request=request, file=name, reason='missing_or_different_payload'))
            for phase, epoch in (('prefill', 0), ('decode', 2)):
                tasks = list(struct.iter_unpack('<7I', (a.binding / phase / 'tasks.bin').read_bytes()))
                stages = list(struct.iter_unpack('<17Q16I', (a.binding / phase / 'stages.bin').read_bytes()))
                trace = list(struct.iter_unpack('<8Q', (folder / phase / 'trace.bin').read_bytes()))
                valid = len(trace) == len(tasks) and all(
                    (r[0], r[1], r[2], r[5]) == (t[2], t[0], t[1], epoch)
                    and r[4] >= r[3] and r[6:8] == stages[t[0]][:2]
                    for t, r in zip(tasks, trace))
                if not valid:
                    failures.append(dict(request=request, phase=phase, reason='task_receipt'))
        receipts.append(job)
        rows.append(dict(job=str(path), pid=job['pid'], start_ticks=job['start_ticks'],
            physical_device_id=job['physical_device_id'], elapsed_s=job['elapsed_s'],
            requests_checked=2, payload_files_per_request=len(payload),
            launches_pass=launch_ok, failures=failures, passed=launch_ok and not failures))
    ordered = sorted(receipts, key=lambda r: r['started_at'])
    if (len({(r['pid'], r['start_ticks']) for r in receipts}) != 5
            or len({r['executable_sha256'] for r in receipts}) != 1
            or len({r['physical_device_id'] for r in receipts}) != 1
            or any(x['finished_at'] > y['started_at'] for x, y in zip(ordered, ordered[1:]))):
        raise ValueError('Processes were reused, overlapped, or changed binary/device')
    result = dict(schema='tilemega.as2.request_repeatability.v1',
        status='pass' if all(r['passed'] for r in rows) else 'fail',
        scope='five fresh serial processes; two requests per process; no performance admission',
        model_id=identity['model_id'], golden_numeric_sha256=digest(a.golden_numeric),
        processes_passed=sum(r['passed'] for r in rows), processes_total=5,
        one_forward_one_launch=all(r['launches_pass'] for r in rows), payload_sha256=payload, rows=rows)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'payload_sha256')}))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
