"""Conservative owner-count audit above the unchanged scheduler/guard."""
import json
import fcntl
from pathlib import Path
import time

from sm90d1_common import ROOT, PYTHON, read, write
from sm90r4_dispatch import run_unit as dispatch_unit


def audit_samples(rows, hf=False):
    # Native serving uses one GPU process; C-1 adds one HF child while its
    # parent retains its CUDA context. Unresolved host PIDs cannot prove ownership.
    limit = 2 if hf else 1
    owners = {pid for row in rows if row['phase'] == 'running' for pid in row['owners']}
    reasons = []
    if len(owners) > limit:
        reasons.append(f'{len(owners)} distinct running GPU PIDs exceed expected {limit}')
    if any(row['owners'] for row in rows if row['phase'] in ('preflight', 'after')):
        reasons.append('GPU owners present before or after the process')
    if not rows:
        reasons.append('guard samples missing')
    return dict(passed=not reasons, reasons=reasons, observed_pids=sorted(owners),
                expected_max_pids=limit, ownership_verified=False,
                evidence='verified sampled owner counts; namespace ownership remains provisional')


def audit_unit(unit):
    unit = Path(unit)
    protocol = read(unit/'unit.json')
    attempt = unit/protocol['try']
    spec = read(attempt/'spec.json')
    state = read(attempt/'scheduler/state.json')
    checks = []
    for name, record in state.items():
        output = Path(record['out'])
        process = read(output/'job/sm90r3-process.json')
        correctness = spec.get('correctness', {})
        hf = bool(correctness.get('c1') or correctness.get('hf64')) and \
             process['fresh'] < correctness.get('reference_processes', 1 << 30)
        samples = [json.loads(line) for line in (output/'guard.jsonl').read_text().splitlines()]
        check = audit_samples(samples, hf)
        guard = read(output/'guard_result.json')
        check.update(job=name, guard_code=guard['code'])
        if guard['code'] or record['status'] != 'done':
            check['passed'] = False
            check['reasons'].append('scheduler or guard did not accept the process')
        checks.append(check)
    result = dict(passed=bool(checks) and all(c['passed'] for c in checks), checks=checks,
                  at=time.time(), method='conservative sampled distinct-PID upper bound')
    write(unit/'hardware-audit.json', result)
    return result


def audit_trace_unit(unit):
    unit = Path(unit)
    checks = []
    for trace in read(unit/'unit.json')['runs']:
        output = Path(trace).parents[1]
        samples = [json.loads(line) for line in (output/'guard.jsonl').read_text().splitlines()]
        check = audit_samples(samples)
        if read(output/'guard_result.json')['code']:
            check['passed'] = False
            check['reasons'].append('guard rejected the trace process')
        checks.append(check)
    result = dict(passed=bool(checks) and all(c['passed'] for c in checks), checks=checks, at=time.time())
    write(unit/'hardware-audit.json', result)
    return result


def quarantine(unit, reason):
    unit = Path(unit)
    rejected = unit/f'rejected-{time.time_ns()}'
    rejected.mkdir()
    for name in ('unit.json', 'result.json', 'hardware-audit.json'):
        path = unit/name
        if path.exists():path.rename(rejected/name)
    write(rejected/'reason.json', dict(reason=reason, at=time.time()))


def run_unit(template, unit, cls):
    unit=Path(unit);unit.mkdir(parents=True,exist_ok=True)
    with (unit/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _run_unit(template,unit,cls)


def _run_unit(template, unit, cls):
    from sm90x1_parallel import enabled, run_unit as parallel_unit
    if enabled(read(template)):
        return parallel_unit(template,unit,cls)
    for _ in range(3):
        # An archived unit need not permanently exclude its now-idle card.
        # claim() still requires the same free-owner, memory, power and lock checks.
        result = dispatch_unit(template, unit, cls, reconsider_idle=True, poll_s=5,stall_s=60,preflight_handoff=True,
                               scheduler_command=[PYTHON,str(ROOT/'scripts/sm90x1_scheduler.py')])
        audit = audit_unit(unit)
        if audit['passed']:return result
        quarantine(unit, audit)
    raise RuntimeError(f'three complete units failed the GPU-owner audit: {unit}')
