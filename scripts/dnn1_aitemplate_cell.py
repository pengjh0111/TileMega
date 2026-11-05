#!/usr/bin/env python3
"""Five-process AIT cell and independent strict PID-window collector."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import shlex
import subprocess
import time

from dnn1_aitemplate import ROOT, AIT_ROOT, require_tvm_complete, save, sha, source_identity
import dnn1_baselines as protocol
from dnn1_baseline_boundaries import idle_boundary, final_boundary

AFFINITY = {0: list(range(32, 40)), 1: list(range(40, 48)), 2: list(range(16, 24)),
            3: list(range(24, 32)), 4: list(range(48, 56)), 5: list(range(56, 64)),
            7: list(range(64, 72))}


def queue(options):
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    gpu, affinity = options.gpu, AFFINITY[options.gpu]
    provider = getattr(options, 'provider', 'original')
    if provider not in ('original', 'bounded-settle', 'fifo-settle'):
        raise ValueError('unknown versioned AIT guard provider')
    python = str(ROOT / 'scripts' / dict(original='dnn1_background_python.py',
        **{'bounded-settle': 'dnn1_settling_background_python.py',
           'fifo-settle': 'dnn1_fifo_background_python.py'})[provider])
    sources = source_identity()['files']
    sources.update({str(Path(__file__).resolve()): sha(__file__)})
    for name in ('dnn1_device.py', 'dnn1_baselines.py', 'dnn1_idle_floor_python.py',
                 'dnn1_resource_entry.py', 'dnn1_baseline_boundaries.py',
                 'dnn1_background_python.py', 'dnn1_background_cards.py'):
        sources[str(ROOT / 'scripts' / name)] = sha(ROOT / 'scripts' / name)
    if provider in ('bounded-settle', 'fifo-settle'):
        for path in (Path(python), ROOT / 'scripts/dnn1_preflight_settle.py',
                     ROOT / 'scripts/dnn1_preflight_settle_guard.patch',
                     ROOT / 'scripts/dnn1_settling_background_python.py',
                     ROOT / 'python/tilemega/bench/gpu_guard.py'):
            sources[str(path)] = sha(path)
    profile_cache = AIT_ROOT / 'profile_cache' / f'gpu{gpu}'
    cell = dict(model=options.model, batch=options.batch, gpu=gpu, cpu_affinity=affinity,
                parallel=True, python=python, timeout_s=14400, cache=str(AIT_ROOT / 'engines'),
                profile_cache=str(profile_cache),
                guard_provider=provider,
                source_sha256=sources, graph_mode=getattr(options, 'graph', True))
    save(out / 'cell.json', cell)
    metadata = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, pci = (x.strip() for x in metadata.split(','))
    domain, bus, slot = pci.split(':')
    pci = f'{int(domain, 16):04x}:{bus}:{slot}'
    env = protocol.gpu_env(gpu, dict(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', NUM_BUILDERS='4',
                                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                                   CACHE_DIR=str(profile_cache)))
    command = ['numactl', '--all', '--physcpubind=' + ','.join(map(str, affinity)), python,
        str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(protocol.POLICY),
        '--out', '{attempt_dir}/job', '--expected-uuid', uuid, '--expected-pci', pci,
        '--numa-node', str(0 if gpu < 4 else 1), '--allow-no-measurements', '--', python,
        str(Path(__file__).resolve()), 'run', '--cell', str(out / 'cell.json'),
        '--out', '{attempt_dir}/job/ait']
    (out / 'queue').mkdir()
    save(out / 'queue/queue_dnn1_ait.json', [dict(name=f'ait-{options.model}-b{options.batch}', gpu=True,
         timing=True, priority=200, timeout_s=86400, needs_free_mib=8192, cwd=str(ROOT),
         after=[], env=env, command=command)])
    launcher = ['env', '-u', 'TILEMEGA_DEVICE_GROUP', '-u', 'TILEMEGA_GPU_LOCK_HELD',
        '-u', 'TILEMEGA_RUN_NONCE', *[f'{key}={value}' for key, value in env.items()], python,
        '-m', 'tilemega.bench.scheduler', '--queue-dir', str(out / 'queue'), '--out',
        str(out / 'scheduler'), '--policy', str(protocol.POLICY), '--deadline-hours', '96']
    path = out / 'launch.sh'
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n' +
        shlex.join(['flock', f'/tmp/tilemega_dnn1_baseline_scheduler_gpu{gpu}.lock', *launcher]) + '\n')
    path.chmod(0o755)
    return 0


def profile_cache_snapshot(directory):
    directory = Path(directory)
    files, errors = {}, {}
    if directory.exists():
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                try:
                    files[str(path)] = sha(path)
                except OSError as error:
                    errors[str(path)] = repr(error)
    return dict(path=str(directory), files=files, errors=errors)


def run(options):
    require_tvm_complete()
    cell = json.loads(options.cell.read_text())
    protocol.require_session(cell)
    if cell['gpu'] == 6:
        raise ValueError('GPU6 is excluded')
    for path, digest in cell['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('prepared AIT source changed: ' + path)
    options.out.mkdir(parents=True, exist_ok=False)
    session = dict(schema=1, system='aitemplate', model=cell['model'], batch=cell['batch'],
                   gpu=cell['gpu'], graph_mode=cell.get('graph_mode', True),
                   started=time.time(), status='running', processes=[])
    profile_cache = cell.get('profile_cache', os.environ.get('CACHE_DIR', ''))
    if (not profile_cache or os.environ.get('CACHE_DIR') != profile_cache or
            not Path(profile_cache).resolve().is_relative_to((AIT_ROOT / 'profile_cache').resolve())):
        raise ValueError('an explicit independent AIT profile cache under runs is required')
    session['profile_cache_before'] = profile_cache_snapshot(profile_cache)
    def publish_session():
        session['profile_cache_after'] = profile_cache_snapshot(profile_cache)
        save(options.out / 'cell-result.json', session)
    for number in range(5):
        for path, digest in cell['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('prepared AIT source changed: ' + path)
        role = 'measurement' if number < 3 else 'correctness'
        folder = options.out / f'{role}-{number}'
        before_drain = idle_boundary(cell)
        if before_drain['status'] != 'empty_observed':
            session.update(status='infrastructure_rejection', error='nonempty pre-process GPU',
                           boundary=before_drain)
            publish_session()
            return 75
        before = final_boundary(before_drain)
        command = [cell['python'], str(ROOT / 'scripts/dnn1_aitemplate.py'), 'sample',
                   '--model', cell['model'], '--batch', str(cell['batch']), '--seed', '20261007',
                   '--out', str(folder / 'payload'), '--cache', cell['cache'],
                   '--round-index', str(number)]
        if role == 'correctness':
            command += ['--correctness-only']
        if not cell.get('graph_mode', True):
            command += ['--no-graph']
        if number == 0:
            command += ['--profile-capture']
        result = protocol.execute(command, folder, os.environ.copy(), cell['timeout_s'])
        drain = idle_boundary(cell)
        after = (drain['samples'][-1]['telemetry'] if drain['samples'] else
                 dict(owners={}, used_mib=float('inf')))
        receipt = folder / 'payload/measurements.json'
        result.update(role=role, before=before, after=after, before_drain=before_drain, drain=drain,
                      receipt=str(receipt), receipt_sha256=sha(receipt) if receipt.exists() else None)
        session['processes'].append(result)
        publish_session()
        if (result['returncode'] or after['owners'] or after['used_mib'] > 1 or
                drain['status'] != 'empty_observed'):
            session.update(status='failed', ended=time.time())
            publish_session()
            return result['returncode'] or 75
    session.update(status='complete_pending_outer_audit', ended=time.time())
    publish_session()
    return 0


def audit_windows(process, receipt, outer):
    issues = []
    snapshots_path = Path(process['receipt']).parent / 'guard.jsonl'
    snapshots = [json.loads(line) for line in snapshots_path.read_text().splitlines()]
    start = [s for s in snapshots if s['label'] == 'before-measurement']
    end = [s for s in snapshots if s['label'] == 'after-measurement']
    if len(start) != 1 or len(end) != 1:
        issues.append('missing or duplicate measurement boundaries')
    else:
        owners = set(map(str, start[0]['pids']))
        if len(owners) != 1 or owners != set(map(str, end[0]['pids'])):
            issues.append('measurement boundary PID set is not one stable PID')
        if end[0]['time'] < start[0]['time']:
            issues.append('measurement boundaries are reversed')
        for snapshot in outer:
            if start[0]['time'] <= snapshot['time'] <= end[0]['time']:
                if set(map(str, snapshot.get('owners', {}))) - owners:
                    issues.append('another PID sampled inside measurement window')
    if process['before']['owners'] or process['after']['owners']:
        issues.append('process boundary contains GPU owners')
    if process['returncode'] != 0 or not receipt.get('correctness_passed'):
        issues.append('process exit or original numerical threshold failed')
    if sha(process['receipt']) != process['receipt_sha256']:
        issues.append('receipt content changed since process completed')
    return dict(issues=issues, snapshots=str(snapshots_path), snapshots_sha256=sha(snapshots_path),
                before_measurement=start, after_measurement=end,
                scope='sampled windows; no host/container PID identity or continuous exclusivity claim')


def collect(options):
    session = json.loads(options.session.read_text())
    outer = [json.loads(line) for line in options.guard.read_text().splitlines()]
    issues = []
    endings = [s for s in outer if s.get('phase') == 'after']
    if len(endings) != 1 or endings[0].get('exit_code') != 0:
        issues.append('outer guard did not finish with exit0')
    elif endings[0]['time'] < session.get('ended', float('inf')):
        issues.append('outer guard ended before the session completed')
    if session.get('status') != 'complete_pending_outer_audit':
        issues.append('cell did not complete all requested processes')
    processes = session['processes']
    if len(processes) != 5 or len(set(p['pid'] for p in processes)) != 5:
        issues.append('P5 requires exactly five distinct fresh process PIDs')
    rows, windows = [], []
    for process in processes:
        receipt = json.loads(Path(process['receipt']).read_text())
        if any(receipt.get(k) != session[k] for k in ('model', 'batch')):
            issues.append('process identifies another cell')
        if receipt.get('system') != 'aitemplate' or receipt.get('precision') != 'bf16':
            issues.append('wrong baseline system or precision')
        if receipt.get('gpu_index') != session['gpu']:
            issues.append('process identifies another GPU')
        if receipt.get('pid') != process['pid']:
            issues.append('receipt identifies another process')
        if any(p.get('owners') or not 0 <= p.get('used_mib', -1) <= 1
               for p in (process['before'], process['after'])):
            issues.append('dirty process memory boundary')
        if process.get('drain', {}).get('status') != 'empty_observed':
            issues.append('process context did not drain')
        try:
            if (process['before'] != final_boundary(process['before_drain']) or
                    process['after'] != final_boundary(process['drain'])):
                issues.append('boundary differs from the final two idle observations')
        except (KeyError, ValueError):
            issues.append('missing two-sample process-and-memory boundary')
        if receipt.get('graph_mode', {}).get('requested') != session.get('graph_mode'):
            issues.append('graph mode differs from the fixed cohort')
        if receipt.get('identity', {}).get('recipe', {}).get('graph_mode') != session.get('graph_mode'):
            issues.append('compiled graph mode identity differs')
        passed, reason = protocol.strict_correctness(session['model'], receipt.get('correctness', []))
        if not passed:
            issues.append('original numerical threshold failed: ' + reason)
        if receipt.get('protocol', {}).get('warmup') != 50:
            issues.append('P-DNN requires 50 warmups')
        windows.append(audit_windows(process, receipt, outer))
        rows.append(receipt)
    if any(w['issues'] for w in windows):
        issues.append('sampled PID window audit failed')
    measurements = [r for r in rows if r['process_role'] == 'measurement']
    if len(measurements) != 3 or any(len(r['samples_ms']) != 200 for r in measurements):
        issues.append('P-DNN requires three 200-sample process receipts')
    if sum(r['process_role'] == 'correctness' for r in rows) != 2:
        issues.append('P5 requires two additional correctness-only processes')
    identities = {json.dumps(r['identity'], sort_keys=True) for r in rows}
    if len(identities) != 1:
        issues.append('processes used different compiled identities')
    for key in ('library_sha256', 'compiled_receipt_sha256', 'inputs_sha256'):
        if any(not r.get(key) for r in rows) or len({r.get(key) for r in rows}) != 1:
            issues.append('processes used different or absent ' + key)
    timing = {}
    if not issues:
        medians = [r['median_ms'] for r in measurements]
        median = statistics.median(medians)
        timing = dict(median_ms=median, p90_ms=statistics.median(r['p90_ms'] for r in measurements),
                      process_range_ms=max(medians)-min(medians),
                      process_range_pct=100*(max(medians)-min(medians))/median)
    result = dict(status='pass' if not issues else 'quarantined', issues=issues,
                  model=session['model'], batch=session['batch'], system='aitemplate',
                  gpu=session['gpu'], precision='bf16', processes=len(rows),
                  graph_mode=session.get('graph_mode'),
                  graph_observations=[r.get('graph_mode') for r in rows],
                  correctness_passes=sum(r.get('correctness_passed', False) for r in rows),
                  correctness_policy='P-CORR-v2: original thresholds in every fresh process',
                  correctness=[r['correctness'] for r in rows], windows=windows,
                  source=[dict(path=p['receipt'], sha256=p['receipt_sha256']) for p in processes],
                  session_sha256=sha(options.session), outer_guard_sha256=sha(options.guard),
                  collector_sha256=sha(__file__), **timing)
    save(options.out, result)
    return int(bool(issues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    preparer = sub.add_parser('queue')
    preparer.add_argument('--model', choices=protocol.MODELS, required=True)
    preparer.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16], required=True)
    preparer.add_argument('--gpu', type=int, choices=sorted(AFFINITY), required=True)
    preparer.add_argument('--out', type=Path, required=True)
    preparer.add_argument('--graph', action=argparse.BooleanOptionalAction, default=True)
    runner = sub.add_parser('run')
    runner.add_argument('--cell', type=Path, required=True)
    runner.add_argument('--out', type=Path, required=True)
    collector = sub.add_parser('collect')
    collector.add_argument('--session', type=Path, required=True)
    collector.add_argument('--guard', type=Path, required=True)
    collector.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    return {'queue': queue, 'run': run, 'collect': collect}[options.command](options)


if __name__ == '__main__':
    raise SystemExit(main())
