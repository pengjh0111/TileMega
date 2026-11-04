#!/usr/bin/env python3
"""H1 (prompt section 9.1): collect the measured peaks into SM90R6_hw_peaks.csv.

One item per process, 5 fresh processes per card per item, median reported.
Each process claims its card and records the SM and memory clocks and the
throttle reasons on both sides of the measurement, so a round that ran while
the card was clamped can be told apart from one that did not.
"""
import argparse, csv, fcntl, hashlib, json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sm90r5_dispatch as dispatch
from sm90d1_common import ROOT

OUT = ROOT / 'runs/sm90r6/h1'
BINARY = OUT / 'hw_peaks'
ITEMS = ('hbm_read', 'hbm_read_bulk', 'bf16_wgmma', 'bf16_mma_sync')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'scripts/sm90r6_hw_peaks.cu'
    command = ['nvcc', '-std=c++17', '-gencode=arch=compute_90a,code=sm_90a', '-O3',
               '-I', str(ROOT / 'include'), '-I', str(ROOT / 'third_party/cutlass/include'),
               '--expt-relaxed-constexpr', '-o', str(BINARY),
               str(ROOT / 'scripts/sm90r6_hw_peaks.cu')]
    with (OUT / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not BINARY.exists() or BINARY.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(command, check=True)
            (OUT / 'build_command.json').write_text(json.dumps(command) + '\n')
    return BINARY


def telemetry(gpu):
    query = ('clocks.sm,clocks.mem,temperature.gpu,power.draw,'
             'clocks_throttle_reasons.active')
    raw = subprocess.run(['nvidia-smi', '-i', str(gpu), f'--query-gpu={query}',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    parts = [p.strip() for p in raw.stdout.strip().split(',')]
    keys = ('sm_mhz', 'mem_mhz', 'temp_c', 'power_w', 'throttle_mask')
    return dict(zip(keys, parts))


def one(binary, gpu, item, gib, process, parallel_trial=False):
    """One measurement, through the guard as section 1.5 item 2 requires.

    The guard refuses a card that is occupied or drawing above its idle power,
    so a peak measured here is known to have had the card to itself.
    """
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       TILEMEGA_DEVICE_INDEX=str(gpu),
                       TILEMEGA_GPU_PID_MODE='namespace_limited',
                       TILEMEGA_TIMING='1', TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)),
                       PYTHONPATH=str(ROOT / 'python'))
    # The guard takes this lock itself and sets _HELD for its child, so the
    # path must be supplied and the held flag must not be pre-set.
    environment['TILEMEGA_GPU_LOCK'] = f'/tmp/tilemega_gpu{gpu}.lock'
    environment.pop('TILEMEGA_GPU_LOCK_HELD', None)
    environment.pop('TILEMEGA_DEVICE_GROUP', None)
    environment.pop('TILEMEGA_RUN_NONCE', None)
    environment.pop('TILEMEGA_NUMA_CONCURRENCY_VERIFIED', None)
    environment.pop('TILEMEGA_NUMA_CONCURRENCY_TRIAL', None)
    if parallel_trial:
        environment['TILEMEGA_NUMA_CONCURRENCY_TRIAL'] = '1'
    parent = OUT / f'gpu{gpu}/{item}'
    parent.mkdir(parents=True, exist_ok=True)
    guard_out = Path(tempfile.mkdtemp(prefix=f'p{process}-', dir=parent))
    node = '0' if gpu <= 3 else '1'
    command = [sys.executable, '-m', 'tilemega.bench.gpu_guard',
               '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'),
               '--out', str(guard_out), '--needs-free-mib', str(12 * 1024),
               '--timeout-s', '2400', '--', 'numactl', f'--cpunodebind={node}',
               str(binary), str(gib), item]
    result = subprocess.run(command, capture_output=True, text=True, env=environment,
                            cwd=str(ROOT), timeout=2700)
    (guard_out / 'stdout.txt').write_text(result.stdout)
    (guard_out / 'stderr.txt').write_text(result.stderr)
    (guard_out / 'command.json').write_text(json.dumps(command) + '\n')
    samples = [json.loads(line) for line in (guard_out / 'guard.jsonl').read_text().splitlines()]
    def mark(row):
        return dict(sm_mhz=str(row['sm_clock_mhz']), mem_mhz=str(row['memory_clock_mhz']),
                    throttle_mask=hex(row['throttle_mask']))
    before, after = mark(samples[0]), mark(samples[-1])
    (guard_out / 'telemetry.json').write_text(json.dumps(dict(before=before, after=after)) + '\n')
    if result.returncode != 0:
        return None, before, after, f'rc={result.returncode} {result.stderr[-160:]}'
    payload = next((line for line in result.stdout.splitlines()
                    if line.startswith('{"samples"')), '')
    if not payload:
        return None, before, after, 'no sample payload'
    parsed = json.loads(payload)
    (guard_out / 'samples.json').write_text(json.dumps(parsed, indent=2) + '\n')
    return parsed, before, after, ''


def run(a):
    binary = build()
    rows = []
    for gpu in a.gpus:
        # Untimed or not, every GPU job takes its card's lock (section 1.5).
        claimed, handle = dispatch.claim('exclusive', f'h1-gpu{gpu}', only=gpu)
        try:
            for item in a.items:
                per_config = {}
                marks = []
                units = {}
                failures = []
                for process in range(a.processes):
                    # A guard rejection is someone else's job arriving between
                    # the claim and the preflight, not a fault of this process,
                    # so it is retried rather than costing a sample.
                    for attempt in range(4):
                        payload, before, after, error = one(binary, claimed, item, a.gib, process,
                                                           a.parallel_trial)
                        if payload is not None or 'rc=75' not in error:
                            break
                        time.sleep(90)
                    if payload is None:
                        failures.append(error)
                        print(json.dumps({'gpu': claimed, 'item': item, 'process': process,
                                          'error': error}), flush=True)
                        continue
                    units = payload['units']
                    for sample in payload['samples']:
                        per_config.setdefault(sample['config'], []).append(sample['value'])
                    marks.append((before, after))
                for config, values in sorted(per_config.items()):
                    throttled = sorted({m[side]['throttle_mask'] for m in marks for side in (0, 1)})
                    rows.append(dict(
                        gpu=claimed, item=item, config=config, processes=len(values),
                        median=f'{statistics.median(values):.3f}',
                        minimum=f'{min(values):.3f}', maximum=f'{max(values):.3f}',
                        spread_pct=f'{(max(values) - min(values)) / statistics.median(values) * 100:.2f}',
                        unit=units.get(item.replace('_bulk', ''), ''),
                        failed_processes=len(failures),
                        sm_mhz=';'.join(sorted({m[0]['sm_mhz'] for m in marks})),
                        mem_mhz=';'.join(sorted({m[0]['mem_mhz'] for m in marks})),
                        sm_mhz_end=';'.join(sorted({m[1]['sm_mhz'] for m in marks})),
                        mem_mhz_end=';'.join(sorted({m[1]['mem_mhz'] for m in marks})),
                        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256((ROOT / 'scripts/sm90r6_hw_peaks.cu').read_bytes()).hexdigest(),
                        throttle_reasons=';'.join(throttled)))
                    print(json.dumps(rows[-1]), flush=True)
        finally:
            handle.close()
    if rows:
        fields = list(rows[0])
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = []
            if a.csv.exists():
                existing = [r for r in csv.DictReader(open(a.csv))
                            if (int(r['gpu']), r['item'], r['config']) not in
                            {(r2['gpu'], r2['item'], r2['config']) for r2 in rows}]
            temporary = a.csv.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(existing + rows)
            temporary.replace(a.csv)
        print(json.dumps({'csv': str(a.csv), 'rows': len(existing) + len(rows)}), flush=True)
    return 0 if rows and all(r['processes'] == a.processes and not r['failed_processes']
                            for r in rows) else 1


def main():
    global OUT, BINARY
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 5, 7])
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--gib', type=int, default=8)
    p.add_argument('--items', nargs='+', choices=ITEMS, default=list(ITEMS))
    p.add_argument('--parallel-trial', action='store_true',
                   help='user-authorized free-card concurrency; does not certify DT-P')
    p.add_argument('--out', type=Path, default=ROOT / 'runs/sm90r6/h1/protocol_v2')
    p.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R6_hw_peaks.csv')
    a = p.parse_args()
    if a.processes < 1 or a.gib < 1 or any(g not in (0, 1, 2, 3, 4, 5, 7) for g in a.gpus):
        p.error('positive processes/GiB and timing cards 0-5,7 are required')
    OUT = a.out.resolve(); BINARY = OUT / 'hw_peaks'
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
