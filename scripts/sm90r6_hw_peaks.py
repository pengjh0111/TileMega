#!/usr/bin/env python3
"""H1 (prompt section 9.1): collect the measured peaks into SM90R6_hw_peaks.csv.

One item per process, 5 fresh processes per card per item, median reported.
Each process claims its card and records the SM and memory clocks and the
throttle reasons on both sides of the measurement, so a round that ran while
the card was clamped can be told apart from one that did not.
"""
import argparse, csv, json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sm90r5_dispatch as dispatch
from sm90d1_common import ROOT

OUT = ROOT / 'runs/sm90r6/h1'
BINARY = OUT / 'hw_peaks'
ITEMS = ('hbm_read', 'hbm_read_bulk', 'bf16_wgmma', 'bf16_mma_sync')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    # One job per card runs concurrently, so they must not all write the same
    # binary at once: compile only when it is older than its source.
    source = ROOT / 'scripts/sm90r6_hw_peaks.cu'
    if BINARY.exists() and BINARY.stat().st_mtime >= source.stat().st_mtime:
        return BINARY
    command = ['nvcc', '-std=c++17', '-gencode=arch=compute_90a,code=sm_90a', '-O3',
               '-I', str(ROOT / 'include'), '-I', str(ROOT / 'third_party/cutlass/include'),
               '--expt-relaxed-constexpr', '-o', str(BINARY),
               str(ROOT / 'scripts/sm90r6_hw_peaks.cu')]
    subprocess.run(command, check=True)
    return BINARY


def telemetry(gpu):
    query = ('clocks.sm,clocks.mem,temperature.gpu,power.draw,'
             'clocks_throttle_reasons.active')
    raw = subprocess.run(['nvidia-smi', '-i', str(gpu), f'--query-gpu={query}',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    parts = [p.strip() for p in raw.stdout.strip().split(',')]
    keys = ('sm_mhz', 'mem_mhz', 'temp_c', 'power_w', 'throttle_mask')
    return dict(zip(keys, parts))


def one(binary, gpu, item, gib, process):
    """One measurement, through the guard as section 1.5 item 2 requires.

    The guard refuses a card that is occupied or drawing above its idle power,
    so a peak measured here is known to have had the card to itself.
    """
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       TILEMEGA_DEVICE_INDEX=str(gpu),
                       TILEMEGA_GPU_PID_MODE='namespace_limited',
                       PYTHONPATH=str(ROOT / 'python'))
    # The guard takes this lock itself and sets _HELD for its child, so the
    # path must be supplied and the held flag must not be pre-set.
    environment['TILEMEGA_GPU_LOCK'] = f'/tmp/tilemega_gpu{gpu}.lock'
    environment.pop('TILEMEGA_GPU_LOCK_HELD', None)
    guard_out = OUT / f'gpu{gpu}/{item}/p{process}'
    guard_out.mkdir(parents=True, exist_ok=True)
    node = '0' if gpu <= 3 else '1'
    command = [sys.executable, '-m', 'tilemega.bench.gpu_guard',
               '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'),
               '--out', str(guard_out), '--needs-free-mib', str(12 * 1024),
               '--timeout-s', '2400', '--', 'numactl', f'--cpunodebind={node}',
               str(binary), str(gib), item]
    before = telemetry(gpu)
    result = subprocess.run(command, capture_output=True, text=True, env=environment,
                            cwd=str(ROOT), timeout=2700)
    after = telemetry(gpu)
    (guard_out / 'stdout.txt').write_text(result.stdout)
    if result.returncode != 0:
        return None, before, after, f'rc={result.returncode} {result.stderr[-160:]}'
    payload = next((line for line in result.stdout.splitlines()
                    if line.startswith('{"samples"')), '')
    if not payload:
        return None, before, after, 'no sample payload'
    return json.loads(payload), before, after, ''


def run(a):
    binary = build()
    rows = []
    for gpu in a.gpus:
        # Untimed or not, every GPU job takes its card's lock (section 1.5).
        claimed, handle = dispatch.claim('exclusive', f'h1-gpu{gpu}', only=gpu)
        try:
            for item in ITEMS:
                per_config = {}
                marks = []
                units = {}
                failures = []
                for process in range(a.processes):
                    # A guard rejection is someone else's job arriving between
                    # the claim and the preflight, not a fault of this process,
                    # so it is retried rather than costing a sample.
                    for attempt in range(4):
                        payload, before, after, error = one(binary, claimed, item, a.gib, process)
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
                        throttle_reasons=';'.join(throttled)))
                    print(json.dumps(rows[-1]), flush=True)
        finally:
            handle.close()
    if rows:
        existing = []
        if a.csv.exists():
            existing = [r for r in csv.DictReader(open(a.csv))
                        if (int(r['gpu']), r['item'], r['config']) not in
                        {(r2['gpu'], r2['item'], r2['config']) for r2 in rows}]
        fields = list(rows[0])
        with a.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing + rows)
        print(json.dumps({'csv': str(a.csv), 'rows': len(existing) + len(rows)}), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 5, 7])
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--gib', type=int, default=8)
    p.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R6_hw_peaks.csv')
    return run(p.parse_args())


if __name__ == '__main__':
    sys.exit(main())
