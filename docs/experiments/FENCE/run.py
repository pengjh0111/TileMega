#!/usr/bin/env python3
"""R4 five-arm measurements. Unsafe arms supply timings, never correctness."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ('full', 'nofence', 'nowait', 'neither', 'l1nosync')
FLAGS = {'full': [], 'nofence': ['-DTILEMEGA_UNSAFE_NO_NOTIFY_FENCE=1'],
         'nowait': ['-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1'],
         'neither': ['-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1', '-DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1'],
         'l1nosync': ['-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1']}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs(arms=ARMS, placements=(0, 5)):
    for model in ('gqa2', 'mha4'):
        for placement in placements:
            for arm in arms:
                yield model, placement, arm


def build(job, raw, extra, arch='sm_89', window=1, kappa=1, source_override=None):
    model, placement, arm = job
    key = f'{model}_p{placement}_{arm}'
    source = source_override or REPO / f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu'
    if window > 1:
        source = raw / 'src' / f'{model}_w{window}.cu'
    policy = shlex.split(subprocess.check_output([str(REPO / 'build-portable/tools/tilemega-wait-policy'),
                         arch, 'bf16', str(REPO)], text=True, stderr=subprocess.DEVNULL))
    cmd = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', f'-arch={arch}', '-lineinfo',
           f'-DTILEMEGA_EVENT_KAPPA={kappa}', '-DTILEMEGA_BARRIER_V2=1', '-DTILEMEGA_EVENT_SOLO=1',
           '-DTILEMEGA_EVENT_RED_PUBLISH=1', *policy, f'-DTILEMEGA_PLACEMENT={placement}',
           f'-DTILEMEGA_SLOT_WINDOW={window}', *extra, *FLAGS[arm],
           *[f'-I{REPO / p}' for p in ('include', 'third_party/cutlass/include',
                                     'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')],
           str(source), str(REPO / 'build-portable/libtilemega.a'), '-L/usr/local/cuda/lib64',
           '-lcudart', '-o', str(raw / 'bin' / key)]
    with (raw / 'log' / f'{key}.build.log').open('w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    meta = dict(command=cmd, exit_code=result.returncode, source_sha256=sha(source),
                head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    if not result.returncode:
        meta['binary_sha256'] = sha(raw / 'bin' / key)
    (raw / 'log' / f'{key}.build.json').write_text(json.dumps(meta, indent=2)+'\n')
    result.check_returncode()
    print(f'BUILT {key}', flush=True)


def measure(raw):
    session = str(time.time_ns())
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            for round_ in range(25):
                for index in range(10):
                    # Rotate all ten placement/arm combinations in one session.
                    placement, arm = [(p, a) for p in (0, 5) for a in ARMS][(round_+index) % 10]
                    key = f'{model}_p{placement}_{arm}'
                    out = raw / 'paired' / f'{model}_s{seq}_p{placement}' / arm
                    out.mkdir(parents=True, exist_ok=True)
                    cmd = [str(raw / 'bin' / key), str(REPO / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3')]
                    log = out / f'r{round_}.log'
                    if log.exists():
                        raise ValueError(f'refusing to overwrite a prior session: {log}')
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    text = result.stdout + result.stderr
                    log.write_text(text)
                    (out / f'r{round_}.json').write_text(json.dumps(dict(command=cmd,
                        exit_code=result.returncode, session=session, round=round_, order=index,
                        time_ns=time.time_ns(), binary_sha256=sha(Path(cmd[0]))))+'\n')
                    # Numerical mismatch is expected in unsafe arms. Missing
                    # timing, launch errors, and timeouts are never samples.
                    if len(re.findall(r'^E2E_TIME ', text, re.M)) != 1:
                        raise ValueError(f'missing timing: {log}')
                    if arm == 'full' and (result.returncode or 'RESULT status=PASS' not in text):
                        raise ValueError(f'full correctness failed: {log}')
                print(f'PAIRED {model} s{seq} round={round_+1}/25', flush=True)


def timing(path):
    lines = re.findall(r'^E2E_TIME (.+)$', path.read_text(), re.M)
    if len(lines) != 1:
        raise ValueError(f'missing/duplicate timing: {path}')
    return {k: float(v) for k, v in re.findall(r'(\w+)=([0-9.eE+-]+)', lines[0])}


def correctness(raw, subset=False):
    cells = ((1, 0), (128, 512), (2048, 0)) if subset else ((4, 3), (128, 3))
    for model in ('gqa2', 'mha4'):
        for seq, past in cells:
            folder = raw / ('seqscan' if subset else 'correctness') / f'{model}_s{seq}_p{past}'
            folder.mkdir(parents=True, exist_ok=True)
            cmd = [str(raw / 'bin' / f'{model}_p0_full'),
                   str(REPO / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p{past}')]
            passes = 0
            for round_ in range(50):
                log = folder / f'r{round_}.log'
                if log.exists():
                    raise ValueError(f'refusing to overwrite {log}')
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                output = result.stdout+result.stderr
                log.write_text(output)
                (folder / f'r{round_}.json').write_text(json.dumps(dict(command=cmd,
                    exit_code=result.returncode, round=round_, time_ns=time.time_ns(),
                    binary_sha256=sha(Path(cmd[0]))))+'\n')
                passes += result.returncode == 0 and 'RESULT status=PASS' in output
            print(f'CORRECT {model} s{seq} p{past} {passes}/50', flush=True)
            if passes != 50:
                raise ValueError(f'{folder}: correctness gate failed')


def sass(raw):
    out = raw / 'sass'
    out.mkdir(exist_ok=True)
    rows = []
    for model in ('gqa2', 'mha4'):
        for placement in (0, 5):
            key = f'{model}_p{placement}_full'
            binary = raw / 'bin' / key
            with (out / f'{key}.sass').open('w') as f:
                subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump', '--dump-sass', str(binary)], stdout=f, check=True)
            with (out / f'{key}.report.txt').open('w') as f:
                subprocess.run(['bash', str(REPO / 'scripts/sass_report.sh'), str(binary)], stdout=f, check=True)
            for chunk in (out / f'{key}.sass').read_text().split('Function : ')[1:]:
                name = chunk.splitlines()[0].strip()
                if 'tilemega_l2_kernel' in name:
                    rows.append(dict(model=model, placement=placement, kernel=name,
                        membar_sc_gpu=len(re.findall(r'\bMEMBAR\.SC\.GPU', chunk)),
                        membar=len(re.findall(r'\bMEMBAR', chunk)),
                        bar_sync=len(re.findall(r'\bBAR\.SYNC', chunk)), binary_sha256=sha(binary)))
    with (out / 'census.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    rng = random.Random(167)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return statistics.median(values), boot[249], boot[9749]


def summarize(raw):
    rows = []
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            for placement in (0, 5):
                cell = raw / 'paired' / f'{model}_s{seq}_p{placement}'
                samples = {k: [] for k in ('l2_ms', 'l1_ms', 'fence_ms', 'wait_ms', 'notify_ms',
                    'barrier_ms', 'protocol_ms', 'fence_over_notify', 'fence_over_protocol',
                    'protocol_over_barrier', 'l2_over_l1')}
                for round_ in range(25):
                    t = {a: timing(cell / a / f'r{round_}.log') for a in ARMS}
                    full, nf, nw, neither = [t[a]['l2_ms'] for a in ARMS[:4]]
                    fence, notify, protocol = full-nf, nw-neither, full-neither
                    barrier = t['full']['l1_ms']-t['l1nosync']['l1_ms']
                    values = (full, t['full']['l1_ms'], fence, full-nw, notify, barrier, protocol,
                              fence/notify, fence/protocol, protocol/barrier, full/t['full']['l1_ms'])
                    for key, value in zip(samples, values):
                        samples[key].append(value)
                row = dict(model=model, seq=seq, placement=placement, pairs=25)
                for metric, values in samples.items():
                    median, low, high = interval(values)
                    row.update({metric: median, metric+'_ci_low': low, metric+'_ci_high': high})
                rows.append(row)
                print(f'FENCE {model} s{seq} p{placement}: {row["fence_ms"]*1000:.3f} us; '
                      f'notify share={row["fence_over_notify"]:.4f}; protocol/barrier='
                      f'{row["protocol_over_barrier"]:.4f}', flush=True)
    with (raw / 'measurements.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('build', 'measure', 'summarize', 'correctness', 'seqscan', 'sass'))
    parser.add_argument('--raw', type=Path, default=HERE / 'raw')
    parser.add_argument('--extra-flags', default='')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--arch', choices=('sm_89', 'sm_120'), default='sm_89')
    parser.add_argument('--window', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--kappa', type=int, default=1)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--placements', nargs='+', type=int, choices=(0, 5), default=(0, 5))
    args = parser.parse_args()
    inherited = [k for k in os.environ if k.startswith('TILEMEGA_')]
    if inherited:
        raise ValueError(f'refusing inherited TILEMEGA variables: {inherited}')
    raw = args.raw.resolve()
    for sub in ('bin', 'log'):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    if args.phase == 'build':
        free = shutil.disk_usage(raw).free // 2**20
        print(f'DISK NEED_MIB=4096 FREE_MIB={free}', flush=True)
        if free < 4096:
            raise RuntimeError('insufficient disk before compilation')
        if args.window > 1:
            (raw / 'src').mkdir(exist_ok=True)
            for model in ('gqa2', 'mha4'):
                subprocess.run(['python3', str(REPO / 'docs/experiments/WINDOW/plan_window.py'),
                    str(REPO / f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu'),
                    str(args.window), str(raw / 'src' / f'{model}_w{args.window}.cu')], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda job: build(job, raw, shlex.split(args.extra_flags), args.arch, args.window,
                                           args.kappa), jobs(args.arms, args.placements)))
    elif args.phase == 'measure':
        measure(raw)
    elif args.phase in ('correctness', 'seqscan'):
        correctness(raw, args.phase == 'seqscan')
    elif args.phase == 'sass':
        sass(raw)
    else:
        summarize(raw)


if __name__ == '__main__':
    main()
