#!/usr/bin/env python3
"""Build and compare the frozen cost-aware chain recipe on bound local plans."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ('rotate', 'chain', 'chain_control', 'original')


def cells():
    return [(m, s) for m in ('gqa2', 'mha4', 'real') for s in (4, 128)]


def fixture(model, seq):
    if model == 'real':
        return REPO / f'docs/experiments/REALMODEL/raw/work/r2sim_s{seq}/export/fixture'
    return REPO / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(job, raw, arch):
    model, seq, arm = job
    if arm in ('chain', 'chain_control'):
        source = raw / ('on' if arm == 'chain' else 'off') / f'plan/{model}_s{seq}_chain.cu'
    elif arm == 'original':
        source = (raw / 'original' / f'plan/{model}_s{seq}_chain.cu' if arch == 'sm_120'
                  else REPO / f'docs/experiments/PLACE_EFT2/raw/plan/{model}_s{seq}_chain.cu')
    else:
        source = (REPO / f'docs/experiments/REALMODEL/raw/work/r2sim_s{seq}/model.cu' if model == 'real'
                  else REPO / f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu')
    key = f'{model}_s{seq}_{arm}'
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', f'-arch={arch}', '-lineinfo',
               '-DTILEMEGA_EVENT_KAPPA=1', f'-DTILEMEGA_PLACEMENT={5 if arm == "rotate" else 0}',
               *[f'-I{REPO / p}' for p in ('include', 'third_party/cutlass/include',
                 'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')],
               str(source), str(REPO / 'build-portable/libtilemega.a'), '-L/usr/local/cuda/lib64',
               '-lcudart', '-o', str(raw / 'bin' / key)]
    with (raw / 'log' / f'{key}.build.log').open('w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    result.check_returncode()
    (raw / 'log' / f'{key}.build.json').write_text(json.dumps(dict(command=command,
        source_sha256=sha(source), binary_sha256=sha(raw / 'bin' / key),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()), indent=2)+'\n')
    print(f'BUILT {key}', flush=True)


def run_one(raw, model, seq, arm, round_, folder, session, order):
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f'r{round_}.log'
    if log.exists():
        raise ValueError(f'refusing to overwrite {log}')
    binary = raw / 'bin' / f'{model}_s{seq}_{arm}'
    cmd = [str(binary), str(fixture(model, seq))]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    log.write_text(output)
    log.with_suffix('.json').write_text(json.dumps(dict(command=cmd, exit_code=result.returncode,
        session=session, round=round_, order=order, time_ns=time.time_ns(), binary_sha256=sha(binary)))+'\n')
    if result.returncode or re.findall(r'^RESULT status=(\S+)', output, re.M) != ['PASS']:
        raise ValueError(f'correctness failed: {log}')


def measure(raw, correctness=False):
    session = str(time.time_ns())
    for model, seq in cells():
        if correctness and model == 'real':
            continue
        count = 50 if correctness else 25
        arms = ('chain',) if correctness else ARMS
        for round_ in range(count):
            for order in range(len(arms)):
                arm = arms[(round_+order) % len(arms)]
                folder = raw / ('correctness' if correctness else 'paired') / f'{model}_s{seq}' / arm
                run_one(raw, model, seq, arm, round_, folder, session, order)
            if not correctness:
                print(f'CHAIN_PAIR {model} s{seq} round={round_+1}/25', flush=True)
        if correctness:
            print(f'CHAIN_CORRECT {model} s{seq} 50/50', flush=True)


def summarize(raw):
    rows = []
    rng = random.Random(167)
    for model, seq in cells():
        samples = {a: [] for a in ARMS}
        for arm in ARMS:
            for i in range(25):
                p = raw / 'paired' / f'{model}_s{seq}' / arm / f'r{i}.log'
                matches = re.findall(r'^E2E_TIME .*?l2_ms=([0-9.eE+-]+)', p.read_text(), re.M)
                if len(matches) != 1:
                    raise ValueError(f'{p}: invalid timing')
                samples[arm].append(float(matches[0]))
        for arm in ARMS:
            ratios = [a/b for a, b in zip(samples[arm], samples['rotate'])]
            boot = sorted(statistics.median(rng.choices(ratios, k=25)) for _ in range(10000))
            rows.append(dict(model=model, seq=seq, arm=arm, l2_ms=statistics.median(samples[arm]),
                             ratio=statistics.median(ratios), ci_low=boot[249], ci_high=boot[9749]))
    with (raw / 'measurements.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('build', 'correctness', 'measure', 'summarize'))
    parser.add_argument('--raw', type=Path, default=HERE / 'final')
    parser.add_argument('--arch', choices=('sm_89', 'sm_120'), default='sm_89')
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    if any(k.startswith('TILEMEGA_') for k in os.environ):
        raise ValueError('refusing inherited TILEMEGA_* overrides')
    raw = args.raw.resolve()
    for folder in ('bin', 'log'):
        (raw / folder).mkdir(parents=True, exist_ok=True)
    if args.phase == 'build':
        free = shutil.disk_usage(raw).free // 2**20
        print(f'DISK NEED_MIB=8192 FREE_MIB={free}', flush=True)
        if free < 8192:
            raise RuntimeError('insufficient disk before compilation')
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda job: build(job, raw, args.arch),
                          [(m, s, a) for m, s in cells() for a in ARMS]))
    elif args.phase in ('correctness', 'measure'):
        measure(raw, args.phase == 'correctness')
    else:
        summarize(raw)


if __name__ == '__main__':
    main()
