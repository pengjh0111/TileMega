#!/usr/bin/env python3
"""Required real-width seq=4 legacy-placement mechanism/arm ablation."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / 'realwidth'
sys.path.insert(0, str(REPO / 'docs/experiments/FENCE'))
import run as fence

C1 = ['-DTILEMEGA_RELEASE_AFTER_BARRIER=1']
C2 = C1 + ['-DTILEMEGA_ASYNC_PUBLISH=1']
CONFIGS = {'baseline': (1, []), 'c1': (1, C1), 'c2': (1, C2),
           'window2': (2, C2), 'local2': (2, C2+['-DTILEMEGA_LOCAL_DEP_SMEM=1'])}


def main():
    global RAW
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('build', 'measure'))
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--arch', choices=('sm_89','sm_120'), default='sm_89')
    args = parser.parse_args()
    RAW = args.raw.resolve()
    if any(k.startswith('TILEMEGA_') for k in os.environ):
        raise ValueError('refusing inherited TILEMEGA overrides')
    RAW.mkdir(parents=True, exist_ok=True)
    source = REPO / 'docs/experiments/REALMODEL/raw/work/r2sim_s4/model.cu'
    fixture = REPO / 'docs/experiments/REALMODEL/raw/work/r2sim_s4/export/fixture'
    if args.phase == 'build':
        free = shutil.disk_usage(RAW).free // 2**20
        print(f'DISK NEED_MIB=8192 FREE_MIB={free}', flush=True)
        if free < 8192:
            raise RuntimeError('insufficient disk before compilation')
        work = []
        for name, (window, flags) in CONFIGS.items():
            raw = RAW / name
            for folder in ('src', 'bin', 'log'):
                (raw / folder).mkdir(parents=True, exist_ok=True)
            if window > 1:
                subprocess.run(['python3', str(REPO / 'docs/experiments/WINDOW/plan_window.py'),
                                str(source), str(window), str(raw / 'src' / f'real_w{window}.cu')], check=True)
            for arm in fence.ARMS:
                work.append((('real', 0, arm), raw, flags, window))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda x: fence.build(x[0], x[1], x[2], arch=args.arch, window=x[3], source_override=source), work))
        return
    session = str(time.time_ns())
    pairs = [(name, arm) for name in CONFIGS for arm in fence.ARMS]
    for round_ in range(25):
        for order in range(len(pairs)):
            name, arm = pairs[(round_+order) % len(pairs)]
            raw = RAW / name
            binary = raw / 'bin' / f'real_p0_{arm}'
            directory = raw / 'paired/real_s4_p0' / arm
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / f'r{round_}.log'
            if log.exists():
                raise ValueError(f'refusing to overwrite {log}')
            cmd = [str(binary), str(fixture)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            output = result.stdout+result.stderr
            log.write_text(output)
            log.with_suffix('.json').write_text(json.dumps(dict(command=cmd, exit_code=result.returncode,
                session=session, round=round_, order=order, configuration=name,
                time_ns=time.time_ns(), binary_sha256=fence.sha(binary)))+'\n')
            if len(re.findall(r'^E2E_TIME ', output, re.M)) != 1:
                raise ValueError(f'missing timing: {log}')
            if arm == 'full' and (result.returncode or 'RESULT status=PASS' not in output):
                raise ValueError(f'full correctness failed: {log}')
        print(f'REAL_MATRIX s4 legacy round={round_+1}/25', flush=True)


if __name__ == '__main__':
    main()
