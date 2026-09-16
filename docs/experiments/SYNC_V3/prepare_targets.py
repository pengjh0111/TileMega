#!/usr/bin/env python3
"""Freeze six candidates' A ceilings from fresh, matching plan traces."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / 'targets_raw'
sys.path.insert(0, str(REPO / 'docs/experiments/TRACE_V2'))
import analyze

CANDIDATES = ('legacy_grid_stride', 'balanced', 'rotate', 'eft', 'wavefront', 'chain')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells():
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            for candidate in CANDIDATES:
                control = REPO / 'docs/experiments/PLAN_CONTRACT/legacy_identity/plan'
                source = (control / f'{model}.cu' if candidate in ('legacy_grid_stride', 'rotate')
                          else control / f'{model}_balanced.cu' if candidate == 'balanced'
                          else REPO / f'docs/experiments/PLACE_EFT2/raw/plan/{model}_s{seq}_{candidate}.cu')
                key = f'{model}_s{seq}_{candidate}'
                yield dict(model=model, seq=seq, candidate=candidate, source=str(source),
                           key=key, placement={'balanced': 4, 'rotate': 5}.get(candidate, 0))


def build(cell):
    key = cell['key']
    cmd = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '-arch=sm_89', '-lineinfo',
           '-DTILEMEGA_EVENT_KAPPA=1', '-DTILEMEGA_TRACE_V2=1',
           f'-DTILEMEGA_PLACEMENT={cell["placement"]}',
           *[f'-I{REPO / p}' for p in ('include', 'third_party/cutlass/include',
                                     'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')],
           cell['source'], str(REPO / 'build-portable/libtilemega.a'),
           '-L/usr/local/cuda/lib64', '-lcudart', '-o', str(RAW / 'bin' / key)]
    with (RAW / 'log' / f'{key}.build.log').open('w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    meta = dict(cell, command=cmd, exit_code=result.returncode,
                source_sha256=sha(Path(cell['source'])), head=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    if not result.returncode:
        meta['binary_sha256'] = sha(RAW / 'bin' / key)
    (RAW / 'log' / f'{key}.build.json').write_text(json.dumps(meta, indent=2)+'\n')
    result.check_returncode()
    print(f'BUILT {key}', flush=True)


def run(cell):
    key = cell['key']
    dump = RAW / 'dump' / key
    dump.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TILEMEGA_MODEL_NAME=cell['model'], TILEMEGA_TRACE_V2='1',
               TILEMEGA_TRACE_V2_OUT=str(dump))
    fixture = REPO / f'docs/experiments/SEQSCAN/raw/fixture/{cell["model"]}_s{cell["seq"]}_p3'
    cmd = [str(RAW / 'bin' / key), str(fixture)]
    with (RAW / 'log' / f'{key}.run.log').open('w') as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
    (RAW / 'log' / f'{key}.run.json').write_text(json.dumps(dict(
        command=cmd, exit_code=result.returncode, time_ns=time.time_ns(),
        binary_sha256=sha(Path(cmd[0]))), indent=2)+'\n')
    result.check_returncode()
    if 'RESULT status=PASS' not in (RAW / 'log' / f'{key}.run.log').read_text():
        raise ValueError(f'{key}: missing correctness PASS')
    print(f'TRACED {key}', flush=True)


def freeze():
    rows = []
    for c in cells():
        dump = RAW / 'dump' / c['key']
        r = analyze.analyze(dump, Path(c['source']), 1)
        floor = max(r['cp_corrected_ns'], r['queue_lb_ns']) / 1e6
        measured = r['measured_l2_ms']
        if measured < floor:
            raise ValueError(f'{c["key"]}: measured {measured} < floor {floor}')
        rows.append(dict(candidate=c['candidate'], model=c['model'], seq=c['seq'],
                         dump=str(dump.relative_to(REPO)), source=str(Path(c['source']).relative_to(REPO)),
                         floor_ms=f'{floor:.9f}', measured_A_ms=f'{measured:.9f}',
                         target_ms=f'{(floor+measured)/2:.9f}',
                         cp_corrected_ns=r['cp_corrected_ns'], queue_lb_ns=r['queue_lb_ns']))
        print(f'TARGET {c["key"]} floor={floor:.6f} measured={measured:.6f}', flush=True)
    target = HERE / 'targets.tsv'
    archive = RAW / 'invalid_targets_pre_resume.tsv'
    if not archive.exists():
        shutil.copyfile(target, archive)
    elif target.read_bytes() != archive.read_bytes():
        raise ValueError('targets already replaced: refusing to alter the frozen table')
    with target.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (RAW / 'frozen.json').write_text(json.dumps(dict(
        target_sha256=sha(target), rows=len(rows), measured_A_definition='matching configuration A trace meta',
        inputs={str(p.relative_to(REPO)): sha(p) for p in sorted((RAW / 'dump').rglob('*.tsv'))},
        supersedes='d549c5a8: configurations were mislabelled as candidates; legacy bounds and unrelated measured_A',
        authorization='User clarified stop conditions are local and known prompt errors may be repaired with an audit trail.'
    ), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('build', 'run', 'freeze'))
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    for sub in ('bin', 'log', 'dump'):
        (RAW / sub).mkdir(parents=True, exist_ok=True)
    if args.phase == 'build':
        need_mib = 4096
        free_mib = shutil.disk_usage(RAW).free // 2**20
        print(f'DISK NEED_MIB={need_mib} FREE_MIB={free_mib}', flush=True)
        if free_mib < need_mib:
            raise RuntimeError('insufficient disk before compilation')
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(build, cells()))
    elif args.phase == 'run':
        for cell in cells():
            run(cell)
    else:
        freeze()


if __name__ == '__main__':
    main()
