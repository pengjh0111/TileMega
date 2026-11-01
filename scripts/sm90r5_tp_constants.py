#!/usr/bin/env python3
"""SM90R5 tensor parallel constants (B1, B9.1): run and collect sm90r5_tp_probe.

`run` takes a whole card group through the dispatcher's claim (machine class,
so nothing else times meanwhile), runs every section in its own fresh process
under the group-mode guard, `processes` times, and keeps each process's JSON
lines. `collect` takes the median over processes of each key's per-process
median, writes SM90R5_constants.csv rows and the target's
`sm90_native_constants.tp` section (ar_ns, head_x_ns, plus the B1 link terms).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, PYTHON, read, write
import sm90r5_dispatch as dispatch

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
OUT = ROOT / 'runs/sm90r5/tp/constants'
SECTIONS = ('--nvl', '--ar', '--head')
CARDS = {2: (0, 5, 7), 4: (0, 1, 2, 3, 4, 5, 7)}


def run(a):
    exclude = tuple(g for g in range(8) if g not in CARDS[a.tp])
    for section in a.sections:
        for fresh in range(a.processes):
            directory = OUT / f'tp{a.tp}' / section.strip('-') / f'f{fresh}'
            if (directory / 'probe.jsonl').exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            group, handles = dispatch.claim_group(f'tp{a.tp}{section}-f{fresh}', a.tp, exclude)
            try:
                visible = ','.join(map(str, group))
                nodes = sorted({dispatch.numa(g) for g in group})
                command = [PYTHON, '-m', 'tilemega.bench.gpu_guard', '--policy', str(POLICY),
                           '--out', str(directory / 'guard'), '--needs-free-mib', '16384',
                           '--timeout-s', '3600', '--', 'numactl',
                           f'--cpunodebind={",".join(map(str, nodes))}', str(a.binary.resolve()), section]
                env = dict(os.environ, TILEMEGA_DEVICE_GROUP=visible, CUDA_VISIBLE_DEVICES=visible,
                           TILEMEGA_DEVICE_INDEX=str(group[0]), TILEMEGA_TIMING='1',
                           TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'))
                for key in ('TILEMEGA_NUMA_NODE', 'TILEMEGA_GPU_LOCK', 'TILEMEGA_GPU_LOCK_HELD'):
                    env.pop(key, None)
                with (directory / 'stdout.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err:
                    code = subprocess.call(command, cwd=ROOT, env=env, stdout=out, stderr=err)
            finally:
                for handle in handles:
                    handle.close()
            if code == 0:
                (directory / 'stdout.jsonl').rename(directory / 'probe.jsonl')
            write(directory / 'result.json', dict(group=group, returncode=code, section=section))
            print(json.dumps(dict(tp=a.tp, section=section, fresh=fresh, group=group, returncode=code)),
                  flush=True)
    return 0


def collect(a):
    rows, table = [], {}
    for path in sorted(OUT.glob('tp*/*/f*/probe.jsonl')):
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                item = json.loads(line)
                table.setdefault((item['name'], item['key']), []).append(item['median_ns'])
    tp = dict(ar_ns={}, head_x_ns={}, nvl_store_ns={}, nvl_gbps={}, fence_sys_ns={})
    names = dict(t_ar='ar_ns', t_head_x='head_x_ns', nvl_store_ns='nvl_store_ns',
                 nvl_gbps='nvl_gbps', t_fence_sys='fence_sys_ns')
    for (name, key), values in sorted(table.items()):
        median = statistics.median(values)
        spread = (max(values) - min(values)) / abs(median) * 100 if median else 0.0
        rows.append(dict(name=name, key=key, processes=len(values), median=f'{median:.3f}',
                         unit='GB/s' if name == 'nvl_gbps' else 'ns', range_pct=f'{spread:.2f}',
                         per_process=';'.join(f'{v:.3f}' for v in values)))
        if name in names:
            # The model's t_ar key has no warp split; the probe names it so the
            # CSV keeps it, and the target takes it off.
            short = ':'.join(part for part in key.split(':') if not part.startswith('wk='))
            tp[names[name]][short] = round(max(median, 1e-3), 3)
    with open(a.csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if a.target:
        target = read(a.target)
        target['sm90_native_constants']['tp'] = dict(
            tp, method='scripts/sm90r5_tp_probe.cu; median over fresh processes of each '
                       'process median; t_ar and t_head_x are with-minus-without differences')
        write(a.target, target)
    print(json.dumps(dict(keys=len(rows), csv=str(a.csv))))
    return 0


def epilogue(a):
    """lm_head epilogue widths of a sharded vocabulary, one card, fresh processes."""
    exclude = tuple(g for g in range(8) if g not in (0, 5, 7))
    for fresh in range(a.processes):
        directory = OUT / 'lm_head' / f'f{fresh}'
        if (directory / 'probe.csv').exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        group, handles = dispatch.claim_group(f'lm-head-f{fresh}', 1, exclude)
        try:
            gpu = group[0]
            command = [PYTHON, '-m', 'tilemega.bench.gpu_guard', '--policy', str(POLICY),
                       '--out', str(directory / 'guard'), '--needs-free-mib', '16384',
                       '--timeout-s', '3600', '--', 'numactl', f'--cpunodebind={dispatch.numa(gpu)}',
                       str(a.binary.resolve()), '--lm-head-tp-only']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                       TILEMEGA_TIMING='1', TILEMEGA_GPU_PID_MODE='namespace_limited',
                       TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)),
                       TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', PYTHONPATH=str(ROOT / 'python'))
            env.pop('TILEMEGA_DEVICE_GROUP', None)
            with (directory / 'stdout.csv').open('w') as out, (directory / 'stderr.log').open('w') as err:
                code = subprocess.call(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        finally:
            for handle in handles:
                handle.close()
        if code == 0:
            (directory / 'stdout.csv').rename(directory / 'probe.csv')
        print(json.dumps(dict(section='lm_head', fresh=fresh, gpu=gpu, returncode=code)), flush=True)
    return 0


def collect_epilogue(a):
    values = {}
    files = sorted(OUT.glob('lm_head/f*/probe.csv'))
    for path in files:
        for row in csv.DictReader(line for line in path.read_text().splitlines()
                                  if line.startswith(('name,', 't_ep,'))):
            values.setdefault(row['key'], []).append(float(row['median']))
    target = read(a.target)
    table = target['sm90_native_constants']['epilogue_ns']['lm_head']
    rows = []
    for key, samples in sorted(values.items()):
        if len(samples) != len(files):
            raise ValueError(f'{key}: {len(samples)} of {len(files)} processes')
        fields = dict(part.split('=') for part in key.split(':') if '=' in part)
        median = statistics.median(samples)
        table.setdefault('wk' + fields['wk'], {}).setdefault(fields['tn'], {})[fields['b']] = round(median, 1)
        rows.append(dict(name='t_ep', key=key, processes=len(samples), median=f'{median:.3f}', unit='ns',
                         range_pct=f'{(max(samples) - min(samples)) / median * 100:.2f}',
                         per_process=';'.join(f'{v:.3f}' for v in samples)))
    write(a.target, target)
    existing = list(csv.DictReader(open(a.csv))) if a.csv.exists() else []
    keep = [r for r in existing if not (r['name'] == 't_ep' and r['key'].startswith('lm_head'))]
    with open(a.csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(keep + rows)
    print(json.dumps(dict(keys=len(rows), processes=len(files))))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    one = sub.add_parser('run')
    one.add_argument('--binary', type=Path, required=True)
    one.add_argument('--tp', type=int, choices=(2, 4), required=True)
    one.add_argument('--sections', nargs='+', default=list(SECTIONS))
    one.add_argument('--processes', type=int, default=5)
    two = sub.add_parser('collect')
    two.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R5_constants.csv')
    two.add_argument('--target', type=Path)
    three = sub.add_parser('epilogue')
    three.add_argument('--binary', type=Path, required=True)
    three.add_argument('--processes', type=int, default=5)
    four = sub.add_parser('collect-epilogue')
    four.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R5_constants.csv')
    four.add_argument('--target', type=Path, default=ROOT / 'configs/targets/sm_90_h800.json')
    a = p.parse_args()
    return dict(run=run, collect=collect, epilogue=epilogue,
                **{'collect-epilogue': collect_epilogue})[a.action](a)


if __name__ == '__main__':
    sys.exit(main())
