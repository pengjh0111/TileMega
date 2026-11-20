#!/usr/bin/env python3
"""Plan non-overlapping A800 baseline cells and isolate each Slurm array task."""
import argparse
import json
from pathlib import Path

from a100_baselines import BATCHES, ENGINES, MODELS
from a800_baselines import RUN
from tilemega.bench.scheduler import write_json


def plan(kind, batch=None):
    completed = set()
    for path in (RUN/'cohorts').glob('*/baseline/state.json'):
        state = json.loads(path.read_text())
        completed.update(name for name, row in state.items() if row['status'] == 'done')
    entries = []
    for model in MODELS:
        large = model == 'qwen3_30b_a3b'
        if large != (kind == 'heavy'):
            continue
        for engine in ENGINES:
            prefix = f'{model}_{engine}'
            prior = BATCHES[BATCHES.index(batch) - 1] if batch else 1
            if not any(f'{prefix}_b{prior:02d}_r{i}' in completed for i in range(3)):
                continue
            batches = (batch,) if batch else (1, 2, 4, 8) if model == 'llama3_2_1b' else (1,)
            for measured_batch in batches:
                already_measured = any(f'{prefix}_b{measured_batch:02d}_r{i}' in completed
                                       for i in range(3))
                groups = [] if already_measured else [[0]]
                for rounds in groups:
                    group = ''.join(map(str, rounds))
                    wave = f'one-b{batch:02d}' if batch else 'one-b01'
                    cohort = (f'{wave}-{engine}-{model.replace("_", "-")}'
                              f'-b{measured_batch:02d}-r{group}')
                    options = ['--cohort', cohort, '--model', model, '--engine', engine,
                               '--batch', str(measured_batch),
                               '--process-timeout-s', '840' if large else '540' if batch else '330',
                               '--heartbeat-timeout-s', '420' if large else '300' if batch else '240']
                    if batch:
                        options.extend(('--expected-target', 'configs/targets/sm_80_a800_sxm4_80gb.json'))
                    for repeat in rounds:
                        options.extend(('--round', str(repeat)))
                    entries.append(dict(model=model, engine=engine, batch=measured_batch,
                                        rounds=rounds, options=options))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'task'))
    parser.add_argument('--kind', choices=('light', 'heavy'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--batch', choices=BATCHES[1:], type=int)
    args = parser.parse_args()
    if args.mode == 'plan':
        if not args.kind or args.manifest.exists():
            parser.error('choose a wave kind and a fresh manifest path')
        tasks = plan(args.kind, args.batch)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.manifest, tasks)
        print(json.dumps(dict(tasks=len(tasks), processes=sum(len(job['rounds']) for job in tasks))))
    else:
        tasks = json.loads(args.manifest.read_text())
        if args.index is None or not 0 <= args.index < len(tasks):
            parser.error('Slurm task index is outside the frozen manifest')
        print('\n'.join(tasks[args.index]['options']))


if __name__ == '__main__':
    main()
