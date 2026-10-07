#!/usr/bin/env python3
"""Compare a candidate checkpoint to the sealed Phase 0 reference bank."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = ROOT / 'runs/dm1-regression/reference'
    (args.out / 'reference').symlink_to(reference, target_is_directory=True)
    queue = []

    def add(name, command, after=(), timeout=28800, env=None):
        queue.append(dict(name=args.checkpoint + '_' + name,
                          command=[str(x) for x in command], cwd=str(ROOT),
                          after=[args.checkpoint + '_' + x for x in after],
                          gpu=False, timeout_s=timeout, env=env or {}))

    add('compiler', ['flock', '/root/r14_work/gpu.lock', 'cmake', '--build',
                     ROOT / 'build-dm', '--target', 'tilemega', '-j', '4'])
    seeds = json.loads((HERE / 'inputs/regression/seeds.json').read_text())
    for recipe in seeds:
        cell = f"{recipe['model']}_B{recipe['batch']}"
        folder = args.out / 'candidate' / cell / recipe['phase']
        definition = args.out / 'definitions' / f"{cell}_{recipe['phase']}.json"
        definition.parent.mkdir(exist_ok=True)
        job = dict(recipe, root=str(ROOT), out=str(folder))
        definition.write_text(json.dumps([job], indent=2) + '\n')
        add(cell + '_' + recipe['phase'],
            [sys.executable, HERE / 'builds_dm.py', '--jobs', definition,
             '--out', folder / 'builds.json'], ['compiler'])
    for model in ('llama', 'qwen3'):
        config = json.loads((ROOT / f'configs/e2e/{model}_r13_final.json').read_text())
        for batch in (1, 16):
            cell = f'{model}_B{batch}'; folder = args.out / 'candidate' / cell
            add('tokens_' + cell,
                ['flock', '/root/r14_work/gpu.lock',
                 '/root/venvs/tilemega-torch213-cu126/bin/python', HERE / 'regression_tokens.py',
                 '--model', config['model']['path'], '--batch', batch,
                 '--prefill', folder / 'prefill/plan.so', '--decode', folder / 'decode/plan.so',
                 '--prompts', ROOT / f'docs/experiments/SERVING_R10/prompts/{model}_ids.json',
                 '--out', folder / 'tokens.json'],
                [cell + '_prefill', cell + '_decode'], env=dict(PYTHONPATH=str(ROOT / 'python')))
    dependencies = [job['name'].removeprefix(args.checkpoint + '_') for job in queue]
    add('compare', [sys.executable, HERE / 'regression_compare.py', '--root', args.out,
                    '--out', HERE / f'results/T1_{args.checkpoint}.json', '--partial'], dependencies)
    queue_dir = args.out / 'queue'; queue_dir.mkdir(exist_ok=True)
    (queue_dir / 'queue_regression.json').write_text(json.dumps(queue, indent=2) + '\n')
    print(f'{args.checkpoint}: {len(queue)} validation jobs prepared', flush=True)


if __name__ == '__main__':
    main()
