#!/usr/bin/env python3
"""Seal R13D geometry and create the eight-plan, two-toolchain G-REG queue."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from identity_dm import sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def handoff_option(value):
    # The manifest records the selected implementation; the CLI names its
    # access-proof selection policy. They are different vocabularies.
    if value == 'last_arriver':
        return 'auto'
    if value in ('off', 'auto'):
        return value
    raise ValueError('unsupported donor handoff: ' + str(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/dm1-regression')
    parser.add_argument('--queue-dir', type=Path, default=HERE / 'queue')
    parser.add_argument('--llm-python', default='/root/venvs/tilemega-torch213-cu126/bin/python')
    args = parser.parse_args()
    inputs = HERE / 'inputs/regression'
    exports = {}
    cache = Path('/root/.cache/tilemega-dm/exports/regression')
    for path in Path('/root/.cache/tilemega/exports').glob('*/manifest.json'):
        manifest = json.loads(path.read_text())
        model = manifest.get('config', {}).get('model_type')
        phase = manifest.get('phase')
        if model not in ('llama', 'qwen3') or phase not in ('prefill', 'decode'):
            continue
        destination = cache / model / phase
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('bridge.json', 'exported_program.pt2', 'manifest.json'):
            shutil.copyfile(path.parent / name, destination / name)
        exports[model, phase] = str(destination / 'bridge.json')
    recipes = []
    for model in ('llama', 'qwen3'):
        config = json.loads((ROOT / f'configs/e2e/{model}_r13_final.json').read_text())
        for batch in (1, 16):
            for phase in ('prefill', 'decode'):
                donor = Path(config['selected_plans'][str(batch)][phase])
                manifest = json.loads(Path(str(donor) + '.plan.json').read_text())
                folder = inputs / f'{model}_B{batch}' / phase
                write(folder / 'manifest.json', manifest)
                shutil.copyfile(str(donor) + '.classes.tsv', folder / 'classes.tsv')
                target = folder / 'target.json'
                shutil.copyfile(manifest['runtime_target'], target)
                overrides = {key: manifest[key] for key in (
                    'pg', 'sync', 'watchdog', 'l2_slim', 'nonpaged_weight_layout',
                    'page_loop_split', 'evict_first', 'evict_last', 'handoff',
                    'arch_paths', 'pdl', 'event_solo', 'event_red_publish', 'barrier_v2')}
                overrides.update(deferred_norm=int(manifest['deferred_norm']),
                                 paged_la=int(manifest['paged_la']),
                                 paged_la_splitk=int(manifest['paged_la_splitk']),
                                 v3_poll_ns=0, kphase_mask=31,
                                 weight_layout='tiled' if manifest['pg'] == 'pages' else 'row')
                overrides['handoff'] = handoff_option(manifest['handoff'])
                overrides = {k: int(v) if isinstance(v, bool) else v for k, v in overrides.items()}
                if manifest.get('pages'):
                    overrides.update(page_bytes=manifest['pages']['page_bytes'],
                                     lookahead_bytes=manifest['pages']['lookahead_bytes'])
                recipes.append(dict(model=model, batch=batch, phase=phase,
                                    export=exports[model, phase], target=str(target),
                                    manifest=str(folder / 'manifest.json'),
                                    classes=str(folder / 'classes.tsv'), overrides=overrides,
                                    donor=str(donor), donor_sha256=sha(donor)))
    write(inputs / 'seeds.json', recipes)
    queue = []
    lock = '/root/r14_work/gpu.lock'
    for side, root in [('reference', args.reference_root), ('candidate', ROOT)]:
        base = args.out / side
        def add(name, command, after, priority=20, timeout=28800, env=None):
            queue.append(dict(name=name, command=[str(v) for v in command], cwd=str(ROOT),
                              env=env or {}, gpu=False, after=after, priority=priority,
                              timeout_s=timeout, out=str(base / name)))
        native = 'DM0_native_' + side
        add(native, ['flock', lock, 'cmake', '--build', root / 'build-dm', '-j', '4'], [], 1)
        add('DM0_ctest_' + side,
            [sys.executable, HERE / 'run_ctest.py', '--root', root, '--out', base / 'ctest.json'],
            [native], 15)
        for recipe in recipes:
            cell = f"{recipe['model']}_B{recipe['batch']}"
            job = dict(recipe, root=str(root), out=str(base / cell / recipe['phase']))
            jobs = HERE / f"definitions/{side}_{cell}_{recipe['phase']}.json"
            write(jobs, [job])
            add(f"DM0_{side}_{cell}_{recipe['phase']}",
                [sys.executable, HERE / 'builds_dm.py', '--jobs', jobs,
                 '--out', base / cell / recipe['phase'] / 'builds.json'], [native])
        for model in ('llama', 'qwen3'):
            config = json.loads((ROOT / f'configs/e2e/{model}_r13_final.json').read_text())
            for batch in (1, 16):
                cell = f'{model}_B{batch}'
                add(f'DM0_tokens_{side}_{cell}',
                    ['flock', lock, args.llm_python, HERE / 'regression_tokens.py',
                     '--model', config['model']['path'], '--batch', batch,
                     '--prefill', base / cell / 'prefill/plan.so',
                     '--decode', base / cell / 'decode/plan.so',
                     '--prompts', ROOT / f'docs/experiments/SERVING_R10/prompts/{model}_ids.json',
                     '--out', base / cell / 'tokens.json'],
                    [f'DM0_{side}_{cell}_{phase}' for phase in ('prefill', 'decode')],
                    env=dict(PYTHONPATH=str(root / 'python')))
    dependencies = [s['name'] for s in queue]
    queue.append(dict(name='DM0_G_REG', command=[sys.executable, str(HERE / 'regression_compare.py'),
                      '--root', str(args.out), '--out', str(HERE / 'results/T1_phase0.json')],
                      gpu=False, after=dependencies, priority=50, timeout_s=600, cwd=str(ROOT)))
    write(args.queue_dir / 'queue_phase0_regression.json', queue)
    print('G-REG queue sealed: ' + str(len(queue)) + ' steps', flush=True)


if __name__ == '__main__':
    main()
