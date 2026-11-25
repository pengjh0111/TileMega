#!/usr/bin/env python3
"""Bind precompiled R11 optimization arms into one screening allocation per cell."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import sha
from a800_r11_gate import read
from tilemega.bench.scheduler import write_json


def sources(scope):
    phases = {'pipeline': scope/'solver-release'}
    phases.update((name, scope/'experiments'/name) for name in ('boundary', 'mainloop', 'qwen'))
    by_cell = defaultdict(dict)
    for phase, folder in phases.items():
        for path in sorted(folder.glob('*/B*/finalists.plan.json')):
            value = read(path)
            cell = value['model'], value['batch']
            if phase in by_cell[cell]:
                raise ValueError(f'multiple {phase} plans for {cell}')
            by_cell[cell][phase] = path
    return by_cell


def referenced(waves):
    for ledger in sorted(waves.glob('*.rounds.json')):
        if ledger.with_name(ledger.name.replace('.rounds.json', '.id')).exists():
            for row in read(ledger)['cells']:
                path = Path(row['plan'])
                if sha(path) != row['sha256']:
                    raise ValueError(f'a submitted screen changed: {path}')
                yield path
                plan = read(path)
                manifest = plan_binding(plan)
                for spec in manifest.get('screen_sources', {}).values():
                    source = Path(spec['path'])
                    if sha(source) != spec['sha256']:
                        raise ValueError(f'a submitted optimization source changed: {source}')
                    yield source


def plan_binding(plan):
    path = Path(plan['finalists'])
    if sha(path) != plan['finalists_sha256']:
        raise ValueError(f'a finalist binding changed: {path}')
    return read(path)


def arms(binding, phase):
    """Return the candidates for one phase with their actual allocation ranks."""
    if 'phase_arms' in binding:
        return binding['phase_arms'][phase]
    return binding['candidates']


def arm_name(rank):
    return 'trial' if rank == 1 else f'rank-{rank}'


def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'retain the original screen binding: {path}')
    else:
        write_json(path, value)


def bind(out, phases):
    proof = {phase: dict(path=str(path), sha256=sha(path)) for phase, path in phases.items()}
    manifests = {phase: plan_binding(read(path)) for phase, path in phases.items()}
    first = next(iter(manifests.values()))
    common = ('model', 'batch', 'control_sha256', 'routing_export')
    control = read(first['control'])
    measured = {}
    labels = {}
    for phase, manifest in manifests.items():
        if (any(manifest[key] != first[key] for key in common)
                or sha(manifest['control']) != first['control_sha256']
                or sha(manifest['prepared']) != manifest['prepared_sha256']
                or sha(manifest['target']) != manifest['target_sha256']):
            raise ValueError(f'{phase} differs from the frozen cell inputs')
        labels[phase] = []
        for item in manifest['candidates']:
            candidate = dict(item, target=item.get('target', manifest['target']),
                             target_sha256=item.get('target_sha256', manifest['target_sha256']))
            digest = candidate['binary_sha256']
            if digest == control['decode_sha256']:
                raise ValueError(f'{phase} uses the control binary as a trial')
            existing = measured.get(digest)
            if existing is None:
                existing = dict(candidate, rank=len(measured)+1)
                measured[digest] = existing
            else:
                if any(existing[key] != candidate[key] for key in
                       ('target_sha256', 'coordinates', 'host_preparation_sha256')):
                    raise ValueError(f'{phase} binds incompatible inputs for the same binary')
                existing['route_check'] = existing.get('route_check') or candidate.get('route_check')
            labels[phase].append(dict(candidate, source_rank=candidate['rank'], rank=existing['rank']))
    if not 1 <= len(measured) <= 15:
        raise ValueError(f'one guarded allocation supports 1 to 15 trials, found {len(measured)}')
    out.mkdir(parents=True, exist_ok=True)
    preparation = out/'screen-sources.json'
    write_once(preparation, proof)
    value = {key: first[key] for key in (*common, 'control')}
    value.update(prepared=str(preparation), prepared_sha256=sha(preparation),
                 target=control['target'], target_sha256=control['target_sha256'],
                 candidates=list(measured.values()), phase_arms=labels, screen_sources=proof)
    write_once(out/'finalists.json', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('scope', 'out', 'release', 'worktree'):
        parser.add_argument(f'--{field}', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('bind screens on the login node with CUDA devices hidden')
    waves = args.scope/'campaign-waves'
    admitted = set(referenced(waves))
    launcher_env = dict(os.environ, PYTHONPATH=f'{args.worktree}/python:{args.worktree}/scripts')
    source = subprocess.check_output(['git', '-C', str(args.worktree), 'rev-parse', 'HEAD'],
                                     text=True, timeout=20).strip()
    ready = []
    errors = {}
    fixed_ready = (args.scope/'experiments.prepared').exists() and (args.scope/'qwen-router.prepared').exists()
    solver_finished = (args.scope/'solver-release.prepared').exists()
    for (model, batch), phases in sorted(sources(args.scope).items()):
        folder = args.out/model/f'B{batch}'
        plan = folder/'finalists.plan.json'
        if plan.exists():
            if read(plan)['source'] != source:
                raise ValueError(f'a screen was planned from a different worktree: {plan}')
            ready.append(str(plan))
        elif fixed_ready and ('pipeline' in phases or solver_finished):
            selected = {phase: path for phase, path in phases.items() if path not in admitted}
            if selected:
                try:
                    bind(folder, selected)
                    subprocess.run([sys.executable, str(args.release), 'plan', str(folder)],
                                   env=launcher_env, check=True, timeout=220)
                    if read(plan)['source'] != source:
                        raise ValueError(f'the launcher used a different worktree: {plan}')
                    ready.append(str(plan))
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    errors[f'{model}/B{batch}'] = str(error)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'status.json', dict(plans=ready, rejected=errors))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
