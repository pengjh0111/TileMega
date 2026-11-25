#!/usr/bin/env python3
"""Bind eligible phase winners for the final per-cell comparison against R10."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import subprocess

from a100_baselines import inputs as model_inputs, inspect_inputs, sha
from a800_baselines import INVENTORY
from a800_native import artifact
from a800_r10_prepare import build, schema10
from a800_r11_collect_round import collect
from a800_r11_gate import read
from a800_r11_prepare import router_changed
from a800_r11_screen import arm_name, arms, plan_binding
from a800_shortlist_abi import freeze_host_preparation
from tilemega.bench.scheduler import write_json


def completed(path):
    accounting = read(path/'accounting.json')
    for ledger in sorted(path.glob('*.rounds.json')):
        ticket = ledger.with_name(ledger.name.replace('.rounds.json', '.id'))
        if not ticket.exists():
            continue
        parent = ticket.read_text().strip()
        for index, item in enumerate(read(ledger)['cells']):
            outcome = accounting.get(parent, {}).get(str(index), {})
            if outcome.get('state') not in ('COMPLETED', 'FAILED'):
                continue
            if sha(item['plan']) != item['sha256']:
                raise ValueError('a measured cell changed its allocation plan')
            plan = read(item['plan'])
            phase = ledger.name.split('-')[0]
            phases = plan_binding(plan)['phase_arms'] if phase == 'screen' else (phase,)
            for name in phases:
                yield name, plan, outcome['job']


def select(waves, runs, inputs, selected=None):
    cells = defaultdict(dict)
    for phase, plan, job in completed(waves):
        if phase == 'final' or (selected is not None and (plan['model'], plan['batch']) not in selected):
            continue
        folder = runs/'optimization/decode-pairs'/job
        if not (folder/'result.json').exists():
            continue
        manifest = plan_binding(plan)
        named = {arm_name(row['rank']): row for row in arms(manifest, phase)}
        observations = [fact for fact, data, _ in collect(runs, job, inputs)
                        if data is not None and fact['arm'] in named]
        if not observations:
            continue
        # Ranking here only compares arms in this allocation. The surviving
        # arms from different phases will meet in the final single-GPU job.
        fact = min(observations, key=lambda row: row['tpot_ms'])
        chosen = named[fact['arm']]
        if chosen['binary_sha256'] != fact['decode_sha256']:
            raise ValueError('the surviving trial differs from the measured binary')
        cell = fact['model'], fact['batch']
        if phase in cells[cell]:
            raise ValueError(f'multiple terminal jobs for {cell} {phase}; supply one accepted allocation')
        cells[cell][phase] = dict(chosen, job=job, evidence_sha256=fact['evidence'],
            target=chosen.get('target', manifest['target']),
            target_sha256=chosen.get('target_sha256', manifest['target_sha256']),
            passed_screen=fact['status'] == 'faster')
    return cells


MISSING = object()


def combine(control, variants):
    base = schema10(deepcopy(control))
    merged = deepcopy(base)
    for variant in variants:
        trial = schema10(deepcopy(variant))
        for domain in (None, 'gemms'):
            old, new, destination = ((base, trial, merged) if domain is None else
                                     (base[domain], trial[domain], merged[domain]))
            for key in sorted(old.keys() | new.keys()):
                if domain is None and key == 'gemms':
                    continue
                before, after = old.get(key, MISSING), new.get(key, MISSING)
                if before == after:
                    continue
                current = destination.get(key, MISSING)
                if current != before and current != after:
                    raise ValueError(f'overlapping coordinate edits at {domain+"." if domain else ""}{key}')
                if after is MISSING:
                    destination.pop(key, None)
                else:
                    destination[key] = deepcopy(after)
    return merged


def combined_arms(out, control, phases, cli, failures):
    fixed = {name:row for name,row in phases.items() if name!='pipeline' and row['passed_screen']}
    solver = phases.get('pipeline')
    groups = [('combined-fixed', fixed)]
    if solver and solver['passed_screen']:
        groups.append(('combined-solver', dict(fixed, pipeline=solver)))
    for label, members in groups:
        if len(members)<2:
            continue
        try:
            coords = combine(control['coordinates'], [row['coordinates'] for row in members.values()])
            if any(coords==schema10(deepcopy(row['coordinates'])) for row in members.values()):
                continue
            target = members['pipeline']['target'] if 'pipeline' in members else control['target']
            args = argparse.Namespace(cli=cli, batch=control['batch'])
            library = build(args, Path(control['export']), target, out, label, coords)
            binary, identity, manifest = artifact(library)
            prefill = () if control['prefill']=='prompt' else (Path(control['prefill']),)
            packing = binary.with_suffix('.host-preparation.json')
            config = read(model_inputs(control['model'])[0]/'config.json')
            freeze_host_preparation(config, control['batch'], (*prefill,binary), packing)
            target_sha = sha(target)
            if identity['target_sha256'] != target_sha:
                raise ValueError('combined artifact changed its bound target')
            yield dict(label=label, binary=str(binary), binary_sha256=identity['binary_sha256'],
                       coordinates=manifest['native_dense'], target=target, target_sha256=target_sha,
                       host_preparation=str(packing), host_preparation_sha256=sha(packing),
                       route_check=router_changed(control['coordinates'], manifest['native_dense']),
                       combined_from={name: row['binary_sha256'] for name,row in members.items()})
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            failures[label] = str(error)


def bind(out, incumbent, phases, cli=None):
    control = read(incumbent)
    chosen = [row for name, row in sorted(phases.items()) if row['passed_screen'] or name == 'recovered']
    if not chosen and 'pipeline' in phases:
        # Retest the best eligible solver finalist against the incumbent even
        # when the first timing found a tie or loss.
        chosen = [phases['pipeline']]
    out.mkdir(parents=True, exist_ok=True)
    failures = {}
    if cli:
        chosen.extend(combined_arms(out, control, phases, cli, failures))
    rows, seen = [], {control['decode_sha256']}
    for row in chosen:
        digest = row['binary_sha256']
        if digest not in seen:
            seen.add(digest)
            rows.append(dict(row, rank=len(rows)+1))
    if not rows:
        raise ValueError('no distinct numeric-gate trial is available for the final paired test')
    scope = out/'prior-allocations.json'
    write_json(scope, dict(phases=phases, combination_rejections=failures))
    write_json(out/'finalists.json', dict(model=control['model'], batch=control['batch'],
        control=str(incumbent), control_sha256=sha(incumbent), prepared=str(scope),
        prepared_sha256=sha(scope), target=control['target'], target_sha256=control['target_sha256'],
        candidates=rows, routing_export=control['export']))


def recovered(legacy, incumbents, preparation, inputs, selected=None):
    runs = legacy.parents[1]
    accounting = read(legacy/'legacy.accounting.json')
    manifest = read(legacy/'legacy.rounds.json')
    for index, item in enumerate(manifest['items']):
        allocation = accounting.get(str(index), {})
        job = allocation.get('job')
        if not job or not (runs/'optimization/decode-pairs'/job/'result.json').exists():
            continue
        if sha(item['binding']) != item['sha256']:
            raise ValueError('the recovered R10 trial changed after submission')
        spec = read(item['binding'])
        args = spec['args']
        cell = args[args.index('--model')+1], int(args[args.index('--batch')+1])
        if selected is not None and cell not in selected:
            continue
        (fact, timing, _), = collect(runs, job, inputs)
        if timing is None or fact['passed_processes'] != 5:
            continue
        model, batch = fact['model'], fact['batch']
        control = read(incumbents/model/f'B{batch}'/'incumbent.json')
        path = Path(args[args.index('--trial')+1])
        _, identity, library_plan = artifact(path)
        if identity['binary_sha256'] != fact['decode_sha256']:
            raise ValueError('the recovered trial does not match its fresh numeric receipt')
        if '--trial-preparation' in args:
            host_preparation = Path(args[args.index('--trial-preparation')+1])
        else:
            folder = preparation/model/f'B{batch}'
            folder.mkdir(parents=True, exist_ok=True)
            host_preparation = folder/'recovered.host-preparation.json'
            prefix = () if control['prefill'] == 'prompt' else (Path(control['prefill']),)
            freeze_host_preparation(read(model_inputs(model)[0]/'config.json'), batch,
                                    (*prefix, path), host_preparation)
        coords = library_plan['native_dense']
        yield (model,batch), dict(label='recovered-r10', binary=str(path), binary_sha256=fact['decode_sha256'],
            coordinates=coords, host_preparation=str(host_preparation),
            host_preparation_sha256=sha(host_preparation),
            route_check=router_changed(control['coordinates'], coords), target=library_plan['runtime_target'],
            target_sha256=identity['target_sha256'], job=job,
            evidence_sha256=fact['evidence'], passed_screen=fact['status'] == 'faster')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('waves', 'runs', 'incumbents', 'out'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    parser.add_argument('--legacy', type=Path, help='include revalidated R10 arms in the final comparison')
    parser.add_argument('--cli', type=Path, help='link compatible combinations of measured winners before retesting')
    parser.add_argument('--cell', nargs=2, metavar=('MODEL', 'BATCH'), action='append',
                        help='prepare only a cell whose screening allocations are terminal')
    args = parser.parse_args()
    inputs = inspect_inputs(INVENTORY)
    selected = None if args.cell is None else {(model,int(batch)) for model,batch in args.cell}
    cells = select(args.waves, args.runs, inputs, selected)
    if args.legacy:
        for cell, trial in recovered(args.legacy, args.incumbents, args.out, inputs, selected):
            cells[cell]['recovered'] = trial
    rejected = {}
    for (model, batch), phases in sorted(cells.items()):
        try:
            bind(args.out/model/f'B{batch}', args.incumbents/model/f'B{batch}'/'incumbent.json', phases, args.cli)
        except ValueError as error:
            rejected[f'{model}/B{batch}'] = str(error)
    if selected is not None:
        for model,batch in selected:
            status = args.out/model/f'B{batch}'/'binding-status.json'
            status.parent.mkdir(parents=True, exist_ok=True)
            reason = rejected.get(f'{model}/B{batch}', '' if (model,batch) in cells else 'no usable screen result')
            write_json(status, dict(ready=not reason, reason=reason))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'final-binding-rejections.json', rejected)


if __name__ == '__main__':
    main()
