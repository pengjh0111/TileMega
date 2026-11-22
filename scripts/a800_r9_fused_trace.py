#!/usr/bin/env python3
"""Prepare matched fused decode traces and check their per-CTA task ledger."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

from a100_baselines import ROOT, sha
from tilemega.bench.scheduler import write_json
from tilemega.build.identity import generate, snapshot


def read(path):
    return json.loads(Path(path).read_text())


def sidecar(path, suffix):
    return Path(str(path)+suffix)


def prepare(args):
    anchor = read(args.anchor)
    original = Path(anchor['decode']).resolve(strict=True)
    identity = read(sidecar(original, '.identity.json'))
    plan = read(sidecar(original, '.plan.json'))
    if (sha(original) != anchor['decode_sha256'] or identity['binary_sha256'] != anchor['decode_sha256']
            or sha(sidecar(original, '.cu')) != identity['generated_source_sha256']
            or sha(sidecar(original, '.plan.json')) != identity['manifest_sha256']
            or identity['nvcc_arch'] != 'sm_80' or identity['trace']
            or not plan['native_dense']['expert_fuse']):
        raise ValueError('expected a frozen SM80 fused decode artifact')
    recorded = shlex.split(sidecar(original, '.build_command.txt').read_text())
    inputs = [word for word in recorded if word.endswith('.cu')]
    output = recorded[recorded.index('-o')+1]
    if len(inputs) != 1 or Path(output).resolve() != original:
        raise ValueError('ambiguous recorded build arguments')
    args.out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for label in ('control', 'trace'):
        library = args.out.resolve()/f'{label}.so'
        source = sidecar(library, '.source.json')
        if not sidecar(library, '.identity.json').exists():
            shutil.copyfile(sidecar(original, '.cu'), sidecar(library, '.cu'))
            shutil.copyfile(sidecar(original, '.plan.json'), sidecar(library, '.plan.json'))
            snapshot(ROOT, source)
            command = [str(library) if word == output else str(sidecar(library, '.cu'))
                       if word == inputs[0] else word.replace(identity['source']['root'], str(ROOT))
                       for word in recorded]
            if label == 'trace':
                command[1:1] = ['-DTILEMEGA_TRACE_STAGE=2', '-DTILEMEGA_TRACE_STEP=1',
                                '-DTILEMEGA_SM80_FUSED_PHASE_TRACE=1',
                                '-DTILEMEGA_SM80_FUSED_NEIGHBOR_TRACE=1',
                                '-DTILEMEGA_SM80_FUSED_PRICES_TRACE=1']
            sidecar(library, '.build_command.txt').write_text(shlex.join(command)+'\n')
            with sidecar(library, '.ptxas.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/run_sm90r1_build.py'),
                                '--receipt', str(sidecar(library, '.build_limits.json')),
                                '--max-gib', '48', '--timeout', '520',
                                '--lock', '/tmp/tilemega_sm90r2_native_build.lock', '--', *command],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=550)
            generate(library, ROOT, source)
        linked = read(sidecar(library, '.identity.json'))
        if (bool(linked['trace']) != (label == 'trace') or linked['binary_sha256'] != sha(library)
                or linked['generated_source_sha256'] != identity['generated_source_sha256']
                or linked['manifest_sha256'] != identity['manifest_sha256']):
            raise ValueError('linked fused trace changed the frozen program')
        arms[label] = dict(path=str(library), sha256=linked['binary_sha256'],
                           identity_sha256=sha(sidecar(library, '.identity.json')),
                           source_digest=linked['source']['source_digest'])
    if arms['control']['source_digest'] != arms['trace']['source_digest']:
        raise ValueError('fused trace arms must build from the same source')
    write_json(args.out/'trace.plan.json', dict(
        model=anchor['model'], batch=anchor['batch'], prefill=anchor['prefill'],
        target=anchor['artifact_target'], target_sha256=anchor['target_sha256'],
        original_sha256=anchor['decode_sha256'], grid=plan['grid'], arms=arms))


def analyze(args):
    plan = read(args.plan)
    report = read(args.run/'ledger/run.json')
    guard = read(args.run/'guard/guard_result.json')
    device = read(args.run/'job/physical-device-identity.json')
    if (guard['code'] or guard['acceptance'] != 'strict' or not guard['ownership_verified']
            or device['compute_capability'] != [8, 0]
            or report['decode_sha256'] != plan['arms']['trace']['sha256']
            or report['reference_sha256'] != plan['arms']['control']['sha256']
            or not report['full_token_equivalence'] or abs(report['overhead_pct']) > 5
            or report['batch'] != plan['batch']):
        raise ValueError('fused task attribution lacks a guarded matched trace within the 5% ceiling')
    linked = read(sidecar(plan['arms']['control']['path'], '.plan.json'))
    gates = {}
    for family in ('expert_gate_up', 'expert_down'):
        shape = next(row for row in linked['gemms'] if row['family'] == family)
        gates[family] = (shape['n']+shape['tile_n']-1)//shape['tile_n']
    ledger = args.run/'ledger'
    with (ledger/'runtime_stages.tsv').open() as stream:
        fused = {int(row['stage']) for row in csv.DictReader(stream, delimiter='\t')
                 if 'mlp.experts.gate_up' in row['name']}
    by_stage = defaultdict(list)
    with (ledger/'stage_trace.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            stage = int(row['stage'])
            if stage in fused and int(row['t_begin']) > 0:
                by_stage[int(row['iteration']), stage].append(row)
    if not by_stage:
        raise ValueError('trace contains no fused decode observations')
    split = Counter()
    ratios, tail = [], []
    grid = plan['grid']
    for rows in by_stage.values():
        if len(rows) != grid or {int(r['worker']) for r in rows} != set(range(grid)):
            raise ValueError('expected every CTA at each observed fused stage')
        counts = {int(row['active_experts']) for row in rows}
        if len(counts) != 1 or min(counts) <= 0:
            raise ValueError('active expert count differs across CTAs')
        active = counts.pop()
        durations = []
        for row in rows:
            worker, gu, dn = (int(row[k]) for k in ('worker', 'gu_tasks', 'dn_tasks'))
            gu_tasks = active*gates['expert_gate_up']
            total = gu_tasks+active*gates['expert_down']
            gu_exact = gu_tasks//grid+(worker<gu_tasks%grid)
            all_exact = total//grid+(worker<total%grid)
            if (gu, dn) != (gu_exact, all_exact-gu_exact):
                raise ValueError('fused stage task marks disagree with the slot assignment')
            split[gu, dn] += 1
            if gu+dn:
                durations.append(int(row['t_tasks_end'])-int(row['t_begin']))
        median = statistics.median(durations)
        if median <= 0 or min(durations) < 0:
            raise ValueError('invalid CTA duration for a fused stage')
        ratios.append(max(durations)/median)
        tail.append(max(durations)-median)
    write_json(args.out, dict(model=plan['model'], batch=plan['batch'],
        stages=len(by_stage), grid=grid, perturbation_pct=report['overhead_pct'],
        max_over_median_cta_span=statistics.median(ratios),
        worst_cta_span_ratio=max(ratios), median_lag_ns=statistics.median(tail),
        task_distribution=[dict(gate_up=gu,down=dn,samples=count)
                           for (gu,dn),count in sorted(split.items())],
        trace_sha256=report['decode_sha256'], control_sha256=report['reference_sha256'],
        checkpoint=report['checkpoint'], gpu_uuid=device['nvml_uuid']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest='mode', required=True)
    make = modes.add_parser('prepare')
    make.add_argument('--anchor', required=True, type=Path)
    collect = modes.add_parser('analyze')
    collect.add_argument('--run', required=True, type=Path)
    collect.add_argument('--plan', required=True, type=Path)
    for mode in (make,collect):
        mode.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    {'prepare': prepare, 'analyze': analyze}[args.mode](args)


if __name__ == '__main__':
    main()
