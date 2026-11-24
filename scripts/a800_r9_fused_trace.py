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
from tilemega.build.identity import digest, generate, snapshot


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
            or (not args.release_ledger and not args.stage_overlap and not plan['native_dense']['expert_fuse'])):
        raise ValueError('expected a frozen SM80 decode artifact compatible with the trace')
    recorded = shlex.split(sidecar(original, '.build_command.txt').read_text())
    if '--split-compile' in recorded:
        recorded[recorded.index('--split-compile')+1] = '1'
    recorded = ['--split-compile=1' if word.startswith('--split-compile=') else word
                for word in recorded]
    inputs = [word for word in recorded if word.endswith('.cu')]
    output = recorded[recorded.index('-o')+1]
    if len(inputs) != 1 or Path(output).resolve() != original:
        raise ValueError('ambiguous recorded build arguments')
    args.out.mkdir(parents=True, exist_ok=True)
    arms = {}
    build_root = ROOT
    if args.reuse_exact_control:
        archived = Path(identity['source']['root'])
        program = archived/'include/tilemega/Codegen/tasks/Sm90DenseProgram.cuh'
        if program.is_file() and 'TILEMEGA_NATIVE_RELEASE_SHARED_BUDGET' in program.read_text():
            proof = snapshot(archived, args.out/'archived-source.json')
            if proof['source_digest'] == identity['source']['source_digest']:
                build_root = archived
                arms['control'] = dict(path=str(original), sha256=identity['binary_sha256'],
                    identity_sha256=sha(sidecar(original, '.identity.json')), source_digest=proof['source_digest'])
    for label in ('control', 'trace'):
        if label in arms:
            continue
        library = args.out.resolve()/f'{label}.so'
        source = sidecar(library, '.source.json')
        if not sidecar(library, '.identity.json').exists():
            shutil.copyfile(sidecar(original, '.cu'), sidecar(library, '.cu'))
            shutil.copyfile(sidecar(original, '.plan.json'), sidecar(library, '.plan.json'))
            snapshot(build_root, source)
            command = [str(library) if word == output else str(sidecar(library, '.cu'))
                       if word == inputs[0] else word.replace(identity['source']['root'], str(build_root))
                       for word in recorded]
            if label == 'trace':
                if args.release_ledger:
                    limits = read(anchor['artifact_target'])['resources']
                    budget = min(limits['max_dynamic_smem_per_cta'],
                                 limits['max_smem_per_sm']//plan['native_dense']['residency'])
                    command[1:1] = ['-DTILEMEGA_TRACE_NATIVE_RELEASE=1',
                                    f'-DTILEMEGA_NATIVE_RELEASE_SHARED_BUDGET={budget}']
                elif args.stage_overlap:
                    command[1:1] = ['-DTILEMEGA_TRACE_STAGE=1', '-DTILEMEGA_TRACE_STEP=1',
                                    '-DTILEMEGA_SM80_STAGE_TRACE_SAMPLE_PERIOD=4']
                else:
                    command[1:1] = ['-DTILEMEGA_TRACE_STAGE=2', '-DTILEMEGA_TRACE_STEP=1',
                                    '-DTILEMEGA_SM80_FUSED_PHASE_TRACE=1',
                                    '-DTILEMEGA_SM80_FUSED_NEIGHBOR_TRACE=1',
                                    '-DTILEMEGA_SM80_FUSED_PRICES_TRACE=1']
            sidecar(library, '.build_command.txt').write_text(shlex.join(command)+'\n')
            with sidecar(library, '.ptxas.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/run_sm90r1_build.py'),
                                '--receipt', str(sidecar(library, '.build_limits.json')),
                                '--max-gib', '48', '--timeout', '520',
                                '--lock', str(args.build_lock), '--', *command],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=550)
            generate(library, build_root, source)
        identity_file = sidecar(library, '.identity.json')
        linked = read(identity_file)
        if (bool(linked['trace']) != (label == 'trace') or linked['binary_sha256'] != sha(library)
                or linked['generated_source_sha256'] != identity['generated_source_sha256']
                or linked['manifest_sha256'] != identity['manifest_sha256']
                or linked['compiler_version'] != identity['compiler_version']):
            raise ValueError('linked fused trace changed the frozen program')
        if label == 'trace':
            payload = {key: value for key, value in linked.items() if key != 'artifact_id'}
            if linked['artifact_id'] != digest(payload) or payload.get('performance_eligible', False) is not False:
                raise ValueError('instrumented release artifact must stay diagnostic-only')
            if 'performance_eligible' not in payload:
                payload['performance_eligible'] = False
                write_json(identity_file, dict(payload, artifact_id=digest(payload)))
        arms[label] = dict(path=str(library), sha256=linked['binary_sha256'],
                           identity_sha256=sha(identity_file),
                           source_digest=linked['source']['source_digest'])
    if arms['control']['source_digest'] != arms['trace']['source_digest']:
        raise ValueError('fused trace arms must build from the same source')
    write_json(args.out/'trace.plan.json', dict(
        model=anchor['model'], batch=anchor['batch'], prefill=anchor['prefill'],
        target=anchor['artifact_target'], target_sha256=anchor['target_sha256'],
        original_sha256=anchor['decode_sha256'], grid=plan['grid'], arms=arms,
        trace_kind='release' if args.release_ledger else 'overlap' if args.stage_overlap else 'fused'))


def accepted_trace(args):
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
        raise ValueError('stage attribution requires a guarded, matched trace within the 5% ceiling')
    return plan, report, device


def analyze(args):
    plan, report, device = accepted_trace(args)
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
    split, active_counts = Counter(), Counter()
    ratios, tail = [], []
    grid = plan['grid']
    for rows in by_stage.values():
        if len(rows) != grid or {int(r['worker']) for r in rows} != set(range(grid)):
            raise ValueError('expected every CTA at each observed fused stage')
        counts = {int(row['active_experts']) for row in rows}
        if len(counts) != 1 or min(counts) <= 0:
            raise ValueError('active expert count differs across CTAs')
        active = counts.pop()
        active_counts[active] += 1
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
        stages_over_1_3=sum(r > 1.3 for r in ratios),
        active_expert_counts=dict(sorted(active_counts.items())),
        task_distribution=[dict(gate_up=gu,down=dn,samples=count)
                           for (gu,dn),count in sorted(split.items())],
        trace_sha256=report['decode_sha256'], control_sha256=report['reference_sha256'],
        checkpoint=report['checkpoint'], gpu_uuid=device['nvml_uuid']))


def release_bandwidth(args):
    from a100_baselines import inputs
    from sm90r4_ledger import bounds, kind, tsv

    plan, report, device = accepted_trace(args)
    if plan.get('trace_kind') != 'release':
        raise ValueError('expected the compact native release trace')
    ledger = args.run/'ledger'
    if args.expected_tokens:
        observed = read(ledger/'reference-tokens.json')
        expected = read(args.expected_tokens)
        if (len(observed)!=plan['batch'] or len(expected)!=plan['batch']
                or any(len(row)!=report['tokens'] or row!=frozen[:len(row)]
                       for row,frozen in zip(observed,expected))):
            raise ValueError('release trace changed the frozen greedy output')
    meta = read(ledger/'native_release.json')
    if (meta['observation'] != 'release_after_cta_barrier' or meta['grid'] != plan['grid']
            or meta['performance_eligible'] or meta['uniform_overhead_subtracted']):
        raise ValueError('invalid release trace ABI')
    headers = {}
    for row in tsv(ledger/'native_release_steps.tsv'):
        worker, iteration, past, begin, end = (int(row[k]) for k in ('worker', 'iteration', 'past', 'begin', 'end'))
        if not 0 < end-begin < 2**32 or (iteration, worker) in headers:
            raise ValueError('invalid CTA trace clock window')
        headers[iteration, worker] = (past, begin, end)
    linked = read(sidecar(plan['arms']['control']['path'], '.plan.json'))
    coords = linked['native_dense']
    moe, fused = coords['model'] == 'moe', coords['expert_fuse']
    reducers = dict(kAttentionMerge='attention', kArgmaxReduce='lm_head', kMoeRoute='router',
                    kMoeCombine='expert_fused' if fused else 'expert_down')
    stages = {int(r['stage']): reducers.get(r['kind_name'], kind(r, moe))
              for r in tsv(ledger/'native_release_stages.tsv')}
    if sorted(stages) != list(range(meta['stages'])):
        raise ValueError('release ledger does not cover the complete stage schedule')
    start = 1 if coords['emb'] == 'la' else 0
    releases = defaultdict(dict)
    for row in tsv(ledger/'native_release_samples.tsv'):
        worker, iteration, stage, low = (int(row[k]) for k in ('worker', 'iteration', 'stage', 'release_low32'))
        if stage < start:
            continue
        _, begin, end = headers[iteration, worker]
        stamp = end-((end-low) % 2**32)
        if stamp < begin or worker in releases[iteration, stage]:
            raise ValueError('release stamp is outside the CTA execution window')
        releases[iteration, stage][worker] = stamp
    config, _ = inputs(plan['model'])
    config = read(config/'config.json')
    expert_counts = defaultdict(dict)
    if args.expert_trace:
        if not fused:
            raise ValueError('observed expert counts require the same fused decode geometry')
        paired = argparse.Namespace(run=args.expert_trace, plan=args.expert_trace.parent/'trace.plan.json')
        route_plan, route_report, _ = accepted_trace(paired)
        if (route_plan['original_sha256'] != plan['original_sha256']
                or route_report['checkpoint'] != report['checkpoint']
                or read(args.expert_trace/'ledger/reference-tokens.json') != read(ledger/'reference-tokens.json')):
            raise ValueError('expert counts belong to another decode artifact or request')
        for row in tsv(args.expert_trace/'ledger/stage_trace.tsv'):
            past, stage, count = (int(row[k]) for k in ('past', 'stage', 'active_experts'))
            if stages[stage] != 'expert_gate_up' or not count:
                continue
            if expert_counts[past].setdefault(stage, count) != count:
                raise ValueError('fused CTA records disagree on routed experts')
    rates, spans, weight = defaultdict(list), defaultdict(list), defaultdict(list)
    iterations = {iteration for iteration, _ in headers}
    for iteration in sorted(iterations):
        workers = set(range(plan['grid']))
        if any(set(releases[iteration, stage]) != workers for stage in range(start, len(stages))):
            raise ValueError('stage boundary lacks some physical CTAs')
        begin = min(headers[iteration, worker][1] for worker in workers)
        positions = {headers[iteration, worker][0] for worker in workers}
        if len(positions) != 1:
            raise ValueError('CTA traces disagree about the decode position')
        measurements = Counter()
        for stage in range(start, len(stages)):
            label = stages[stage]
            if (label == 'qkv' and coords['group_wait'] and stages.get(stage+1) == 'attention'):
                continue
            if label == 'attention' and coords['group_wait']:
                label = 'qkv_attention'
            if label == 'expert_gate_up' and fused:
                label = 'expert_fused'
            boundary = min(releases[iteration, stage].values())
            if boundary < begin:
                raise ValueError('release boundaries moved backwards')
            measurements[label] += boundary-begin
            begin = boundary
        # At 1 GB/s, the existing logical floor in us is bytes / 1000.
        logical = {stage: value*1000 for stage,value in bounds(config, plan['batch'], moe, 1).items()}
        logical['embedding'] = 2 * config['hidden_size'] * plan['batch']
        past = positions.pop()
        logical['attention'] *= past/575
        if 'qkv_attention' in measurements:
            logical['qkv_attention'] = logical['qkv']+logical['attention']
        if fused:
            logical['expert_fused'] = logical['expert_gate_up']+logical['expert_down']
        if args.expert_trace:
            counts = expert_counts[past]
            if len(counts) != config['num_hidden_layers']:
                raise ValueError('measured expert schedule does not cover this decode position')
            hidden = config['hidden_size']
            intermediate = config.get('moe_intermediate_size') or config['intermediate_size']
            logical['expert_fused'] = 6 * hidden * intermediate * sum(counts.values())
        for label, span in measurements.items():
            if label in logical and span > 0:
                spans[label].append(span)
                weight[label].append(logical[label])
                rates[label].append(logical[label]/span)
    rows = [dict(stage=label, logical_bytes=statistics.median(weight[label]),
                 stage_span_ns=statistics.median(spans[label]), effective_gbps=statistics.median(values),
                 samples=len(values), bytes_basis=(
                     ('observed active experts' if args.expert_trace else 'expected active experts')
                     if label.startswith('expert_') else 'weight and position-matched KV bytes'))
            for label,values in sorted(rates.items())]
    if not {'o', 'lm_head'} <= set(rates) or not ({'qkv', 'qkv_attention'} & set(rates)):
        raise ValueError('release ledger lacks native projection boundaries')
    write_json(args.out, dict(model=plan['model'], batch=plan['batch'], gpu_uuid=device['nvml_uuid'],
        original_sha256=plan['original_sha256'], perturbation_pct=report['overhead_pct'], rows=rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest='mode', required=True)
    make = modes.add_parser('prepare')
    make.add_argument('--anchor', required=True, type=Path)
    kind = make.add_mutually_exclusive_group()
    kind.add_argument('--release-ledger', action='store_true')
    kind.add_argument('--stage-overlap', action='store_true',
                      help='sample all CTA intervals every fourth native decode iteration')
    make.add_argument('--reuse-exact-control', action='store_true')
    make.add_argument('--build-lock', type=Path, default=Path('/tmp/tilemega_sm90r2_native_build.lock'))
    make.add_argument('--out', required=True, type=Path)
    for name in ('analyze', 'release-bandwidth'):
        collect = modes.add_parser(name)
        for flag in ('run', 'plan', 'out'):
            collect.add_argument(f'--{flag}', required=True, type=Path)
        if name == 'release-bandwidth':
            collect.add_argument('--expert-trace', type=Path)
            collect.add_argument('--expected-tokens', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare' and args.reuse_exact_control and not args.release_ledger:
        make.error('an archived control is supported for the compact release ledger')
    {'prepare': prepare, 'analyze': analyze, 'release-bandwidth': release_bandwidth}[args.mode](args)


if __name__ == '__main__':
    main()
