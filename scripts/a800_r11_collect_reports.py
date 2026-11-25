#!/usr/bin/env python3
"""Collect the R11 numeric gate, coverage, rank, and stage tables."""
import argparse
import csv
from pathlib import Path

from a100_baselines import ROOT, inspect_inputs, sha
from a100_results import write_csv
from a800_baselines import INVENTORY
from a800_native import artifact
from a800_r11_collect_round import collect
from a800_r11_diagnostics import collect as collect_stages
from a800_r11_finish import completed
from a800_r11_gate import read
from a800_r11_mainloop import make as mainloop_rows, role_evidence
from a800_r11_rank_coverage import summarize
from a800_r11_screen import arm_name, arms
from tilemega.bench.scheduler import write_json


def coverage_cells(scope, old, pipeline_jobs, recovery_jobs):
    reference = ROOT/'docs/results/A800_R10_candidates.csv'
    with reference.open(newline='') as stream:
        published = {(row['model'], int(row['batch'])): row for row in csv.DictReader(stream)}
    runs = scope.parent
    for (model, batch), item in sorted(published.items()):
        fresh = scope/'solver-release'/model/f'B{batch}'
        binding = fresh/'finalists.json'
        if not binding.exists():
            continue
        spec = read(binding)
        old_search = old/model/f'B{batch}'/'enum.so.search'
        if sha(old_search/'receipt.json') != item['receipt_sha256']:
            raise ValueError(f'R10 {model} B{batch} search receipt differs from the published result')
        if sha(spec['prepared']) != spec['prepared_sha256']:
            raise ValueError(f'R11 {model} B{batch} shortlist changed')
        yield dict(model=model, batch=batch, runs=str(runs), finalists=str(binding),
                   r10_coverage=str(old_search/'candidate-coverage.jsonl'),
                   r11_coverage=str(Path(spec['prepared']).parent/'candidate-coverage.jsonl'),
                   r11_coordinates=spec['candidates'][0]['coordinates'],
                   job=pipeline_jobs.get((model, batch)), recovery=recovery_jobs.get((model, batch)))


def gate_rows(legacy, inputs):
    accounting = legacy/'legacy.accounting.json'
    if not accounting.exists():
        return
    states = read(accounting)
    runs = legacy.parents[1]
    for index, row in enumerate(read(legacy/'legacy.rounds.json')['items']):
        result = states.get(str(index), {})
        job = result.get('job')
        if not job or not (runs/'optimization/decode-pairs'/job/'result.json').exists():
            continue
        if sha(row['binding']) != row['sha256']:
            raise ValueError('the R10 recheck binding changed')
        (fact, _, _), = collect(runs, job, inputs)
        if fact['processes'] != 5:
            raise ValueError('each R10 borderline arm requires five fresh processes')
        yield fact


def qwen_incumbent_resources(scope):
    """Bind the resident-CTA memory bound to the R10 Qwen binaries."""
    for item in sorted((scope/'incumbents/qwen3_30b_a3b').glob('B*/incumbent.json')):
        anchor = read(item)
        _, identity, manifest = artifact(Path(anchor['decode']))
        if (identity['binary_sha256'] != anchor['decode_sha256']
                or identity['target_sha256'] != anchor['target_sha256']
                or sha(anchor['target']) != anchor['target_sha256']):
            raise ValueError('Qwen residency must come from the frozen R10 decode')
        target = read(anchor['target'])['resources']
        shared = identity['shared_memory_bytes']
        yield dict(batch=anchor['batch'], grid=manifest['grid'], num_sms=target['num_sms'],
                   shared_bytes_per_cta=shared, shared_bytes_per_sm=target['max_smem_per_sm'],
                   max_ctas_by_shared_memory=target['max_smem_per_sm']//shared,
                   decode_sha256=anchor['decode_sha256'])


def report(scope, legacy, old, out):
    out.mkdir(parents=True, exist_ok=True)
    waves = scope/'campaign-waves'
    runs = scope.parent
    trace_scope = scope/'overlap'
    inputs = inspect_inputs(INVENTORY)
    pairs = (completed(waves) if (waves/'accounting.json').exists() else ())
    timings = {}
    pipeline_jobs = {}
    recovery_jobs = {}
    mainloop_ready = False
    for phase, plan, job in pairs:
        if not (runs/'optimization/decode-pairs'/job/'result.json').exists():
            continue
        binding = plan['finalists']
        if sha(binding) != plan['finalists_sha256']:
            raise ValueError('an R11 timing binding changed')
        if phase == 'pipeline':
            pipeline_jobs[plan['model'], plan['batch']] = job
            continue
        if phase == 'routerfix':
            recovery_jobs[plan['model'], plan['batch']] = dict(job=job, finalists=binding)
            continue
        if phase == 'mainloop':
            mainloop_ready = True
            continue
        arm_names = {arm_name(row['rank']): row['label'] for row in arms(read(binding), phase)}
        for fact, _, _ in collect(runs, job, inputs):
            if fact['arm'] in arm_names:
                key = f'{fact["model"]}/B{fact["batch"]}/{arm_names[fact["arm"]]}'
                timings[key] = fact

    result = summarize(list(coverage_cells(scope, old, pipeline_jobs, recovery_jobs)), out, inputs)
    gate = list(gate_rows(legacy, inputs))
    if gate:
        write_csv(out/'A800_R11_gate.csv', gate)
    result.update(legacy_cells=len(gate), legacy_passed=sum(row['pass'] for row in gate))
    result['qwen_incumbent_resources'] = list(qwen_incumbent_resources(scope))
    has_traces = (trace_scope/'accounting.tsv').exists()
    if has_traces:
        collect_stages(runs, trace_scope, timings, out)
    if mainloop_ready:
        everyone = role_evidence(trace_scope, runs) if has_traces else {}
        rows = list(mainloop_rows(waves, runs, inputs, everyone))
        if rows:
            write_csv(out/'A800_R11_mainloop.csv', rows)
    write_json(out/'collection.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scope', 'legacy', 'r10', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    report(args.scope, args.legacy, args.r10, args.out)


if __name__ == '__main__':
    main()
