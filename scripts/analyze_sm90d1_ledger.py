#!/usr/bin/env python3
"""Reconstruct a telescoping L1 stage ledger without calling gaps barriers."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import statistics as st

from sm90d1_common import read, write


def tsv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def category(stage):
    kind, name = stage['kind_name'], stage['name']
    if kind in ('kGemm', 'kGemmCombine'):
        family = next((value for key, value in (
            ('.qkv.', 'qkv'), ('.o_proj.', 'o'), ('.gate_up.', 'gate_up'),
            ('.down_proj.', 'down'), ('lm_head', 'lm_head')) if key in name), 'other-gemm')
        return family + ('-combine' if kind == 'kGemmCombine' else '')
    return {'kFusedAttention': 'attention', 'kAttentionMerge': 'merge',
            'kArgmaxReduce': 'argmax', 'kEmbedding': 'embedding',
            'kRMSNorm': 'norm'}.get(kind, kind)


def stage_bytes(stage, past, batch, config):
    """Logical compulsory bytes, not measured DRAM transactions."""
    h = config['hidden_size']
    d = config.get('head_dim', h // config['num_attention_heads'])
    kv = config['num_key_value_heads']
    inter = config['intermediate_size']
    family = category(stage).removesuffix('-combine')
    dimensions = {'qkv': (h, (config['num_attention_heads'] + 2*kv)*d),
                  'o': (h, h), 'gate_up': (h, 2*inter), 'down': (inter, h),
                  'lm_head': (h, config['vocab_size'])}
    if stage['kind_name'] == 'kGemm':
        k, n = dimensions.get(family, (0, 0))
        split = stage['_geometry']['split_k']
        return int(stage['weight_bytes']) + batch*k*2 + batch*n*(4*split if split > 1 else 2)
    if stage['kind_name'] == 'kFusedAttention':
        return batch*kv*past*d*4 + batch*(config['num_attention_heads']+2*kv)*d*2
    if stage['kind_name'] == 'kAttentionMerge':
        splits = (1088 + stage['_attention_block']-1)//stage['_attention_block']
        return batch*config['num_attention_heads']*(d+1)*4*splits + batch*h*2
    if stage['kind_name'] == 'kGemmCombine':
        k, n = dimensions[family]
        out = n//2 if family == 'gate_up' else n
        residual = batch*out*2 if family in ('o', 'down') else 0
        return batch*n*4*stage['_geometry']['split_k'] + batch*out*2 + residual
    if stage['kind_name'] in ('kEmbedding', 'kRMSNorm'):
        return batch*h*4
    if stage['kind_name'] == 'kArgmaxReduce':
        return batch*4
    return None


def analyze(folder, model_config, batch, plan):
    stages = {int(row['stage']): row for row in tsv(folder / 'runtime_stages.tsv')}
    gemms = iter(plan['gemms'])
    previous_geometry, block = None, plan['attention_kv_block']
    for stage in stages.values():
        if stage['kind_name'] == 'kGemm':
            previous_geometry = next(gemms)
        if stage['kind_name'] in ('kGemm', 'kGemmCombine'):
            stage['_geometry'] = previous_geometry
        if stage['kind_name'] == 'kFusedAttention':
            block = int(stage['kv_block'])
        stage['_attention_block'] = block
    groups, profiles, steps = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in tsv(folder / 'stage_trace.tsv'):
        row = {k: int(v) for k, v in row.items()}
        if row['t_release']:
            groups[row['iteration'], row['stage']].append(row)
    for row in tsv(folder / 'task_profile.tsv') if (folder / 'task_profile.tsv').exists() else []:
        row = {k: v if k == 'kind' else int(v) for k, v in row.items()}
        profiles[row['iteration'], row['stage']].append(row)
    for row in tsv(folder / 'step_trace.tsv'):
        row = {k: int(v) for k, v in row.items()}
        if row['kernel_begin']:
            steps[row['iteration']].append(row)
    keep = read(folder / 'trace.json').get('retain_latest_iterations', len(steps))
    steps = {i: steps[i] for i in sorted(steps)[-keep:]}
    details, closure = [], []
    for iteration in sorted(steps):
        step = steps[iteration]
        begin = min(row['kernel_begin'] for row in step)
        end = max(row['kernel_end'] for row in step)
        previous_end = begin
        previous_by_worker = {row['worker']: begin for row in step}
        envelopes = 0
        for index in sorted(stages):
            rows = groups.get((iteration, index), [])
            if not rows:
                continue
            critical = max(rows, key=lambda row: row['t_tasks_end'])
            release = max(row['t_release'] for row in rows)
            active = [row for row in rows if row['tasks']]
            task_rows = profiles.get((iteration, index), [])
            by_worker = defaultdict(list)
            for row in task_rows:
                by_worker[row['worker']].append(row)
            durations = {worker: sum(row['run_end']-row['run_begin'] for row in tasks)
                         for worker, tasks in by_worker.items()}
            critical_tasks = sorted(by_worker.get(critical['worker'], []), key=lambda row: row['run_begin'])
            complete = len(task_rows) == sum(row['tasks'] for row in rows)
            mean = st.mean(durations.get(row['worker'], 0) for row in active) if active and complete else None
            body = durations.get(critical['worker']) if complete else None
            startup = (critical_tasks[0]['run_begin'] - previous_by_worker[critical['worker']]
                       if critical_tasks and complete else None)
            gaps = sum(b['run_begin']-a['run_end'] for a, b in zip(critical_tasks, critical_tasks[1:])) if complete else None
            logical = stage_bytes(stages[index], critical['past'], batch, model_config)
            floor = logical / 3130 if logical is not None else None
            envelope = release - previous_end
            tail = release - critical['t_tasks_end']
            residual = envelope-(body+startup+gaps+tail) if None not in (body, startup, gaps) else None
            frequency = [((r['local_end']-r['local_begin'])/(r['run_end']-r['run_begin']))
                         for r in critical_tasks if r.get('local_end') and r['run_end']-r['run_begin'] >= 5000]
            freq = st.median(frequency) if frequency else None
            first = critical_tasks[0] if critical_tasks else None
            first_block = None
            if first and first['first_ready']:
                if first.get('local_begin') and freq:
                    first_block = (first['first_ready']-first['local_begin'])/freq
                elif not first.get('local_begin'):
                    first_block = first['first_ready']-first['run_begin']
            details.append(dict(iteration=iteration, past=critical['past'], stage=index,
                kind=category(stages[index]), tasks=sum(row['tasks'] for row in rows),
                active_ctas=len(active), max_tasks_per_cta=max(row['tasks'] for row in rows),
                profile_complete=complete, bytes=logical, floor_ns=floor, envelope_ns=envelope,
                critical_cta=critical['worker'], start_ns=startup, first_block_ns=first_block,
                body_critical_ns=body,
                body_mean_ns=mean, gaps_ns=gaps, release_ns=tail,
                skew_ns=release-min(row['t_release'] for row in rows),
                inefficiency_ns=mean-floor if mean is not None and floor is not None else None,
                imbalance_ns=body-mean if body is not None and mean is not None else None,
                residual_ns=residual, clock_cycles_per_ns_fit=freq))
            envelopes += envelope
            previous_end = release
            previous_by_worker = {row['worker']: row['t_release'] for row in rows}
        boundary = end-previous_end
        closure.append(dict(iteration=iteration, past=step[0]['past'], stage_sum_ns=envelopes,
            boundary_ns=boundary, step_ns=end-begin,
            closure_error_fraction=(envelopes+boundary-(end-begin))/(end-begin),
            token_lag_ns=max(row['token_lag'] for row in step),
            kv_lag_ns=max(row['kv_lag'] for row in step)))
    return details, closure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    details, closure = analyze(a.trace, read(a.model_config), a.batch, read(a.plan))
    write(a.out, dict(evidence='verified timestamp arithmetic; inferred compulsory-byte floors',
        performance_eligible=False, stages=details, steps=closure,
        limitations=['globaltimer timestamps are quantized to measured resolution',
                     'unprofiled stages and unknown byte floors remain null',
                     'logical byte floor excludes redundant transactions and cache writebacks',
                     'clock conversion is a local fit, not an assumption of locked clocks']))


if __name__ == '__main__':
    main()
