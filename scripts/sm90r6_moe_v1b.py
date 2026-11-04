#!/usr/bin/env python3
"""Compare F predictions with the existing R5 MoE stage marks, without sampling."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from sm90d1_common import read, sha
from sm90r6_merge_analysis import write


def compare(prediction, marks, trace, identity):
    model = prediction['input']['model']; batch = prediction['input']['batch']
    run = read(trace / 'run.json'); artifact = read(identity)
    if (artifact['plan']['native_dense'] != prediction['coordinates'] or
            artifact['binary_sha256'] != run['decode_sha256'] or
            run['trace']['tokens'] != run['reference']['tokens']):
        raise ValueError('R5 marks do not match the frozen F coordinates or token window')
    mapping = dict(gemm0='qkv', gemm1='o', gemm5='router', gemm4='lm_head',
                   attention='attention', route_local='expert_fused',
                   gemm6='expert_fused', gemm7='expert_fused', moe_combine='expert_fused',
                   argmax='lm_head', embedding='embedding')
    if prediction['coordinates']['emb'] == 'la':
        mapping['embedding'] = 'lm_head'
    spaces = defaultdict(lambda: defaultdict(list))
    for space in prediction['spaces']:
        spaces[mapping[space['kind']]][space['kind']].append(
            (space['last_end_ns'] - max(0, space['first_start_ns'])) / 1000)
    additive = defaultdict(float)
    additive_mapping = dict(route_la='router', combine_la='expert_fused',
                            argmax_la='lm_head', embedding_la='lm_head')
    if prediction['coordinates']['emb'] == 'stage':
        additive_mapping['embedding_la'] = 'embedding'
    for stage in prediction['additive_stages']['stages']:
        additive[additive_mapping.get(stage['kind'], stage['kind'])] += stage['stage_ns'] / 1000
    rows = []
    for mark in marks:
        if (mark['model'] != model or int(mark['batch']) != batch or
                mark['arm'] != 'REF-R4' or mark['segment'] != 'stage'):
            continue
        kind = mark['stage']
        # LA prologue stamps survive the trace ring but are not steady-step embedding.
        if kind == 'embedding' and prediction['coordinates']['emb'] == 'la':
            continue
        components = {k: statistics.median(v) for k, v in spaces[kind].items()}
        if not components or kind not in additive:
            raise ValueError('unmapped physical stage: ' + kind)
        measured = float(mark['median_us']); flow = sum(components.values())
        rows.append(dict(model=model, batch=batch, arm='F (R5 REF-R4 marks)', stage=kind,
            processes=int(mark['processes']), instances=int(mark['instances']),
            measured_boundary_us=measured, flow_space_sum_us=flow,
            flow_over_measured=flow / measured, additive_stage_us=additive[kind],
            additive_over_measured=additive[kind] / measured,
            flow_space_medians_us=json.dumps(components, sort_keys=True, separators=(',', ':')),
            instrumentation_overhead_pct=run['overhead_pct'],
            evidence='verified existing marks and coordinate identity; inferred stage aggregation',
            note='sum of semantic-space median spans versus R5 critical-CTA boundary time; not equal observables; one existing process; token-window check only; no resampling',
            raw=str(trace), trace_sha256=sha(trace / 'stage_trace.tsv'),
            identity_sha256=sha(identity)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--marks', type=Path, required=True)
    p.add_argument('--r5-d0', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.marks.open() as stream:
        marks = list(csv.DictReader(stream))
    rows = []; batches = set()
    for line in a.predictions.read_text().splitlines():
        if not line.strip():
            continue
        prediction = json.loads(line); cell = prediction['input']
        if cell['model'] != 'qwen3_30b_a3b':
            continue
        if cell['batch'] in batches:
            raise ValueError('duplicate F prediction')
        batches.add(cell['batch'])
        rows.extend(compare(prediction, marks, a.r5_d0 / 'before' / cell['id'] / 'trace',
                            a.r5_d0 / cell['id'] / 'decode.trace2.so.identity.json'))
    if batches != {1, 16}:
        raise ValueError('both existing MoE marks cells are required')
    for row in rows:
        row.update(predictions_sha256=sha(a.predictions), marks_sha256=sha(a.marks))
    write(a.out, rows)
    print(json.dumps(dict(cells=2, rows=len(rows))))


if __name__ == '__main__':
    main()
