#!/usr/bin/env python3
"""D0 analysis (prompt §4 A0.1 item 6): split a stage into its serial segments.

For every stage instance the CTA that arrives last is the critical one: the
boundary's release waits on it, so its marks are what the step actually paid.
Each segment is reported relative to that CTA's begin, together with
`release - arrive` (the barrier tail) and the stage time (this release minus the
previous stage's release), then reduced over steps and layers per stage type and
compared with the probe constant that priced the same work.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, read, write

# Segment names per stage type, in the order appendix D stamps them, with the
# probe constant each one is compared against (None: priced by no probe).
SEGMENTS = {
    'qkv':       [('first_slot', 0, None), ('mma_end', 1, None), ('epilogue_end', 2, None)],
    'o':         [('first_slot', 0, None), ('mma_end', 1, None), ('epilogue_end', 2, None)],
    'lm_head':   [('first_slot', 0, None), ('mma_end', 1, None), ('last_arriver', 2, None),
                  ('argmax_end', 3, 'argmax_la'), ('finish_end', 4, None)],
    'router':    [('first_slot', 0, None), ('mma_end', 1, None), ('logits_end', 2, None),
                  ('la_decided', 3, None), ('topk_end', 4, None), ('schedule_end', 5, 'route_la')],
    'attention': [('prepare_end', 0, 'qprep'), ('first_kv_slot', 1, None), ('blocks_end', 2, None),
                  ('finish_end', 3, None), ('la_decided', 4, None), ('merge_end', 5, 'merge')],
    'expert_fused': [('local_route', 0, 'route_local'), ('first_slot', 1, None),
                     ('gate_up_end', 2, None), ('down_wait_end', 3, None),
                     ('combine_begin', 4, None), ('combine_end', 5, 'combine')],
}


def tsv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def kind(row, moe, fused):
    name, k = row['name'], row['kind_name']
    if k == 'kEmbedding':
        return 'embedding'
    if k == 'kFusedAttention':
        return 'attention'
    if 'lm_head' in name:
        return 'lm_head'
    for key, value in (('qkv', 'qkv'), ('o_proj', 'o'), ('router', 'router'),
                       ('gate_up', 'gate_up'), ('down', 'down')):
        if key in name:
            if not moe or key in ('qkv', 'o_proj', 'router'):
                return value
            # MR7's M56 is emitted at the gate_up stage and carries its weight
            # name; a fused program has no separate down stage.
            return 'expert_fused' if fused else 'expert_' + value
    return 'other'


def probes(target, batch, model):
    """The probe values the marks are compared with, in microseconds."""
    c = read(target)['sm90_native_constants']
    experts, queries, norm = (128, 8, 1) if model.startswith('qwen3_30b') else (64, 1, 2)
    grid, result = 'g132', {}
    suffix = '' if experts == 128 else f':e{experts}:all1'
    route = c.get('route_la_ns', {}).get(grid + suffix, {})
    if str(batch) in route:
        result['route_la'] = route[str(batch)] / 1e3
    attention = c.get('attention_ns', {}).get(f'd128:q{queries}:norm{norm}', {})
    if 'prepare' in attention:
        result['qprep'] = attention['prepare'] / 1e3
    merge = attention.get('merge', {})
    # Chunks are a coordinate, so the merge probe is keyed by the cell's S.
    result['merge_by_chunks'] = {int(k): v / 1e3 for k, v in merge.items()}
    for key, name in (('route_local_ns', 'route_local'), ('combine_ns', 'combine')):
        section = c.get(key, {})
        entry = section.get(f'b{batch}:e{experts}') or section.get(str(batch))
        if isinstance(entry, (int, float)):
            result[name] = entry / 1e3
    return result


def analyze(trace, batch, model, target, arm, chunks):
    stages = tsv(trace / 'runtime_stages.tsv')
    moe = any('router' in s['name'] for s in stages)
    fused = moe and not any('down' in s['name'] for s in stages)
    kinds = {int(s['stage']): kind(s, moe, fused) for s in stages}
    rows = tsv(trace / 'stage_trace.tsv')
    if 'm0' not in rows[0]:
        raise SystemExit('stage_trace.tsv has no marks: rebuild with --stage-level 2')
    by = {}
    for r in rows:
        by.setdefault((int(r['iteration']), int(r['stage'])), []).append(r)
    samples = {}
    for (iteration, stage), group in sorted(by.items()):
        # The critical CTA is the last to arrive; the release waits on it.
        critical = max(group, key=lambda r: int(r['t_tasks_end']))
        begin, arrive = int(critical['t_begin']), int(critical['t_tasks_end'])
        # The critical CTA's own release, not the earliest in the grid: a
        # group-waiting qkv (D8) releases per KV group, so CTAs leave at very
        # different times and min(release) can precede max(arrive).
        release = int(critical['t_release'])
        previous = by.get((iteration, stage - 1))
        start = max(int(r['t_release']) for r in previous) if previous else begin
        name = kinds.get(stage, 'other')
        store = samples.setdefault(name, {})
        for label, index, _ in SEGMENTS.get(name, []):
            value = int(critical[f'm{index}'])
            # Zero means the geometry never reached that mark (e.g. no merge).
            if value:
                store.setdefault(label, []).append((value - begin) / 1e3)
        store.setdefault('barrier_tail', []).append((release - arrive) / 1e3)
        store.setdefault('stage', []).append((release - start) / 1e3)
    p = probes(target, batch, model)
    if chunks:
        p['merge'] = p.get('merge_by_chunks', {}).get(chunks)
    out = []
    for name, store in samples.items():
        order = [label for label, _, _ in SEGMENTS.get(name, [])] + ['barrier_tail', 'stage']
        probe_of = {label: key for label, _, key in SEGMENTS.get(name, []) if key}
        for label in order:
            values = store.get(label)
            if not values:
                continue
            median = st.median(values)
            ordered = sorted(values)
            p90 = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1) + 0.5))]
            probe = p.get(probe_of.get(label, ''))
            out.append(dict(model=model, batch=batch, arm=arm, stage=name, segment=label,
                            median_us=round(median, 3), p90_us=round(p90, 3),
                            probe_us=round(probe, 3) if probe else '',
                            ratio=round(median / probe, 3) if probe else '',
                            instances=len(values)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, action='append', required=True,
                   help='a process trace directory; repeat for several processes')
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--chunks', type=int, default=0, help="the cell's attention S, for the merge probe")
    p.add_argument('--target', type=Path, default=ROOT / 'configs/targets/sm_90_h800.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for index, directory in enumerate(a.trace):
        for row in analyze(directory, a.batch, a.model, a.target, a.arm, a.chunks):
            rows.append(dict(row, process=index))
    # Across processes the median of medians, as the protocol reports elsewhere.
    merged = {}
    for row in rows:
        merged.setdefault((row['stage'], row['segment']), []).append(row)
    final = []
    for (stage, segment), group in merged.items():
        first = group[0]
        final.append(dict(model=first['model'], batch=first['batch'], arm=first['arm'],
                          stage=stage, segment=segment,
                          median_us=round(st.median(r['median_us'] for r in group), 3),
                          p90_us=round(max(r['p90_us'] for r in group), 3),
                          probe_us=first['probe_us'], ratio=first['ratio'],
                          processes=len(group),
                          instances=sum(r['instances'] for r in group)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)
    print(json.dumps({'rows': len(final), 'out': str(a.out),
                      'stages': sorted({r['stage'] for r in final})}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
