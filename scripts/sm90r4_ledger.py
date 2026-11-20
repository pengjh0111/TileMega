#!/usr/bin/env python3
"""MR0 stage ledger (prompt §4 A2): per stage type, measured vs predicted vs bound.

`run` (guarded, inside random_checkpoint): real frozen prompts, decode to past
~583 so the trace ring keeps steps at past 567-582 with real routing; the
untraced reference times the same window for the instrumentation overhead.
`analyze`: telescoping release ledger -> per-type us/step against the
solver's per-stage prediction and the DRAM bound.
"""
import argparse
import csv
import ctypes as C
import gc
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

from sm90d1_common import read, sha, write

SLOTS = 16


def run(a):
    import torch
    from tilemega.serving.baseline_common import require_guard
    from tilemega.serving.engine import ServingEngine
    require_guard()
    os.environ['TILEMEGA_TRACE_LAUNCHES'] = str(SLOTS)
    ids = json.loads(a.prompt_ids.read_text())[:a.batch]
    checkpoint = os.environ['TILEMEGA_RANDOM_CHECKPOINT']
    result = {}
    reference_tokens = None
    prefill = None if str(a.prefill) == 'prompt' else a.prefill
    for label, decode in (('trace', a.decode), ('reference', a.reference)):
        if decode is None or (a.arm and a.arm != label):
            continue
        with ServingEngine(checkpoint, prefill, decode, a.batch, prompt_len=64,
                           max_new_tokens=a.tokens, mode='L1', decode_loop=True,
                           step_events=True, prefill_mode='L1') as engine:
            prompt = torch.tensor(ids, dtype=torch.int32)
            for _ in range(2):
                generation = engine.generate(prompt, a.tokens)
            window = generation.step_ms[-SLOTS:]
            result[label] = dict(window_step_ms=window, window_mean_ms=sum(window) / len(window),
                                 tokens=generation.tokens[:, :8].tolist())
            tokens = generation.tokens.tolist()
            write(a.out / (label+'-tokens.json'),tokens)
            if reference_tokens is not None and tokens != reference_tokens:
                raise AssertionError('instrumented and reference generation tokens differ')
            reference_tokens = tokens
            if label == 'trace':
                dump = engine.decode_lib.lib.tm_plan_dump_serving_trace
                dump.argtypes = [C.c_void_p, C.c_char_p]
                dump.restype = C.c_int
                a.out.mkdir(parents=True, exist_ok=True)
                if dump(engine.decode.handle, os.fsencode(str(a.out.resolve()))):
                    raise RuntimeError('serving trace dump failed')
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    if 'reference' in result and 'trace' in result:
        result['overhead_pct'] = (result['trace']['window_mean_ms'] /
                                  result['reference']['window_mean_ms'] - 1) * 100
    receipt = f'run-{a.arm}.json' if a.arm else 'run.json'
    write(a.out / receipt, dict(result, decode_sha256=sha(a.decode), batch=a.batch,
                                   pid=os.getpid(),
                                   tokens=a.tokens, slots=SLOTS, driver_sha256=sha(__file__),
                                   reference_sha256=sha(a.reference) if a.reference else None,
                                   checkpoint=read(Path(checkpoint)/'random_checkpoint.json')['checkpoint_sha256'],
                                   full_token_equivalence=not a.arm and a.reference is not None,
                                   diagnostic_only=True, performance_eligible=False))


def isolated(a):
    if not a.reference or a.arm:
        raise ValueError('independent trace requires both arms')
    command = [sys.executable, __file__, 'run', '--decode', str(a.decode),
               '--reference', str(a.reference), '--prefill', str(a.prefill),
               '--batch', str(a.batch), '--tokens', str(a.tokens),
               '--prompt-ids', str(a.prompt_ids), '--out', str(a.out)]
    for arm in ('reference', 'trace'):
        subprocess.run([*command, '--arm', arm], check=True)
    reference = read(a.out / 'run-reference.json')
    trace = read(a.out / 'run-trace.json')
    identity = ('checkpoint', 'decode_sha256', 'reference_sha256',
                'batch', 'tokens', 'driver_sha256')
    if any(reference[k] != trace[k] for k in identity) or reference['pid'] == trace['pid']:
        raise ValueError('trace arms must share inputs and run in separate processes')
    if read(a.out / 'reference-tokens.json') != read(a.out / 'trace-tokens.json'):
        raise AssertionError('instrumented and reference generation tokens differ')
    trace.update(reference=reference['reference'], reference_pid=reference['pid'],
                 full_token_equivalence=True,
                 overhead_pct=100 * (trace['trace']['window_mean_ms'] /
                                     reference['reference']['window_mean_ms'] - 1))
    write(a.out / 'run.json', trace)


def tsv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def kind(row, moe):
    name, k = row['name'], row['kind_name']
    if k == 'kEmbedding':
        return 'embedding'
    if k == 'kFusedAttention':
        return 'attention'
    if 'lm_head' in name:
        return 'lm_head'
    for key, value in (('qkv', 'qkv'), ('o_proj', 'o'), ('router', 'router'), ('gate_up', 'gate_up'),
                       ('down', 'down')):
        if key in name:
            return value if not moe or key in ('qkv', 'o_proj', 'router') else 'expert_' + value
    return 'other'


def analyze(a):
    stages = tsv(a.trace / 'runtime_stages.tsv')
    moe = any('router' in s['name'] for s in stages)
    prediction = read(a.receipt)
    ranked = prediction['ranked'][prediction['winner']]['prediction']['stages']
    fused = any(s['kind'] == 'expert_fused' for s in ranked)
    kinds = {int(s['stage']): kind(s, moe) for s in stages}
    if fused:
        if 'expert_down' in kinds.values() or 'expert_gate_up' not in kinds.values():
            raise ValueError('the recorded expert stage boundaries differ from the fused search plan')
        kinds = {index: 'expert_fused' if label == 'expert_gate_up' else label
                 for index, label in kinds.items()}
    rows = tsv(a.trace / 'stage_trace.tsv')
    by = {}
    for r in rows:
        by.setdefault((int(r['iteration']), int(r['stage'])), []).append(r)
    iterations = sorted({i for i, _ in by})
    count = len(stages)
    per_step = []
    cta_timing = {}
    joint = []
    for it in iterations:
        # With emb = LA the prologue (stage 0) runs only before step 0.
        present = [g for g in range(count) if (it, g) in by]
        if present != list(range(count)) and present != list(range(1, count)):
            continue
        # The barrier of stage g releases when its last CTA arrives; the
        # earliest release bounds that instant from above (CTA polls differ).
        release = {g: min(int(r['t_release']) for r in by[(it, g)]) for g in present}
        begin = min(int(r['t_begin']) for r in by[(it, present[0])])
        totals = {}
        for g in present:
            start = release[g - 1] if g != present[0] else begin
            label = kinds[g]
            totals[label] = totals.get(label, 0) + (release[g] - start) / 1e3
            # Group waits permit a successor to start on one CTA before another
            # CTA finishes its predecessor. Retain the boundary skew explicitly.
            local = [(int(r['t_begin']), int(r['t_tasks_end'])) for r in by[it, g]]
            starts, ends = zip(*local)
            cta_timing.setdefault(label, []).append({
                'entry_spread_us': (max(starts)-min(starts))/1000,
                'longest_local_body_us': max(end-begin for begin,end in local)/1000,
                'last_body_after_first_release_us': (max(ends)-release[g])/1000})
            if label == 'attention' and kinds.get(g-1) == 'qkv' and (it, g-1) in by:
                active = [r for r in by[it, g] if int(r['tasks'])]
                qkv = [r for r in by[it, g-1] if int(r['tasks'])]
                if active and qkv:
                    first_attention = min(int(r['t_begin']) for r in active)
                    last_attention = max(int(r['t_tasks_end']) for r in active)
                    first_qkv = min(int(r['t_begin']) for r in qkv)
                    last_qkv = max(int(r['t_tasks_end']) for r in qkv)
                    overlap = min(last_qkv, last_attention) - max(first_qkv, first_attention)
                    joint.append({
                        'span_overlap_us': max(0, overlap)/1000,
                        'attention_active_ctas': len(active),
                        'attention_tasks': sum(int(r['tasks']) for r in active),
                        'attention_max_tasks_per_cta': max(int(r['tasks']) for r in active),
                        'attention_active_entry_spread_us':
                            (max(int(r['t_begin']) for r in active)-first_attention)/1000,
                        'attention_active_longest_body_us':
                            max(int(r['t_tasks_end'])-int(r['t_begin']) for r in active)/1000,
                        'attention_longest_cta_boundary_wait_us':
                            max(int(r['t_release'])-int(r['t_tasks_end']) for r in active)/1000,
                    })
        totals['step'] = (release[present[-1]] - begin) / 1e3
        per_step.append(totals)
    measured = {k: st.median(s[k] for s in per_step) for k in per_step[0]}
    folding = {'route_la': 'router', 'combine_la': 'expert_fused' if fused else 'expert_down', 'argmax_la': 'lm_head',
               'embedding_la': 'lm_head', 'embedding': 'embedding'}
    predicted = {}
    for s in ranked:
        k = folding.get(s['kind'], s['kind'])
        predicted[k] = predicted.get(k, 0) + s['instances'] * s['stage_ns'] / 1e3
    config = read(a.config)
    bound = bounds(config, a.batch, moe, a.dram_gbps)
    if fused:
        bound['expert_fused'] = bound.pop('expert_gate_up') + bound.pop('expert_down')
    predicted['step'] = sum(predicted.values())
    out = []
    for k in sorted(measured, key=lambda k: -measured[k]):
        out.append(dict(stage=k, measured_us=round(measured[k], 1), predicted_us=round(predicted.get(k, 0), 1),
                        bound_us=round(bound.get(k, 0), 1),
                        over_prediction_us=round(measured[k] - predicted.get(k, 0), 1)))
    cta = {label: dict(instances=len(samples),
                **{metric: st.median(s[metric] for s in samples) for metric in samples[0]})
           for label, samples in cta_timing.items()}
    report = dict(steps=len(per_step), rows=out, trace=str(a.trace), per_stage_instance=cta)
    if joint:
        report['qkv_attention'] = dict(
            instances=len(joint),
            instances_with_span_overlap=sum(r['span_overlap_us'] > 0 for r in joint),
            **{metric: st.median(r[metric] for r in joint) for metric in joint[0]})
    write(a.out, report)
    for r in out:
        print(r)


def bounds(c, batch, moe, gbps):
    h, layers = c['hidden_size'], c['num_hidden_layers']
    heads, kv = c['num_attention_heads'], c['num_key_value_heads']
    d = c.get('head_dim') or h // heads
    past = 575
    result = dict(qkv=layers * 2 * h * (heads + 2 * kv) * d, o=layers * 2 * heads * d * h,
                  attention=layers * batch * kv * past * d * 4, lm_head=2 * h * c['vocab_size'])
    if moe:
        e, k = c['num_experts'], c['num_experts_per_tok']
        inner = c.get('moe_intermediate_size') or c['intermediate_size']
        active = e * (1 - (1 - k / e) ** batch)
        result.update(router=layers * 2 * h * e, expert_gate_up=layers * active * 2 * h * 2 * inner,
                      expert_down=layers * active * 2 * inner * h)
    else:
        inner = c['intermediate_size']
        result.update(gate_up=layers * 2 * h * 2 * inner, down=layers * 2 * inner * h)
    result = {k: v / gbps / 1e3 for k, v in result.items()}
    result['step'] = sum(result.values())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--prefill', type=Path, required=True)
    r.add_argument('--decode', type=Path, required=True)
    r.add_argument('--reference', type=Path)
    r.add_argument('--arm', choices=('trace', 'reference'), help=argparse.SUPPRESS)
    r.add_argument('--independent-arms', action='store_true')
    r.add_argument('--prompt-ids', type=Path, required=True)
    r.add_argument('--batch', type=int, required=True)
    r.add_argument('--tokens', type=int, default=520)
    r.add_argument('--out', type=Path, required=True)
    z = sub.add_parser('analyze')
    z.add_argument('--trace', type=Path, required=True)
    z.add_argument('--receipt', type=Path, required=True)
    z.add_argument('--config', type=Path, required=True)
    z.add_argument('--batch', type=int, required=True)
    z.add_argument('--dram-gbps', type=float, default=3241.24)
    z.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'run':
        return isolated(a) if a.independent_arms else run(a)
    return analyze(a)


if __name__ == '__main__':
    main()
