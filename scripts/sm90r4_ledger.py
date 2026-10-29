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
    for label, decode in (('trace', a.decode), ('reference', a.reference)):
        if decode is None:
            continue
        with ServingEngine(checkpoint, a.prefill, decode, a.batch, prompt_len=64,
                           max_new_tokens=a.tokens, mode='L1', decode_loop=True,
                           step_events=True, prefill_mode='L1') as engine:
            prompt = torch.tensor(ids, dtype=torch.int32)
            for _ in range(2):
                generation = engine.generate(prompt, a.tokens)
            window = generation.step_ms[-SLOTS:]
            result[label] = dict(window_step_ms=window, window_mean_ms=sum(window) / len(window),
                                 tokens=generation.tokens[:, :8].tolist())
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
    if 'reference' in result:
        result['overhead_pct'] = (result['trace']['window_mean_ms'] /
                                  result['reference']['window_mean_ms'] - 1) * 100
    write(a.out / 'run.json', dict(result, decode_sha256=sha(a.decode), batch=a.batch,
                                   tokens=a.tokens, slots=SLOTS, driver_sha256=sha(__file__),
                                   diagnostic_only=True, performance_eligible=False))


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
    kinds = {int(s['stage']): kind(s, moe) for s in stages}
    rows = tsv(a.trace / 'stage_trace.tsv')
    by = {}
    for r in rows:
        by.setdefault((int(r['iteration']), int(r['stage'])), []).append(r)
    iterations = sorted({i for i, _ in by})
    count = len(stages)
    per_step = []
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
            totals[kinds[g]] = totals.get(kinds[g], 0) + (release[g] - start) / 1e3
        totals['step'] = (release[present[-1]] - begin) / 1e3
        per_step.append(totals)
    measured = {k: st.median(s[k] for s in per_step) for k in per_step[0]}
    prediction = read(a.receipt)
    ranked = prediction['ranked'][prediction['winner']]['prediction']['stages']
    folding = {'route_la': 'router', 'combine_la': 'expert_down', 'argmax_la': 'lm_head',
               'embedding_la': 'lm_head', 'embedding': 'embedding'}
    predicted = {}
    for s in ranked:
        k = folding.get(s['kind'], s['kind'])
        predicted[k] = predicted.get(k, 0) + s['instances'] * s['stage_ns'] / 1e3
    config = read(a.config)
    bound = bounds(config, a.batch, moe, a.dram_gbps)
    out = []
    for k in sorted(measured, key=lambda k: -measured[k]):
        out.append(dict(stage=k, measured_us=round(measured[k], 1), predicted_us=round(predicted.get(k, 0), 1),
                        bound_us=round(bound.get(k, 0), 1),
                        over_prediction_us=round(measured[k] - predicted.get(k, 0), 1)))
    write(a.out, dict(steps=len(per_step), rows=out, trace=str(a.trace)))
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
    return run(a) if a.action == 'run' else analyze(a)


if __name__ == '__main__':
    main()
