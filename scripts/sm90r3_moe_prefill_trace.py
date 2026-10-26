#!/usr/bin/env python3
"""M8: stage ledger of the frozen MoE prefill (prompt §8 item 7; diagnostic only).

Inside `tilemega.serving.random_checkpoint` under the scheduler guard: times
the uninstrumented prefill, then the traced clone (same generated program
with stage-boundary stamps), dumps the clone's stage trace and attributes
each launch's TTFT to stage types. The overhead row compares the two means.
"""
import argparse
import csv
import ctypes as C
import json
import os
from collections import defaultdict
from pathlib import Path
import statistics

import torch

from tilemega.cache import atomic_json
from tilemega.serving.engine import ServingEngine

ROOT = Path(__file__).resolve().parents[1]


def category(kind, name):
    """Stage type from the runtime kind and the stage's weight name."""
    if kind != 'kGemm':
        return kind[1:] if kind.startswith('k') else kind
    for key, label in (('router', 'gemm.router'), ('mlp.gate.weight', 'gemm.router'), ('gate_up', 'gemm.expert_gate_up'),
                       ('experts.down', 'gemm.expert_down'), ('qkv', 'gemm.qkv'), ('o_proj', 'gemm.o'),
                       ('lm_head', 'gemm.lm_head')):
        if key in name:
            return label
    return 'gemm.other'


def time_prefill(engine, launches):
    stream = torch.cuda.current_stream()
    samples = []
    for _ in range(launches):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        engine.prefill.launch(0, 1, stream.cuda_stream)
        end.record(stream)
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples


def table(path):
    latest = path.with_name(path.stem + '_latest.tsv')
    with (latest if latest.exists() else path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def attribute(folder):
    stages = {int(r['stage']): r for r in table(folder / 'runtime_stages.tsv')}
    begin = defaultdict(lambda: None)
    for r in table(folder / 'step_trace.tsv'):
        if int(r['kernel_begin']):
            i, t = int(r['iteration']), int(r['kernel_begin'])
            begin[i] = t if begin[i] is None else min(begin[i], t)
    release = defaultdict(int)
    for r in table(folder / 'stage_trace.tsv'):
        if int(r['t_release']):
            key = int(r['iteration']), int(r['stage'])
            release[key] = max(release[key], int(r['t_release']))
    launches = []
    for iteration in sorted(begin):
        previous, rows = begin[iteration], defaultdict(lambda: [0.0, 0])
        for stage in sorted(s for i, s in release if i == iteration):
            end = release[iteration, stage]
            info = stages[stage]
            row = rows[category(info['kind_name'], info['name'])]
            row[0] += (end - previous) / 1e3
            row[1] += 1
            previous = end
        launches.append(dict(iteration=iteration, total_us=(previous - begin[iteration]) / 1e3,
                             categories={k: dict(us=v[0], stages=v[1]) for k, v in rows.items()}))
    return launches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True, help='frozen (uninstrumented) prefill')
    p.add_argument('--trace-prefill', type=Path, required=True)
    p.add_argument('--decode', type=Path, required=True, help='decode library the prefill pairs with')
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--launches', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    checkpoint = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    os.environ['TILEMEGA_TRACE_LAUNCHES'] = str(max(64, a.launches))
    prompt = torch.tensor(json.loads((ROOT / 'configs/prompts/qwen_isl64.json').read_text())[:a.batch],
                          dtype=torch.int32)
    a.out.mkdir(parents=True, exist_ok=True)
    report = dict(batch=a.batch, launches=a.launches, diagnostic_only=True, prefill=str(a.prefill),
                  trace_prefill=str(a.trace_prefill))
    for label, library in (('reference', a.prefill), ('trace', a.trace_prefill)):
        with ServingEngine(checkpoint, library, a.decode, a.batch, prompt_len=prompt.shape[1], max_new_tokens=2,
                           mode='L1', prefill_mode='L1', decode_loop=False, step_events=False) as engine:
            engine.state.kv_storage.zero_()
            engine.state.tokens[:, :prompt.shape[1]].copy_(prompt)
            samples = time_prefill(engine, a.launches + 1)[1:]
            report[label + '_ms'] = dict(samples=samples, median=statistics.median(samples))
            if label == 'trace':
                dump = engine.prefill_lib.lib.tm_plan_dump_serving_trace
                dump.argtypes = [C.c_void_p, C.c_char_p]
                dump.restype = C.c_int
                (a.out / 'trace').mkdir(exist_ok=True)
                status = dump(engine.prefill.handle, os.fsencode(str((a.out / 'trace').resolve())))
                if status:
                    raise RuntimeError(f'prefill trace dump failed: {status}')
        del engine
        torch.cuda.empty_cache()
    report['overhead_pct'] = (report['trace_ms']['median'] / report['reference_ms']['median'] - 1) * 100
    launches = attribute(a.out / 'trace')[1:]
    totals = defaultdict(list)
    for launch in launches:
        for name, row in launch['categories'].items():
            totals[name].append(row['us'])
    report['by_category_us'] = {k: statistics.median(v) for k, v in sorted(totals.items(), key=lambda x: -sum(x[1]))}
    report['stages_per_category'] = {k: v['stages'] for k, v in launches[-1]['categories'].items()} if launches else {}
    report['launches_traced'] = launches
    atomic_json(a.out / 'm8.json', report)
    print(json.dumps({k: report[k] for k in ('reference_ms', 'overhead_pct', 'by_category_us')}, default=str)[:4000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
