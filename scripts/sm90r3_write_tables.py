#!/usr/bin/env python3
"""Write the SM90R3 report tables (prompt §14): T1-T4, T6, T7.

T5 (constants) is written by the constants collector and T8 (resources) by the
build receipts; T9/T10 belong to Phase N. Every ratio is against the frozen
historical baselines, per engine, with no anchor.
"""
import argparse
import csv
import json
import math
from pathlib import Path

R = Path('/root/TileMega/runs/sm90r3')
M = Path('/root/TileMega-m')
ENGINES = ('vllm', 'sglang', 'tensorrt-llm', 'torch.compile')
RATIO_ENGINES = ('vllm', 'sglang', 'tensorrt-llm')
DENSE = ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b')
MOE = 'qwen3_30b_a3b'
BATCHES = (1, 2, 4, 8, 16)
STAGES = ('ref_r2', 'ref0', 'ref1', 'ref2', 'ref3', 'final')


def write(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})
    return len(rows)


def baselines():
    table = {}
    for row in csv.DictReader((M / 'docs/results/R16_final_baselines.csv').open()):
        if row['system'] in ENGINES:
            table[(row['model'], int(row['batch']), row['system'])] = dict(
                tpot_ms=float(row['tpot_ms']), ttft_ms=float(row['ttft_ms']),
                e2e_ms=float(row['e2e_ms']))
    return table


def cells():
    return [(m, b) for m in DENSE for b in BATCHES] + [(MOE, b) for b in BATCHES]


def result(kind, cell):
    path = R / f'v/results/{kind}-v-{cell}.json'
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=M / 'docs/results')
    a = p.parse_args()
    base = baselines()
    t1, t2, t4, t7 = [], [], [], []
    ratios = {}
    for model, batch in cells():
        cell = f'{model}-B{batch}'
        final, corr = result('final', cell), result('corr', cell)
        config = M / f'configs/serving/sm90r3/{model}_b{batch}.json'
        if not final or not config.exists():
            continue
        arm = next(iter(final['arms'].values()))
        c = json.loads(config.read_text())
        coordinates = c['coordinates']
        key = lambda name: coordinates['gemms'].get(name, {})
        t1.append(dict(model=model, batch=batch, tpot_ms=round(arm['tpot_ms'], 4),
                       ttft_ms=round(arm['ttft_ms'], 3), e2e_ms=round(arm['e2e_ms'], 2),
                       tokens_per_s=round(arm['tokens_per_s'], 1),
                       process_range_pct=round(arm['range_pct'], 3),
                       fresh_processes=arm['processes'],
                       c1_pass=corr and corr['c1_pass'], c2_pass=corr and corr['c2_pass'],
                       processes_token_equal=corr and corr['processes_token_equal'],
                       residency=coordinates['residency'], cluster=coordinates.get('cluster', 1),
                       group_wait=coordinates.get('group_wait'), emb=coordinates.get('emb'),
                       layer_loop=coordinates.get('layer_loop'),
                       weight_stream=coordinates.get('weight_stream'),
                       attention=f"S{coordinates['attention']['chunks']}/"
                                 f"{coordinates['attention']['block_keys']}keys/"
                                 f"{coordinates['attention']['slots']}slots/"
                                 f"{coordinates['attention'].get('merge', 'la')}",
                       qkv_tn=key('qkv').get('tile_n'), o_tn=key('o').get('tile_n'),
                       lm_head_tn=key('lm_head').get('tile_n'),
                       winner=c['winner'], binary_sha256=c['binary_sha256'][:16]))
        row = dict(model=model, batch=batch, tpot_ms=round(arm['tpot_ms'], 4))
        best = None
        for engine in RATIO_ENGINES:
            value = base.get((model, batch, engine))
            if not value:
                continue
            row['ratio_' + engine] = round(value['tpot_ms'] / arm['tpot_ms'], 4)
            best = value['tpot_ms'] if best is None else min(best, value['tpot_ms'])
        if (model, batch, 'torch.compile') in base:
            row['ratio_torch_compile'] = round(
                base[(model, batch, 'torch.compile')]['tpot_ms'] / arm['tpot_ms'], 4)
        if best:
            row['ratio_fastest'] = round(best / arm['tpot_ms'], 4)
            ratios.setdefault('moe' if model == MOE else 'dense', []).append(row['ratio_fastest'])
        t2.append(row)
        if corr:
            row7 = dict(model=model, batch=batch, protocol='P-CORR',
                        c1_positions=1024 if model != MOE else 256,
                        c1_pass=corr['c1_pass'], c2_pass=corr['c2_pass'],
                        fresh_processes=corr['fresh_processes'],
                        processes_token_equal=corr['processes_token_equal'],
                        hf64_recorded=corr.get('hf64_pass'),
                        max_gap=max((r['c1']['max_gap'] for r in corr['rows'] if r.get('c1')),
                                    default=''))
            t7.append(row7)
        receipt = M / f'runs/sm90r3/solve/FINAL/{cell}/decode.so.search/receipt.json'
        if receipt.exists():
            ranked = json.loads(receipt.read_text())['ranked']
            measured = json.loads((M / f'runs/sm90r3/solve/FINAL/{cell}'
                                   '/decode.so.search/measure/psel.json').read_text())
            order = sorted(measured['arms'], key=lambda k: measured['arms'][k]['tpot_ms'])
            for index, candidate in enumerate(ranked):
                name = f'c{index}'
                if name not in measured['arms']:
                    continue
                t4.append(dict(model=model, batch=batch, candidate=name,
                               origin=candidate['origin'],
                               predicted_ns=round(candidate['prediction']['step_ns'], 1),
                               measured_ms=round(measured['arms'][name]['tpot_ms'], 4),
                               tau=round(measured['arms'][name]['tpot_ms'] * 1e6 /
                                         candidate['prediction']['step_ns'], 4),
                               measured_rank=order.index(name) + 1,
                               selected=name == measured['winner'],
                               threshold_pct=round(measured['threshold_pct'], 3)))
    t3 = []
    for cell in ('qwen3_0_6b-B1', 'qwen3_0_6b-B16', 'llama3_2_1b-B1', 'llama3_2_1b-B16',
                 'llama3_2_3b-B4', 'llama3_1_8b-B1', 'llama3_1_8b-B16'):
        path = R / f'v/results/ablation-v-{cell}.json'
        if not path.exists():
            continue
        arms = json.loads(path.read_text())['arms']
        reference = arms.get('ref_r2', {}).get('tpot_ms')
        model, batch = cell.rsplit('-B', 1)
        for stage in STAGES:
            if stage not in arms:
                continue
            t3.append(dict(model=model, batch=int(batch), stage=stage,
                           tpot_ms=round(arms[stage]['tpot_ms'], 4),
                           speedup_vs_ref_r2=round(reference / arms[stage]['tpot_ms'], 4)
                           if reference else '',
                           process_range_pct=round(arms[stage]['range_pct'], 3),
                           fresh_processes=arms[stage]['processes']))
    decisions = json.loads((R / 'decisions.json').read_text()) if (R / 'decisions.json').exists() else []
    counts = {
        'T1': write(a.out / 'SM90R3_llm_final.csv', t1, list(t1[0]) if t1 else ['model']),
        'T2': write(a.out / 'SM90R3_llm_ratios.csv', t2,
                    ['model', 'batch', 'tpot_ms', 'ratio_vllm', 'ratio_sglang',
                     'ratio_tensorrt-llm', 'ratio_torch_compile', 'ratio_fastest']),
        'T3': write(a.out / 'SM90R3_ablation.csv', t3,
                    ['model', 'batch', 'stage', 'tpot_ms', 'speedup_vs_ref_r2',
                     'process_range_pct', 'fresh_processes']),
        'T4': write(a.out / 'SM90R3_solver.csv', t4,
                    ['model', 'batch', 'candidate', 'origin', 'predicted_ns', 'measured_ms',
                     'tau', 'measured_rank', 'selected', 'threshold_pct']),
        'T7': write(a.out / 'SM90R3_correctness.csv', t7,
                    ['model', 'batch', 'protocol', 'c1_positions', 'c1_pass', 'c2_pass',
                     'fresh_processes', 'processes_token_equal', 'hf64_recorded', 'max_gap']),
    }
    if decisions:
        counts['T6'] = write(a.out / 'SM90R3_decisions.csv', decisions,
                             ['decision', 'when', 'input', 'measurement', 'branch'])
    geo = {k: round(math.exp(sum(map(math.log, v)) / len(v)), 4) for k, v in ratios.items()}
    low = {k: round(min(v), 4) for k, v in ratios.items()}
    print(json.dumps(dict(rows=counts, geomean=geo, minimum=low), indent=1))


if __name__ == '__main__':
    main()
