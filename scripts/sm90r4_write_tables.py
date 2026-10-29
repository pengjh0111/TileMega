#!/usr/bin/env python3
"""Write the SM90R4 report tables (prompt §10) from the run directory.

Each table is written only when its inputs exist, so the script is safe to
re-run as measurements land. T8 (constants) is written by the constants
collector; this script writes the rest.
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

R = Path('/root/TileMega/runs/sm90r4')
ROOT = Path('/root/TileMega-r4')
ENGINES = ('vllm', 'sglang', 'tensorrt-llm', 'torch.compile')
RATIO_ENGINES = ('vllm', 'sglang', 'tensorrt-llm')
BATCHES = (1, 2, 4, 8, 16)
NEW = ('mistral_7b_v0_3', 'olmoe_1b_7b')
SYSTEM = {'vllm': 'vllm', 'sglang': 'sglang', 'trtllm': 'tensorrt-llm',
          'torchcompile': 'torch.compile'}


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})
    print(f'{path.name}: {len(rows)} rows')
    return len(rows)


def new_baselines():
    """T3: the measured cells of the two new models, three fresh processes each."""
    found = {}
    for path in R.glob('baselines/gpu*/raw/*/attempt*/job/measurements/B*/measurements.json'):
        value = json.loads(path.read_text())
        parts = path.parts
        name = parts[parts.index('raw') + 1]
        model = next((m for m in NEW if m in name), None)
        engine = next((s for s in SYSTEM if f'_{s}_' in name), None)
        if model is None or engine is None:
            continue
        # The batch is the measurement directory (B<n>), not a field.
        key = (model, SYSTEM[engine], int(path.parent.name.lstrip('B')))
        found.setdefault(key, []).append(value)
    rows = []
    for (model, system, batch) in sorted(found):
        values = found[(model, system, batch)]
        checkpoints = {v.get('checkpoint', {}).get('checkpoint_sha256') for v in values}
        rows.append(dict(model=model, system=system, batch=batch,
                         fresh_processes=len(values),
                         tpot_ms=f"{st.median(v['tpot_seconds'] * 1e3 for v in values):.4f}",
                         ttft_ms=f"{st.median(v['ttft_seconds'] * 1e3 for v in values):.4f}",
                         e2e_ms=f"{st.median(v['e2e_seconds'] * 1e3 for v in values):.3f}",
                         tokens_per_second=f"{st.median(v['output_tokens_per_second'] for v in values):.3f}",
                         process_range_pct=f"{(max(v['tpot_seconds'] for v in values) / min(v['tpot_seconds'] for v in values) - 1) * 100:.2f}",
                         checkpoint_sha256=(checkpoints.pop() if len(checkpoints) == 1 else 'MIXED'),
                         versions=json.dumps(values[0].get('versions', {}), separators=(',', ':'))[:120],
                         source='new (R16 protocol, SM90R4)'))
    # Engines that refuse a model are N/A rows with their own error.
    for model, system, reason in unsupported():
        for batch in BATCHES:
            rows.append(dict(model=model, system=system, batch=batch, fresh_processes=0,
                             tpot_ms='N/A', ttft_ms='N/A', e2e_ms='N/A', tokens_per_second='N/A',
                             source='new (R16 protocol, SM90R4)', error=reason))
    return rows


def unsupported():
    path = R / 'baselines/unsupported.json'
    if not path.exists():
        return []
    return [(row['model'], row['system'], row['error']) for row in json.loads(path.read_text())]


def moe_ledger():
    """T7: MR0's per-stage-type ledger against the prediction and the bound."""
    rows = []
    for batch in (1, 16):
        path = R / f'mr0/ledger-B{batch}.json'
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        run = next(R.glob(f'mr0/raw/mr0-ledger-B{batch}/attempt*/job/trace/run.json'), None)
        overhead = json.loads(run.read_text()).get('overhead_pct') if run else None
        for entry in value['rows']:
            rows.append(dict(stage=entry['stage'], model='qwen3_30b_a3b', batch=batch,
                             arm='REF-R3', steps=value['steps'],
                             measured_us=entry['measured_us'], predicted_us=entry['predicted_us'],
                             bound_us=entry['bound_us'], over_prediction_us=entry['over_prediction_us'],
                             trace_overhead_pct='' if overhead is None else f'{overhead:.2f}'))
    return rows


def pairs():
    """Every collected P-PAIR: the two arms, the change and the branch."""
    rows = []
    for path in sorted(R.glob('pair/*/result.json')):
        value = json.loads(path.read_text())
        spec = value['spec']
        arms = list(spec['arms'])
        rows.append(dict(name=spec['name'], model=spec['model'], batch=spec['batch'],
                         gpu=spec['gpu'], concurrency=spec.get('concurrency', 'serial'),
                         arm_a=arms[0], arm_b=arms[1],
                         tpot_a_ms=f"{value['arms'][arms[0]]['tpot_ms']:.4f}",
                         tpot_b_ms=f"{value['arms'][arms[1]]['tpot_ms']:.4f}",
                         range_a_pct=f"{value['arms'][arms[0]]['range_pct']:.2f}",
                         range_b_pct=f"{value['arms'][arms[1]]['range_pct']:.2f}",
                         processes=value['arms'][arms[0]]['processes'],
                         change_pct=f"{value['change_pct']:.2f}",
                         threshold_pct=f"{value['threshold_pct']:.2f}",
                         decision=value['decision']))
    return rows


def final_cells(tag):
    """T1/T5: accepted cells written by the config writer."""
    rows = []
    for path in sorted((ROOT / f'configs/serving/{tag}').glob('*.json')):
        value = json.loads(path.read_text())
        measured = value.get('measured', {})
        correctness = value.get('correctness', {})
        rows.append(dict(model=value['model'], batch=value['batch'],
                         tpot_ms=measured.get('tpot_ms', ''), ttft_ms=measured.get('ttft_ms', ''),
                         e2e_ms=measured.get('e2e_ms', ''),
                         process_range_pct=measured.get('process_range_pct', ''),
                         fresh_processes=measured.get('fresh_processes', ''),
                         c1=correctness.get('c1', ''), c2=correctness.get('c2', ''),
                         config=path.name, binary_sha256=value.get('binary_sha256', '')[:16]))
    return rows


def ratios(cells, base, source):
    """T2/T6: per-engine ratios plus the fastest engine, with the baseline source."""
    rows = []
    for cell in cells:
        model, batch = cell['model'], int(cell['batch'])
        if not cell.get('tpot_ms'):
            continue
        ours = float(cell['tpot_ms'])
        row = dict(model=model, batch=batch, tpot_ms=f'{ours:.4f}', baseline_source=source)
        fastest = None
        for engine in ENGINES:
            value = base.get((model, batch, engine))
            row[f'ratio_{engine}'] = f'{value / ours:.4f}' if value else 'N/A'
            if value and engine in RATIO_ENGINES and (fastest is None or value < fastest):
                fastest = value
        row['ratio_fastest'] = f'{fastest / ours:.4f}' if fastest else 'N/A'
        rows.append(row)
    return rows


def baseline_table():
    base = {}
    frozen = ROOT / 'docs/results/R16_final_baselines.csv'
    if frozen.exists():
        for row in csv.DictReader(frozen.open()):
            if row['system'] in ENGINES:
                base[(row['model'], int(row['batch']), row['system'])] = float(row['tpot_ms'])
    new = ROOT / 'docs/results/SM90R4_baselines_new.csv'
    if new.exists():
        for row in csv.DictReader(new.open()):
            if row['tpot_ms'] not in ('', 'N/A'):
                base[(row['model'], int(row['batch']), row['system'])] = float(row['tpot_ms'])
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'docs/results')
    a = p.parse_args()
    fields3 = ['model', 'system', 'batch', 'fresh_processes', 'tpot_ms', 'ttft_ms', 'e2e_ms',
               'tokens_per_second', 'process_range_pct', 'checkpoint_sha256', 'versions', 'source', 'error']
    rows = new_baselines()
    if rows:
        write(a.out / 'SM90R4_baselines_new.csv', rows, fields3)
    rows = moe_ledger()
    if rows:
        write(a.out / 'SM90R4_moe_ledger.csv', rows,
              ['stage', 'model', 'batch', 'arm', 'steps', 'measured_us', 'predicted_us', 'bound_us',
               'over_prediction_us', 'trace_overhead_pct'])
    rows = pairs()
    if rows:
        write(a.out / 'SM90R4_pairs.csv', rows,
              ['name', 'model', 'batch', 'gpu', 'concurrency', 'arm_a', 'arm_b', 'tpot_a_ms', 'tpot_b_ms',
               'range_a_pct', 'range_b_pct', 'processes', 'change_pct', 'threshold_pct', 'decision'])
    cells = final_cells('sm90r4')
    if cells:
        write(a.out / 'SM90R4_llm_final.csv', cells,
              ['model', 'batch', 'tpot_ms', 'ttft_ms', 'e2e_ms', 'process_range_pct', 'fresh_processes',
               'c1', 'c2', 'config', 'binary_sha256'])
        base = baseline_table()
        new_models = [c for c in cells if c['model'] in NEW]
        frozen_models = [c for c in cells if c['model'] not in NEW]
        table = (ratios(new_models, base, 'new') +
                 ratios(frozen_models, base, 'frozen R16_final_baselines.csv'))
        if table:
            write(a.out / 'SM90R4_llm_ratios.csv', table,
                  ['model', 'batch', 'tpot_ms', 'baseline_source'] +
                  [f'ratio_{e}' for e in ENGINES] + ['ratio_fastest'])


if __name__ == '__main__':
    main()
