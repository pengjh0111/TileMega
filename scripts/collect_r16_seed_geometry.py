#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Collect R16 seed-geometry TileMega cells: three fresh protocol rounds plus checks."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics

SYSTEMS = ('vllm', 'sglang', 'tensorrt-llm', 'torch.compile')


def load_mbu(root):
    spec = importlib.util.spec_from_file_location('mbu', root/'scripts/estimate_r16_mbu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(directory, model, batch, variant):
    rounds, tokens = [], []
    for r in range(3):
        path = directory/f'final-{variant}'/f'round{r}'/f'B{batch}'
        file = path/'measurements.json'
        if not file.exists():
            return None
        m = json.loads(file.read_text())
        rounds.append(m)
        tokens.append((path/'tokens_N1024_run3.json').read_text())
    check = directory/f'check-{variant}'/'check-summary.json'
    check = json.loads(check.read_text()) if check.exists() else None
    median = lambda key: statistics.median(r[key] for r in rounds)
    return dict(model=model, batch=batch, policy=variant,
                gpus=';'.join(str(r['gpu_index']) for r in rounds),
                e2e_ms=1e3*median('e2e_seconds'), ttft_ms=1e3*median('ttft_seconds'),
                tpot_ms=1e3*median('tpot_seconds'), tokens_per_second=median('output_tokens_per_second'),
                tpot_range_pct=100*(max(r['tpot_seconds'] for r in rounds)-min(r['tpot_seconds'] for r in rounds))/median('tpot_seconds'),
                cross_process_tokens_equal=len(set(tokens)) == 1,
                repeat8_mismatches=check and check['repeat8'] and check['repeat8']['mismatches'],
                c1_pass=check and check['c1'] and check['c1']['pass'],
                c1_max_gap=check and check['c1'] and check['c1']['max_gap'],
                c2_pass=check and check['c2'] and check['c2']['pass'],
                acceptance=rounds[0]['acceptance'],
                checkpoint_sha256=rounds[0]['checkpoint']['checkpoint_sha256'][:16])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/TileMega'))
    p.add_argument('--cells', type=Path, required=True, help='lines: model batch policy [build root]')
    p.add_argument('--builds', type=Path, required=True)
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--measured-gbps', type=float, default=3131.57)
    p.add_argument('--nominal-gbps', type=float, default=3350.0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    mbu = load_mbu(a.root)
    base = {}
    for row in csv.DictReader(a.baselines.open()):
        base[(row['model'], int(row['batch']), row['system'])] = row
    rows, ratios = [], {s: [] for s in SYSTEMS}
    for line in a.cells.read_text().split('\n'):
        if not line.strip():
            continue
        model, batch, variant, *root = line.split()
        r = cell((Path(root[0]) if root else a.builds)/model/f'B{batch}', model, int(batch), variant)
        if r is None:
            continue
        config = json.loads((a.root/'runs/r16/models'/model/'config.json').read_text())
        payload = mbu.dense_payload(config, r['batch'])
        if payload:
            step = payload['logical_bytes_per_decode']
            r['mbu_measured'] = step/(r['tpot_ms']*1e-3)/(a.measured_gbps*1e9)
            r['mbu_nominal'] = step/(r['tpot_ms']*1e-3)/(a.nominal_gbps*1e9)
        valid = bool(r['c1_pass'] and r['c2_pass'] and r['repeat8_mismatches'] == 0 and r['cross_process_tokens_equal'])
        r['valid'] = valid
        for s in SYSTEMS:
            b = base.get((model, r['batch'], s))
            if b:
                r[f'{s}_tpot_ms'] = float(b['tpot_ms'])
                r[f'speedup_tpot_vs_{s}'] = float(b['tpot_ms'])/r['tpot_ms']
                r[f'speedup_e2e_vs_{s}'] = float(b['e2e_ms'])/r['e2e_ms']
                if valid:
                    ratios[s].append((float(b['tpot_ms'])/r['tpot_ms'], float(b['e2e_ms'])/r['e2e_ms'],
                                      r['tokens_per_second']/float(b['tokens_per_second'])))
        rows.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with a.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    geo = lambda xs: math.exp(sum(map(math.log, xs))/len(xs)) if xs else float('nan')
    summary = {s: dict(cells=len(v), tpot=geo([x[0] for x in v]), e2e=geo([x[1] for x in v]),
                       throughput=geo([x[2] for x in v])) for s, v in ratios.items()}
    a.out.with_suffix('.geomeans.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(dict(cells=len(rows), valid=sum(r['valid'] for r in rows), geomeans=summary), indent=1))


if __name__ == '__main__':
    main()
