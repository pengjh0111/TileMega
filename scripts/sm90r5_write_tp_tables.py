#!/usr/bin/env python3
"""SM90R5 TP result tables (T7, T8) from the solve receipts and final/corr runs.

T7 `SM90R5_tp_final.csv`: one row per (tp, batch) with the P-FINAL TPOT, the
per-launch values, the winner's coordinates and its correctness verdicts.
T8 `SM90R5_tp_ratios.csv`: the same cells against each TP baseline engine and
against TP1, with the scaling efficiency TP1 / (T x TPOT) for TileMega and for
every engine measured at that degree.

A cell with no measurement yet is written with status and empty metrics rather
than skipped, so the table always covers the whole matrix.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, read

SOLVE = ROOT / 'runs/sm90r5/tp/solve'
FINAL = ROOT / 'runs/sm90r5/tp/final'
CORR = ROOT / 'runs/sm90r5/tp/corr'
BATCHES = (1, 2, 4, 8, 16)
MODEL = 'qwen3_1_7b'


def brief(coordinates):
    """The coordinate fields the R3/R4 final tables carry, as one string each."""
    gemms = coordinates['gemms']
    attention = coordinates['attention']
    return dict(
        qkv_tn=gemms['qkv']['tile_n'], o_tn=gemms['o']['tile_n'],
        gate_up_tn=gemms['gate_up']['tile_n'], down_tn=gemms['down']['tile_n'],
        lm_head_tn=gemms['lm_head']['tile_n'],
        attention=f"S{attention['chunks']}/{attention['block_keys']}keys/"
                  f"{attention['slots']}slots/{attention['merge']}",
        emb=coordinates['emb'], group_wait=coordinates['group_wait'],
        cluster=coordinates['cluster'], residency=coordinates['residency'])


def tp1_reference():
    """REF-R3's TileMega TP1 cells and the frozen TP1 baselines."""
    tilemega, engines = {}, {}
    for row in csv.DictReader(open(ROOT / 'docs/results/SM90R3_llm_final.csv')):
        if row['model'] == MODEL:
            tilemega[int(row['batch'])] = float(row['tpot_ms'])
    for row in csv.DictReader(open(ROOT / 'docs/results/R16_final_baselines.csv')):
        if row['model'] == MODEL and row['tpot_ms'] not in ('', 'N/A'):
            engines.setdefault(row['system'], {})[int(row['batch'])] = float(row['tpot_ms'])
    return tilemega, engines


def baselines():
    path = ROOT / 'docs/results/SM90R5_tp_baselines.csv'
    result = {}
    if not path.exists():
        return result
    for row in csv.DictReader(open(path)):
        if row['tpot_ms'] not in ('', 'N/A'):
            result.setdefault((int(row['tp']), int(row['batch'])), {})[row['system']] = float(row['tpot_ms'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tps', type=int, nargs='+', default=[2, 4])
    p.add_argument('--final-csv', type=Path, default=ROOT / 'docs/results/SM90R5_tp_final.csv')
    p.add_argument('--ratios-csv', type=Path, default=ROOT / 'docs/results/SM90R5_tp_ratios.csv')
    a = p.parse_args()
    tp1_tilemega, tp1_engines = tp1_reference()
    external = baselines()
    final_rows, ratio_rows = [], []
    for tp in a.tps:
        for batch in BATCHES:
            receipt = SOLVE / f'tp{tp}-B{batch}/decode.so.search/receipt.json'
            final = FINAL / f'tp{tp}-B{batch}/final.json'
            corr = CORR / f'tp{tp}-B{batch}/corr.json'
            row = dict(model=MODEL, tp=tp, batch=batch)
            if not receipt.exists():
                final_rows.append(dict(row, status='solve pending'))
                continue
            data = read(receipt)
            # `measurement` is null until the selection is accepted.
            measurement = data.get('measurement') or {}
            winner = measurement.get('winner_rank')
            coordinates = next((item['coordinates'] for item in data['ranked']
                                if item['rank'] == winner), data['ranked'][0]['coordinates'])
            row.update(brief(coordinates), winner_rank=winner,
                       psel_arms=json.dumps(measurement.get('arms', {})),
                       predicted_step_ns=data['ranked'][0].get('predicted_ns', ''),
                       target_sha256=data['target_sha256'])
            if final.exists():
                measured = read(final)
                row.update(status='complete', tpot_ms=f"{measured['tpot_ms']:.4f}",
                           per_launch_ms=';'.join(f'{v:.4f}' for v in measured['per_launch_ms']),
                           range_pct=f"{measured['range_pct']:.3f}",
                           launches=len(measured['per_launch_ms']),
                           ttft_ms=f"{measured['ttft_ms']:.3f}", e2e_ms=f"{measured['e2e_ms']:.1f}",
                           binary_sha256=measured['binary_sha256'])
            else:
                row.update(status='P-FINAL pending')
            if corr.exists():
                verdict = read(corr)
                row.update(c1_pass=verdict['c1_pass'], c1_max_gap=verdict['c1_max_gap'],
                           c2_pass=verdict['c2_pass'], c2_launches=verdict['c2_launches'],
                           step_loop_pass=verdict['step_loop'])
            final_rows.append(row)
            if 'tpot_ms' not in row:
                continue
            tpot = float(row['tpot_ms'])
            reference = tp1_tilemega.get(batch)
            ratio = dict(model=MODEL, tp=tp, batch=batch, tpot_ms=f'{tpot:.4f}',
                         tp1_tilemega_ms=f'{reference:.4f}' if reference else '',
                         tilemega_scaling=f'{reference / (tp * tpot):.4f}' if reference else '')
            for system, value in sorted(external.get((tp, batch), {}).items()):
                ratio[f'{system}_ms'] = f'{value:.4f}'
                ratio[f'ratio_vs_{system}'] = f'{value / tpot:.4f}'
                tp1_value = tp1_engines.get(system, {}).get(batch)
                if tp1_value:
                    ratio[f'{system}_scaling'] = f'{tp1_value / (tp * value):.4f}'
            measured = external.get((tp, batch), {})
            if measured:
                fastest = min(measured, key=measured.get)
                ratio.update(fastest_engine=fastest,
                             ratio_vs_fastest=f'{measured[fastest] / tpot:.4f}')
            ratio_rows.append(ratio)
    for path, rows in ((a.final_csv, final_rows), (a.ratios_csv, ratio_rows)):
        if not rows:
            continue
        fields = []
        for row in rows:
            fields += [key for key in row if key not in fields]
        with open(path, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(dict(csv=str(path), rows=len(rows))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
