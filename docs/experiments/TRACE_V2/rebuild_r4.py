#!/usr/bin/env python3
"""Recompute R3 traces without changing any historical dump or report."""
import argparse
import csv
import hashlib
from pathlib import Path

import analyze

REPO = Path(__file__).resolve().parents[3]


def inventory():
    rows = []
    for d in sorted((REPO / 'docs/experiments/PLACE_EFT2/raw/final').glob('trace_*')):
        _, config, model, seq, candidate = d.name.split('_')
        if candidate == 'chain':
            source = REPO / f'docs/experiments/PLACE_EFT2/raw/plan/{model}_{seq}_chain.cu'
        else:
            source = REPO / f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu'
        rows.append(dict(dump=str(d.relative_to(REPO)), source=str(source.relative_to(REPO)),
                         window=1 if config in ('a', 'b') else 2,
                         config=config, candidate=candidate, suite='placement'))
    for d in sorted((REPO / 'docs/experiments/WINDOW/raw/final').glob('trace_*')):
        _, window, model, seq = d.name.split('_')
        source = REPO / f'docs/experiments/WINDOW/raw/src/{model}_{window}.cu'
        rows.append(dict(dump=str(d.relative_to(REPO)), source=str(source.relative_to(REPO)),
                         window=int(window[1:]), config=window, candidate='legacy_grid_stride',
                         suite='window'))
    if sum(r['suite'] == 'placement' for r in rows) != 32:
        raise ValueError('expected exactly 32 R3 placement dumps')
    if sum(r['suite'] == 'window' for r in rows) != 12:
        raise ValueError('expected exactly 12 R3 window dumps')
    return rows


def write_tsv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'r4_rebuild')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs, results, hashes = inventory(), [], []
    for item in inputs:
        source, dump = REPO / item['source'], REPO / item['dump']
        result = analyze.analyze(dump, source, item['window'])
        result['task_dag_source'] = item['source']
        result.update(item)
        results.append(result)
        for path in [source, *(dump / n for n in ('slots.tsv', 'waits.tsv', 'events.tsv', 'meta.tsv'))]:
            hashes.append(dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f"REBUILD {item['dump']} cp_ns={result['cp_corrected_ns']} nodes={result['cp_corrected_nodes']}", flush=True)
    write_tsv(args.out / 'manifest.tsv', inputs)
    write_tsv(args.out / 'inputs_sha256.tsv', hashes)
    write_tsv(args.out / 'analysis.tsv', results)
    gates = []
    for r in results:
        floor = max(r['cp_corrected_ns'], r['queue_lb_ns'])
        split = [r['cp_split_' + key + '_ns'] for key in ('task', 'wait', 'prerun_barrier', 'publish', 'gap')]
        gates.append(dict(dump=r['dump'], a_a=int(r['measured_l2_ms'] * 1e6 >= floor),
                          a_b=int(min(split) >= 0 and abs(sum(split)-r['cp_reconstructed_ns']) <= .01*r['cp_reconstructed_ns']
                                  and r['cp_split_wait_ns'] <= r['kernel_span_ns']),
                          measured_ns=r['measured_l2_ms']*1e6, floor_ns=floor,
                          split_sum_ns=sum(split), reconstructed_ns=r['cp_reconstructed_ns']))
    write_tsv(args.out / 'gates.tsv', gates)
    pair = [r for r in results if r['suite']=='placement' and r['config']=='a' and r['model']=='gqa2' and r['seq']=='4']
    a_c = len(pair)==2 and len({(r['cp_corrected_nodes'],r['cp_corrected_task_ns']) for r in pair})==1
    print(f"A-a {sum(g['a_a'] for g in gates)}/{len(gates)} A-b {sum(g['a_b'] for g in gates)}/{len(gates)} A-c {'PASS' if a_c else 'FAIL'}")
    return int(not a_c or any(not g['a_a'] or not g['a_b'] for g in gates))


if __name__ == '__main__':
    raise SystemExit(main())
