#!/usr/bin/env python3
"""Retain native pricing curves while omitting unused diagnostic observations."""
import argparse
import json
from pathlib import Path

from sm90d1_common import ROOT, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('keep the source calibration and use a new evidence directory')
    target = read(a.target)
    native = target['sm90_dense_decode']
    metrics = {'v3_sparse_last_arrival_to_last_departure', 'bulk_stream_per_active_cta',
        'activation_wait_and_shared_read', 'native_ldmatrix_mma_k16',
        'epilogue_store_return_at_input_k', 'epilogue_swiglu_return_at_input_k',
        'epilogue_argmax_return_at_input_k', 'epilogue_residual_return',
        'activation_inactive_rows_initialize', 'bulk_single_completion',
        'attention_critical_before_barrier'}
    index = native['observation_columns'].index('metric')
    before = native['observations']
    retained = [r for r in before if (r[index] if isinstance(r, list) else r['metric']) in metrics]
    if {r[index] if isinstance(r, list) else r['metric'] for r in retained} != metrics:
        raise ValueError('one of the required native price curves is absent')
    if all(isinstance(row, list) for row in retained):
        columns = native['observation_columns']
        live = [i for i in range(len(columns)) if any(row[i] is not None for row in retained)]
        native['observation_columns'] = [columns[i] for i in live]
        retained = [[row[i] for i in live] for row in retained]
    native['observations'] = retained
    a.out.mkdir(parents=True)
    output = a.out / 'target.json'
    output.write_text(json.dumps(target, separators=(',', ':')) + '\n')
    write(a.out / 'receipt.json', dict(source=str(a.target.resolve()), source_sha256=sha(a.target),
        target_sha256=sha(output), source_bytes=a.target.stat().st_size,
        compact_bytes=output.stat().st_size, observations_before=len(before),
        observations_after=len(retained), retained_metrics=sorted(metrics),
        source_fit_status=native['fit_status'], solver_gate_pass=False,
        evidence='diagnostic compaction only; predictions require an equivalence audit'))
    print(output)


if __name__ == '__main__':
    main()
