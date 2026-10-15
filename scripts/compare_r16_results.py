#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compare complete R16 cells; keep every explicit native arm separate."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

LLM_METRICS = ('e2e_ms', 'ttft_ms', 'tpot_ms', 'tokens_per_second')


def complete(row):
    return str(row['protocol_complete']).lower() == 'true' and int(row['fresh_processes']) == 3


def positive(row, field):
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('nonpositive or nonfinite measurement: ' + field)
    return value


def compare(rows, family):
    rows = [row for row in rows if complete(row)]
    def is_native(row):
        return row['system'].startswith(('tilemega', 'TM-'))
    native = [row for row in rows if is_native(row)]
    baselines = [row for row in rows if not is_native(row)]
    output = []
    for tm in native:
        for base in baselines:
            if (tm['model'], int(tm['batch'])) != (base['model'], int(base['batch'])):
                continue
            if family == 'dnn' and (not tm['exported_program_sha256'] or
                    tm['exported_program_sha256'] != base['exported_program_sha256']):
                continue
            shared = family == 'dnn' or (bool(tm.get('source_checkpoint_sha256')) and
                    tm.get('source_checkpoint_sha256') == base.get('source_checkpoint_sha256'))
            row = dict(model=tm['model'], batch=tm['batch'], native_system=tm['system'],
                native_candidate=tm.get('candidate_id', ''),
                native_comparison_arm=tm.get('comparison_arm') or tm.get('candidate_id') or
                    tm['system'] + '@' + tm.get('native_artifact_ids', ''),
                native_artifact_ids=tm.get('native_artifact_ids', ''),
                baseline_system=base['system'], baseline_mode=base.get('compile_mode', ''),
                native_precision=tm.get('precision', 'bf16'), baseline_precision=base.get('precision', 'bf16'),
                checkpoint_shared=shared,
                checkpoint_scope='identical exported program' if family == 'dnn' else
                    ('identical checkpoint hash' if shared else 'independent random initialization; MoE routing unpaired'),
                native_gpu=tm['gpu'], baseline_gpu=base['gpu'], same_gpu=tm['gpu'] == base['gpu'],
                acceptance='strict' if tm['acceptance'] == base['acceptance'] == 'strict' else
                    'provisional_container_pid_namespace', default_promoted=False)
            for field in LLM_METRICS if family == 'llm' else ('median_ms', 'p90_ms'):
                t, b = positive(tm, field), positive(base, field)
                row['native_' + field], row['baseline_' + field] = t, b
                row[field + '_speedup'] = t / b if field == 'tokens_per_second' else b / t
            output.append(row)
    return output


def geomeans(rows, family):
    groups = {}
    for row in rows:
        key = tuple(row[key] for key in ('native_system', 'native_comparison_arm', 'baseline_system',
                    'baseline_mode', 'native_precision', 'baseline_precision', 'checkpoint_scope', 'acceptance'))
        groups.setdefault(key, []).append(row)
    output = []
    for key, cells in sorted(groups.items()):
        coordinates = {(cell['model'], int(cell['batch'])) for cell in cells}
        if len(coordinates) != len(cells):
            raise ValueError('ambiguous native artifacts for the same comparison coordinate')
        output.append(dict(zip(('native_system', 'native_comparison_arm', 'baseline_system', 'baseline_mode',
                    'native_precision', 'baseline_precision', 'checkpoint_scope', 'acceptance'), key),
            matched_cells=len(cells), coordinates=';'.join(f'{model}/B{batch}' for model, batch in sorted(coordinates)),
            **{field + '_speedup': math.exp(statistics.mean(math.log(cell[field + '_speedup']) for cell in cells))
               for field in (LLM_METRICS if family == 'llm' else ('median_ms', 'p90_ms'))}))
    return output


def write(path, rows):
    with path.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', type=Path)
    parser.add_argument('--dnn', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for family in ('llm', 'dnn'):
        source = getattr(args, family)
        if source is not None:
            with source.open() as stream:
                rows = compare(list(csv.DictReader(stream)), family)
            write(args.out / f'R16_{family}_ratios.csv', rows)
            write(args.out / f'R16_{family}_geomeans.csv', geomeans(rows, family))
            print(f'{family}: {len(rows)} matched comparisons; speedup >1 means faster TileMega')


if __name__ == '__main__':
    main()
