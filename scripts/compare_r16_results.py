#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compare complete R16 cells; keep every explicit native arm separate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

LLM_METRICS = ('e2e_ms', 'ttft_ms', 'tpot_ms', 'tokens_per_second')
PAIR_SCOPE = ('device_pairing', 'boot_pairing', 'native_gpu', 'baseline_gpu',
              'native_gpu_uuid', 'baseline_gpu_uuid', 'native_boot_id', 'baseline_boot_id')


def complete(row):
    return str(row['protocol_complete']).lower() == 'true' and int(row['fresh_processes']) == 3


def positive(row, field):
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('nonpositive or nonfinite measurement: ' + field)
    return value


def physical_pairing(tm, base):
    def uuid(row):
        return (str(row.get('physical_gpu_uuid', '')).removeprefix('GPU-').lower()
                if row.get('physical_gpu_identity_status') == 'verified' else '')
    left, right = uuid(tm), uuid(base)
    known = bool(left and right)
    boot_left, boot_right = tm.get('hardware_boot_id', ''), base.get('hardware_boot_id', '')
    boot_known = bool(boot_left and boot_right)
    return dict(native_gpu_uuid=left, baseline_gpu_uuid=right,
        native_boot_id=boot_left, baseline_boot_id=boot_right,
        same_gpu_index=str(tm['gpu']) == str(base['gpu']),
        same_gpu=known and left == right,
        device_pairing=('same_physical_gpu' if left == right else 'different_physical_gpu')
            if known else 'unknown_physical_gpu',
        boot_pairing=('same_boot' if boot_left == boot_right else 'different_boot')
            if boot_known else 'unknown_boot')


def compare(rows, family):
    rows = [row for row in rows if complete(row)]
    def is_native(row):
        return row['system'].lower().startswith(('tilemega', 'tm-'))
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
                native_temporary_workaround=tm.get('temporary_workaround', False),
                native_runtime_adjustment=tm.get('runtime_adjustment', ''),
                baseline_system=base['system'], baseline_mode=base.get('compile_mode', ''),
                native_precision=tm.get('precision', 'bf16'), baseline_precision=base.get('precision', 'bf16'),
                checkpoint_shared=shared,
                checkpoint_scope='identical exported program' if family == 'dnn' else
                    ('identical checkpoint hash' if shared else 'independent random initialization; MoE routing unpaired'),
                native_gpu=tm['gpu'], baseline_gpu=base['gpu'], **physical_pairing(tm, base),
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
                    'baseline_mode', 'native_precision', 'baseline_precision', 'checkpoint_scope', 'acceptance') + PAIR_SCOPE)
        groups.setdefault(key, []).append(row)
    output = []
    for key, cells in sorted(groups.items()):
        coordinates = {(cell['model'], int(cell['batch'])) for cell in cells}
        if len(coordinates) != len(cells):
            raise ValueError('ambiguous native artifacts for the same comparison coordinate')
        output.append(dict(zip(('native_system', 'native_comparison_arm', 'baseline_system', 'baseline_mode',
                    'native_precision', 'baseline_precision', 'checkpoint_scope', 'acceptance') + PAIR_SCOPE, key),
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


def candidate_rows(report):
    if report.get('evidence') != 'verified_independent_receipt_analysis':
        raise ValueError('independent candidate receipt analysis required')
    rows = []
    for item in report['candidates']:
        if not item['protocol_complete'] or item['fresh_processes'] != 3:
            continue
        if item['status'] != 'completed' or item.get('default_promoted') is not False:
            raise ValueError('unexpected candidate completion or promotion state')
        # Keep source, body candidate and execution coordinates in each arm;
        # separately compiled implementations must not share a geometric mean.
        arm = f"{item['source_head']}:{item['candidate']}:{item['executor']}:loop{item['device_loop']}"
        rows.append(dict(model=item['model'], batch=item['batch'], system='TileMega',
            candidate_id=item['candidate'], comparison_arm=arm, precision='bf16',
            fresh_processes=3, protocol_complete=True, gpu=str(item['gpu']),
            **{key: item.get(key, '') for key in ('physical_gpu_uuid', 'physical_gpu_identity_status', 'hardware_boot_id')},
            acceptance=item['acceptance'],
            source_checkpoint_sha256=item['source_checkpoint_sha256'],
            native_artifact_ids=json.dumps(item['native_artifact_ids'], sort_keys=True),
            e2e_ms=item['e2e_seconds']*1000, ttft_ms=item['ttft_seconds']*1000,
            tpot_ms=item['tpot_seconds']*1000, tokens_per_second=item['output_tokens_per_second']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', type=Path)
    parser.add_argument('--dnn', type=Path)
    parser.add_argument('--candidates', type=Path,
                        help='independent collect_r16_candidates.py candidates.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.candidates and not args.llm:
        parser.error('--candidates requires --llm')
    args.out.mkdir(parents=True, exist_ok=True)
    for family in ('llm', 'dnn'):
        source = getattr(args, family)
        if source is not None:
            with source.open() as stream:
                cells = list(csv.DictReader(stream))
            if family == 'llm' and args.candidates:
                cells.extend(candidate_rows(json.loads(args.candidates.read_text())))
            rows = compare(cells, family)
            write(args.out / f'R16_{family}_ratios.csv', rows)
            write(args.out / f'R16_{family}_geomeans.csv', geomeans(rows, family))
            print(f'{family}: {len(rows)} matched comparisons; speedup >1 means faster TileMega')


if __name__ == '__main__':
    main()
