#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Report one selected policy against the four complete frozen LLM baselines."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

from collect_r16_final_baselines import SYSTEMS
from collect_r16_final_matrix import EXPECTED
from compare_r16_results import compare, complete, geomeans, write


METRICS = ('e2e_ms', 'ttft_ms', 'tpot_ms', 'tokens_per_second')


def unique_cells(rows, native=False):
    result = {}
    for row in rows:
        if not complete(row):
            raise ValueError('final policy input contains an incomplete cell')
        coordinate = row['model'], int(row['batch'])
        if coordinate not in EXPECTED:
            raise ValueError('input workload is outside the final matrix')
        if not native and row['system'] not in SYSTEMS:
            raise ValueError('unknown external baseline in final policy input')
        if native and row['system'] != 'TileMega':
            raise ValueError('final native input must contain only selected TileMega cells')
        key = coordinate if native else (*coordinate, row['system'])
        if key in result:
            raise ValueError('ambiguous final policy or baseline cell')
        result[key] = row
    return result


def policy_means(ratios):
    groups = collections.defaultdict(list)
    for row in ratios:
        groups[row['baseline_system']].append(row)
    result = []
    for system, rows in sorted(groups.items()):
        coordinates = {(row['model'], int(row['batch'])) for row in rows}
        if len(coordinates) != len(rows):
            raise ValueError('policy geometric mean contains duplicate coordinates')
        result.append(dict(baseline_system=system, matched_cells=len(rows),
            expected_cells=30, full_matrix=coordinates==EXPECTED,
            coordinates=';'.join(f'{m}/B{b}' for m,b in sorted(coordinates)),
            device_pairings=json.dumps(dict(collections.Counter(row['device_pairing'] for row in rows)),sort_keys=True),
            boot_pairings=json.dumps(dict(collections.Counter(row['boot_pairing'] for row in rows)),sort_keys=True),
            identical_checkpoint_cells=sum(row['checkpoint_shared'] for row in rows),
            acceptance='strict' if all(row['acceptance']=='strict' for row in rows)
                else 'provisional_container_pid_namespace',
            scope='descriptive selected-policy mean; mixed/unknown device pairing is explicit; not a nonregression gate',
            **{field+'_speedup':math.exp(statistics.mean(math.log(row[field+'_speedup']) for row in rows))
               for field in METRICS}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True,
                        help='R16_selected_llm.csv from collect_r16_final_matrix.py')
    parser.add_argument('--baselines', type=Path, required=True,
                        help='R16_final_baselines.csv from collect_r16_final_baselines.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    native = list(csv.DictReader(args.native.open()))
    baselines = list(csv.DictReader(args.baselines.open()))
    selected = unique_cells(native, native=True)
    external = unique_cells(baselines)
    for row in native:
        row['comparison_arm'] = 'R16-final-selected-policy-v1'
    ratios = compare(native+baselines, 'llm')
    means = policy_means(ratios)
    wide = []
    for (model, batch), row in sorted(selected.items()):
        item = dict(model=model, batch=batch, native_candidate=row['candidate_id'],
            native_gpu=row['gpu'], native_gpu_uuid=row['physical_gpu_uuid'],
            native_source_checkpoint=row['source_checkpoint_sha256'])
        item.update({'tilemega_'+field:row[field] for field in METRICS})
        for system in SYSTEMS:
            other = external.get((model, batch, system))
            for field in METRICS:
                item[system+'_'+field] = other[field] if other else ''
        wide.append(item)
    missing = [dict(model=m,batch=b) for m,b in sorted(EXPECTED-set(selected))]
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/'R16_final_llm_table.csv', wide)
    write(args.out/'R16_final_llm_ratios.csv', ratios)
    write(args.out/'R16_final_llm_geomeans.csv', means)
    write(args.out/'R16_final_llm_pairing_geomeans.csv', geomeans(ratios, 'llm'))
    full = not missing and len(external)==120 and len(ratios)==120
    (args.out/'performance-coverage.json').write_text(json.dumps(dict(
        complete=full, selected_native_cells=len(selected), external_cells=len(external),
        comparisons=len(ratios), missing_native=missing,
        default_promoted=False, accuracy='model-level numeric gates are checked by the native collector',
        limits='device pairing, shared checkpoint and PID ownership remain explicit in ratio CSV; MoE without routing trace has no MBU'),indent=2)+'\n')
    print(json.dumps(dict(selected_cells=len(selected), external_cells=len(external),
                         comparisons=len(ratios), complete=full)))
    return int(args.require_complete and not full)


if __name__ == '__main__':
    raise SystemExit(main())
