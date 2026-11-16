#!/usr/bin/env python3
"""Merge the complete eight-cell, eight-arm R6 P-ABL matrix."""
import argparse
import json
from pathlib import Path

from sm90d1_common import read, sha
from sm90r6_ablate import ARMS
from sm90r6_ablation_gates import validate_subset
from sm90r6_merge_analysis import write


def compact(coordinates):
    geometry = {}
    for family, g in coordinates['gemms'].items():
        geometry[family] = [g['tile_n'], g.get('tile_k', g['slot_bytes'] // (2 * g['tile_n'])),
                            g['warps_n'], g['warps_k'], g['slots'], g.get('register_stages', 1),
                            int(g.get('multicast', False)), g['mma']]
    structure = {k: v for k, v in coordinates.items() if k != 'gemms'}
    return [json.dumps(v, sort_keys=True, separators=(',', ':')) for v in (geometry, structure)]


def merge(root,allow_partial=False,only_cells=None):
    cells = read(root / 'ab5/domains-v1/cells.json')
    expected_cells=8
    if only_cells is not None:
        names=set(only_cells)
        if not names or not names<={cell['id'] for cell in cells}:
            raise ValueError('unknown or empty per-cell audit domain')
        cells=[cell for cell in cells if cell['id'] in names];expected_cells=len(names)
    frozen = {r['input']['id']: r for r in
              (json.loads(line) for line in (root / 'acceptance/v1/predictions.jsonl').read_text().splitlines()
               if line.strip())}
    rows = []
    for cell in cells:
        name = cell['id']; directory = root / 'ablation/pabl-v1' / name
        excluded=()
        missing={'olmoe_1b_7b-B1':'AB1a','qwen3_30b_a3b-B16':'AB1b'}
        if allow_partial and name in missing:
            excluded=(missing[name],);directory=root/'ablation/pabl-partial-v1'/name
        expected=tuple(arm for arm in ARMS if arm not in excluded)
        manifest = read(directory / 'arms.json')
        validate_subset(manifest, cell['model'], cell['batch'],excluded)
        measured = read(directory / 'measurement/result.json')
        if measured['source_sha256'] != sha(directory / 'arms.json') or set(measured['arms']) != set(expected):
            raise ValueError('P-ABL manifest changed or matrix incomplete')
        if excluded and (measured.get('excluded_arm')!=excluded[0] or measured['protocol']!='P-ABL partial seven-arm'):
            raise ValueError('partial P-ABL must declare the approved missing arm and protocol deviation')
        for arm in expected:
            binary = Path(manifest[arm]['binary']); identity = read(str(binary) + '.identity.json')
            coordinate = identity['plan']['native_dense']; selected_rank = ''; solve_seconds = ''
            receipt_path = ''; receipt_sha = ''
            if arm == 'F':
                prediction = frozen[name]
                if prediction['coordinates'] != coordinate:
                    raise ValueError('F coordinates changed')
                pool, replay, objective = prediction['pool_ns'], prediction['replay_ns'], 'frozen F prediction'
                a0_source=read(root/'a0/finalists-v1'/name/'source.json')
                if sha(a0_source['receipt'])!=a0_source['receipt_sha256']:
                    raise ValueError('A0 search receipt changed')
                matches=[c for c in read(a0_source['receipt'])['evaluated'] if c['coordinates']==coordinate]
                if len(matches)!=1:raise ValueError('F structural class is not unique in original E')
                structural_class=matches[0]['structural_class']
            else:
                if arm in ('A0', 'AB1a', 'AB1b'):
                    if arm == 'A0':
                        build = root / 'a0/finalists-v1' / name
                        selected = root / 'a0' / ('psel-v2' if name == 'olmoe_1b_7b-B16' else 'psel-v1') / name
                    else:
                        prefix = 'measured' if arm == 'AB1a' else 'traffic'
                        build = root / 'ab1' / (prefix + '-finalists-v1') / name
                        selected = root / 'ab1' / (prefix + '-psel-v1') / name
                else:
                    source_arm = 'AB4b' if arm == 'AB4c' else arm
                    build = root / 'ablation/finalists-v1' / name / source_arm / 'build'
                    selected = root / 'ablation/psel-v1' / name / source_arm
                source = read(build / 'source.json'); receipt_path = source['receipt']; receipt_sha = sha(receipt_path)
                if receipt_sha != source['receipt_sha256']:
                    raise ValueError('retained search receipt changed')
                receipt = read(receipt_path); winner = read(selected / 'result.json')
                selected_rank = winner['winner_rank']; candidate = receipt['finalists'][selected_rank]
                if candidate['coordinates'] != coordinate:
                    raise ValueError('P-ABL arm is not the selected search coordinate')
                selected_binary = build / ('candidate-' + str(selected_rank)) / 'decode.so'
                if winner['winner_binary_sha256'] != sha(selected_binary):
                    raise ValueError('selected binary changed')
                if arm != 'AB4c' and sha(binary) != sha(selected_binary):
                    raise ValueError('P-ABL binary differs from P-SEL')
                pool, replay = candidate['pool_ns'], candidate['replay_ns']
                structural_class=candidate['structural_class']
                objective = receipt.get('objective', 'pipeline')
                solve_seconds = receipt.get('seconds', '')
                if arm == 'AB4c':
                    objective += '; AB4b native-program prediction, no kernel-launch cost model'
            compile_result = read(str(binary) + '.compile-result.json')
            if compile_result['status'] != 'compiled' or not compile_result['inputs_unchanged']:
                raise ValueError('unstable selected build')
            limits = Path(compile_result['bounded_receipt'])
            if sha(limits) != compile_result['bounded_receipt_sha256']:
                raise ValueError('build duration evidence changed')
            build_time = read(limits)
            geometry, structure = compact(coordinate)
            value = measured['arms'][arm]; comparison = measured['comparisons'][arm]
            rows.append(dict(model=cell['model'], batch=cell['batch'], arm=arm,
                tpot_ms=value['tpot_ms'], range_pct=value['range_pct'], processes=value['processes'],
                change_vs_a0_pct=comparison['change_pct'], threshold_pct=comparison['threshold_pct'],
                decision=comparison['decision'], pool_ns=pool, replay_ns=replay, objective=objective,
                selected_rank=selected_rank, solve_seconds=solve_seconds,
                winner_build_seconds=build_time['ended'] - build_time['started'],
                winner_build_queue_seconds=build_time['started'] - build_time['queued'],
                geometry_tn_tk_wn_wk_slots_rs_mc_mma=geometry, structure=structure,
                structural_class=structural_class,
                coordinates_sha256=identity['plan']['coordinates_sha256'], binary_sha256=sha(binary),
                gpu=measured['gpu'], concurrency='trial', evidence='verified',
                measurement_status='verified',measurement_protocol=measured['protocol'],
                search_receipt=receipt_path, search_receipt_sha256=receipt_sha,
                raw=str(directory / 'measurement/result.json'), raw_sha256=sha(directory / 'measurement/result.json')))
        for arm in excluded:
            gap={key:'' for key in rows[-1]}
            gap.update(model=cell['model'],batch=cell['batch'],arm=arm,processes=0,
                decision='incomplete',measurement_status='unavailable_AB1_arm',
                measurement_protocol=measured['protocol'],evidence='stated exception',
                raw=str(root/'ablation/pabl-v1'/name/'incomplete.json'),
                raw_sha256=sha(root/'ablation/pabl-v1'/name/'incomplete.json'))
            rows.append(gap)
    if len(cells) != expected_cells or len(rows) != expected_cells*8:
        raise ValueError('complete eight-cell/eight-arm matrix required')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-partial',action='store_true',help='include the two declared seven-arm cells with explicit missing rows')
    p.add_argument('--cell',action='append',help='audit selected completed cells without claiming the full matrix')
    a = p.parse_args(); write(a.out, merge(a.root,a.allow_partial,a.cell))
