#!/usr/bin/env python3
"""Audit the independent-process B4 queue without imposing a single-card unit."""
import argparse
import collections
import json
from pathlib import Path

from sm90d1_common import ROOT, read, sha, write
from sm90r6_ab4_supplement import ARMS, validate
from sm90r6_dynamic_ab4 import PROTOCOL, summarize
from sm90r6_audit_final_measurements import audit_directory
from sm90r6_merge_ab4_supplement import MODELS, audit_correctness, comparison_table
from sm90r6_merge_ablation import compact
from sm90r6_merge_analysis import write as csv_write


def audit_timing(root, item, cell, expected_binary):
    row = item['result']
    name = item['job']['id']
    directory = root / 'process-audits' / name
    write(directory / 'inputs.json', read(root / 'inputs.json'))
    write(directory / 'unit/result.json', dict(rows=[row], arms=summarize([row])))
    result = audit_directory(directory, [row['arm']], cell['model'], 4)
    attempt = Path(row['raw'])
    record = read(attempt / 'job/sm90r3-process.json')
    if (record['decode_sha256'] != sha(expected_binary) or
            record['arm'] != item['job']['arm'] or record['batch'] != 4 or
            record['model'] != cell['model']):
        raise ValueError('timing artifact or process identity differs from queued item')
    pgid = (attempt / 'pgid').read_text().strip()
    if not pgid.isdigit():
        raise ValueError('invalid process group identity')
    result.update(item=name, pgid=pgid)
    write(directory / 'audit.json', result)
    return result


def merge(root):
    cells = read(root / 'cells.json')
    if len(cells) != 4 or {c['model'] for c in cells} != MODELS or any(c['batch'] != 4 for c in cells):
        raise ValueError('expected all four B4 models')
    source = read(root / 'source.json')
    previous = Path(source['previous'])
    if source['protocol'] != PROTOCOL or sha(source['handoff']) != source['handoff_sha256']:
        raise ValueError('dynamic scheduling authorization or handoff changed')
    if any(sha(p) != digest for p, digest in source['source_hashes'].items()):
        raise ValueError('retained numerical toolchain changed')
    if any(sha(p) != digest for p, digest in read(root / 'inputs.json').items()):
        raise ValueError('collection inputs changed')
    items = read(root / 'queue/items.json')
    if any(v['status'] != 'done' for v in items.values()):
        raise ValueError('queue has unfinished or failed items')
    counts = collections.Counter(v['job']['kind'] for v in items.values())
    if counts != dict(occupancy=4, llm=64, c2=40):
        raise ValueError('unexpected accepted item counts: ' + str(counts))
    fresh_groups = set()
    for item in items.values():
        if item['attempts']:
            accepted = item['attempts'][-1]
            if accepted.get('code') != 0 or item['result']['gpu'] != accepted['gpu']:
                raise ValueError('accepted attempt identity mismatch')
            if read(Path(accepted['directory']) / 'guard_result.json')['code']:
                raise ValueError('guard rejected accepted item')
        if item['job']['kind'] != 'occupancy':
            result = item['result']
            attempt = Path(result['raw'] if item['job']['kind'] == 'llm' else result['directory'])
            group = (attempt / 'pgid').read_text().strip()
            if not group.isdigit() or group in fresh_groups:
                raise ValueError('accepted process group reused')
            fresh_groups.add(group)
    if len(fresh_groups) != 104:
        raise ValueError('expected 104 independent accepted process groups')
    rows, processes, correctness, audits = [], [], [], []
    pgids = set()
    for cell in cells:
        directory = root / cell['id']
        original = previous / cell['id']
        if read(directory / 'complete.json')['status'] != 'complete':
            raise ValueError('incomplete cell')
        arms = read(directory / 'arms.json')
        validate(arms, cell['model'], 4)
        receipt_path = original / 'receipt.json'
        receipt = read(receipt_path)
        if receipt['input'] != cell:
            raise ValueError('search input changed')
        for field in ('seed', 'seed_source', 'export'):
            if sha(cell[field]) != cell[field + '_sha256']:
                raise ValueError('search input artifact changed')
        if read(original / 'build/source.json')['receipt_sha256'] != sha(receipt_path):
            raise ValueError('compiled receipt changed')
        candidates = read(original / 'build/candidates.json')
        if len(candidates) != 4 or any(v['status'] != 'compiled' for v in candidates):
            raise ValueError('four compiled finalists required')
        occupancy = items[cell['id'] + '-occupancy']['result']
        if (len(occupancy['candidates']) != 4 or
                any(v['effective_residency'] != v['requested_residency'] for v in occupancy['candidates']) or
                occupancy != read(directory / 'occupancy.json')):
            raise ValueError('occupancy evidence changed or requires rescore')
        for value in occupancy['candidates']:
            binary = candidates[value['rank']]['binary']
            if (value['binary_sha256'] != sha(binary) or
                    value['identity_sha256'] != sha(binary + '.identity.json')):
                raise ValueError('occupancy artifact changed')
        if occupancy['origin'] == 'v1':
            raw = read(occupancy['raw'])
            if raw['candidates'] != occupancy['candidates']:
                raise ValueError('imported occupancy result changed')
        winner = read(directory / 'psel/result.json')
        selected = receipt['finalists'][winner['winner_rank']]
        if winner['winner_binary_sha256'] != sha(arms['AB4b']['binary']):
            raise ValueError('selected artifact changed')
        psel_audits = []
        psel_rows = [items[name]['result'] for name in winner['items']]
        labels = [f'c{v["rank"]}' for v in candidates]
        if (len(winner['items']) != 8 or [v['arm'] for v in psel_rows] != labels + labels[::-1] or
                winner['rows'] != psel_rows or winner['arms'] != summarize(psel_rows)):
            raise ValueError('P-SEL process domain or summary changed')
        for name in winner['items']:
            item = items[name]
            psel_audits.append(audit_timing(root, item, cell,
                candidates[int(item['job']['arm'][1:])]['binary']))
        if len({v['checkpoint'] for v in psel_audits}) != 1:
            raise ValueError('P-SEL checkpoint differs across processes')
        ranked = sorted(winner['arms'], key=lambda k: winner['arms'][k]['tpot_ms'])
        best, second = ranked[:2]
        values = winner['arms']
        threshold = max(.5, 2 * max(values[best]['range_pct'], values[second]['range_pct']))
        gap = (values[second]['tpot_ms'] / values[best]['tpot_ms'] - 1) * 100
        predictions = {f'c{v["rank"]}': v['predicted_ns'] for v in candidates}
        expected = min((best, second), key=lambda k: predictions[k]) if gap <= threshold else best
        if (winner['winner_rank'] != int(expected[1:]) or winner['threshold_pct'] != threshold or
                winner['top_gap_pct'] != gap or winner['predicted_ns'] != predictions):
            raise ValueError('P-SEL selection does not reproduce')
        measured_path = directory / 'measurement/result.json'
        measured = read(measured_path)
        order = ['AB4b', 'AB4c', 'AB4c', 'AB4b', 'AB4b', 'AB4c']
        raw_rows = [items[name]['result'] for name in measured['items']]
        if (measured['protocol'] != PROTOCOL or measured['logical_order'] != order or
                len(raw_rows) != 6 or [v['arm'] for v in raw_rows] != order or
                measured['rows'] != raw_rows or measured['arms'] != summarize(raw_rows)):
            raise ValueError('dynamic timing protocol or summary changed')
        timing_audits = []
        for index, name in enumerate(measured['items']):
            item = items[name]
            value = item['result']
            audit = audit_timing(root, item, cell, arms[value['arm']]['binary'])
            timing_audits.append(audit)
            attempt = Path(value['raw'])
            measurement = next((attempt / 'job/measurements').rglob('measurements.json'))
            data = read(measurement)
            processes.append(dict(model=cell['model'], batch=4, logical_order=index,
                **{k: value[k] for k in ('arm', 'tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s', 'gpu')},
                item=name, pid=data['pid'], pgid=audit['pgid'],
                started=item['attempts'][-1]['started'], ended=item['attempts'][-1]['ended'],
                tokens=value['tokens'], tokens_sha256=sha(value['tokens']), checkpoint=audit['checkpoint'],
                raw=str(attempt), measurement=str(measurement), measurement_sha256=sha(measurement),
                guard_sha256=sha(attempt / 'guard_result.json'),
                process_record_sha256=sha(attempt / 'job/sm90r3-process.json'), protocol=PROTOCOL))
        for audit in psel_audits + timing_audits:
            if audit['pgid'] in pgids:
                raise ValueError('fresh process group reused across timing items')
            pgids.add(audit['pgid'])
        checkpoints = {v['checkpoint'] for v in timing_audits}
        tokens = [read(v['tokens']) for v in raw_rows]
        if len(checkpoints) != 1 or any(v != tokens[0] for v in tokens[1:]):
            raise ValueError('cross-arm checkpoint or token mismatch')
        if measured['checkpoint'] != next(iter(checkpoints)):
            raise ValueError('checkpoint summary changed')
        record = dict(cell=cell['id'], psel=dict(processes=8, records=psel_audits),
                      timing=dict(processes=6, records=timing_audits))
        audits.append(record)
        for label in ARMS:
            arm = arms[label]
            binary = Path(arm['binary'])
            identity = read(str(binary) + '.identity.json')
            coordinates = identity['plan']['native_dense']
            if coordinates != selected['coordinates']:
                raise ValueError('selected arm coordinates differ')
            outcome = read(str(binary) + '.compile-result.json')
            if outcome['status'] != 'compiled' or not outcome['inputs_unchanged']:
                raise ValueError('unstable selected build')
            limits = Path(outcome['bounded_receipt'])
            if sha(limits) != outcome['bounded_receipt_sha256']:
                raise ValueError('build receipt changed')
            build_time = read(limits)
            geometry, structure = compact(coordinates)
            value = measured['arms'][label]
            change = measured['comparisons'][label]
            expected_change = (value['tpot_ms'] / measured['arms']['AB4b']['tpot_ms'] - 1) * 100
            threshold = max(.5, 2 * max(value['range_pct'], measured['arms']['AB4b']['range_pct']))
            decision = ('indistinguishable' if abs(expected_change) <= threshold else
                        'faster' if expected_change < 0 else 'slower')
            if (value['processes'] != 3 or change != dict(change_pct=expected_change,
                    threshold_pct=threshold, decision=decision)):
                raise ValueError('timing comparison does not reproduce')
            corr = audit_correctness(arm, cell)
            correctness.append(dict(model=cell['model'], batch=4, arm=label,
                **{k: v for k, v in corr.items() if k != 'hashes'}, protocol=PROTOCOL))
            record[label] = corr
            per_gpu = {}
            for gpu in sorted({v['gpu'] for v in raw_rows if v['arm'] == label}):
                per_gpu[str(gpu)] = summarize([v for v in raw_rows if v['arm'] == label and v['gpu'] == gpu])[label]
            rows.append(dict(model=cell['model'], batch=4, arm=label, **value,
                change_vs_ab4b_pct=expected_change, threshold_pct=threshold, decision=decision,
                ab4b_over_arm_speedup=measured['arms']['AB4b']['tpot_ms'] / value['tpot_ms'],
                pool_ns=selected['pool_ns'], replay_ns=selected['replay_ns'],
                objective=receipt['objective'] + ('; AB4b program prediction, no kernel-launch cost model' if label == 'AB4c' else ''),
                selected_rank=winner['winner_rank'], solve_seconds=receipt['seconds'],
                search_evaluations=receipt['evaluations'], psel_gap_pct=gap,
                psel_threshold_pct=winner['threshold_pct'],
                psel_tie_broken_by_prediction=winner['tie_broken_by_prediction'],
                winner_build_seconds=build_time['ended'] - build_time['started'],
                winner_build_queue_seconds=build_time['started'] - build_time['queued'],
                geometry_tn_tk_wn_wk_slots_rs_mc_mma=geometry, structure=structure,
                structural_class=selected['structural_class'], coordinates_sha256=identity['plan']['coordinates_sha256'],
                binary=str(binary), binary_sha256=sha(binary),
                gpus=sorted(map(int, per_gpu)), per_gpu=per_gpu, checkpoint=measured['checkpoint'],
                concurrency='trial', evidence='verified', measurement_protocol=PROTOCOL,
                search_receipt=str(receipt_path), search_receipt_sha256=sha(receipt_path),
                seed_source=cell['seed_source'], seed_source_sha256=cell['seed_source_sha256'],
                c1=arm['c1'], c1_sha256=sha(arm['c1']), c2=arm['c2'], c2_sha256=sha(arm['c2']),
                raw=str(measured_path), raw_sha256=sha(measured_path)))
    if len(rows) != 8 or len(processes) != 24 or len(correctness) != 8 or len(pgids) != 56:
        raise ValueError('incomplete final acceptance counts')
    return rows, processes, correctness, audits


def write_tables(root, out):
    rows, processes, correctness, audits = merge(root)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        'SM90R6_ab4_b4.csv': rows,
        'SM90R6_ab4_b4_processes.csv': processes,
        'SM90R6_ab4_b4_correctness.csv': correctness,
        'SM90R6_ab4_comparison.csv': comparison_table(rows, ROOT / 'docs/results/SM90R6_ablation.csv'),
    }
    items, attempts = [], []
    for name, item in read(root / 'queue/items.json').items():
        job, result = item['job'], item['result']
        items.append(dict(item=name, model=job['model'], batch=4, kind=job['kind'],
            arm=job.get('arm', ''), status=item['status'], gpu=result.get('gpu', ''),
            origin=result['origin'], attempts=len(item['attempts']),
            result=result, definition=job, protocol=PROTOCOL))
        for index, value in enumerate(item['attempts']):
            path = Path(value['directory']) / 'guard_result.json'
            attempts.append(dict(item=name, attempt=index, **value,
                guard=str(path), guard_sha256=sha(path), protocol=PROTOCOL))
    tables['SM90R6_ab4_b4_items.csv'] = items
    tables['SM90R6_ab4_b4_attempts.csv'] = attempts
    for name, values in tables.items():
        csv_write(out / name, [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                               for k, v in row.items()} for row in values])
    audit_path = root / 'summary/audit.json'
    write(audit_path, dict(verified=True, cells=4, arms=8, timing_processes=24,
        psel_processes=32, c2_processes=40, c1_processes=8,
        occupancy_items=4, accepted_processes=104, protocol=PROTOCOL,
        script_sha256=sha(__file__), records=audits))
    return rows, [out / name for name in tables], audit_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, files, audit = write_tables(args.root.resolve(), args.out.resolve())
    print(dict(arms=len(rows), files=len(files), audit=str(audit)))
