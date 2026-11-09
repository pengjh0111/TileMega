#!/usr/bin/env python3
"""Audit and summarize the four-model B4 AB4b/AB4c supplement."""
import argparse
import json
from pathlib import Path

from sm90d1_common import ROOT, read, sha, write as write_json
from sm90r6_ab4_supplement import ARMS, validate
from sm90r6_audit_final_measurements import audit_directory
from sm90r6_merge_ablation import compact
from sm90r6_merge_analysis import write as write_csv

MODELS = {'qwen3_1_7b', 'llama3_1_8b', 'qwen3_30b_a3b', 'olmoe_1b_7b'}


def audit_correctness(arm, cell):
    c2 = read(arm['c2'])
    digest = sha(arm['binary'])
    prompt = ('olmo' if cell['model'].startswith('olmo') else
              'llama3' if cell['model'].startswith('llama') else 'qwen')
    prompt_digest = sha(ROOT / f'configs/prompts/{prompt}_isl64.json')
    checkpoints, tokens, pgids = set(), set(), set()
    hashes = {arm[key]: sha(arm[key]) for key in ('c1', 'c2')}
    for item in c2['processes']:
        directory = Path(item['directory'])
        pgid = (directory / 'guard/pgid').read_text().strip()
        if not pgid.isdigit() or pgid in pgids:
            raise ValueError('C2 reused a process group')
        pgids.add(pgid)
        if read(directory / 'result.json')['returncode'] or read(directory / 'guard/guard_result.json')['code']:
            raise ValueError('unaccepted C2 process')
        checks = read(directory / 'checks/result.json')
        if (not checks['passed'] or checks['batch'] != cell['batch'] or checks['steps'] != 1024 or
                int(checks['gpu']) != item['gpu'] or checks['prompt_sha256'] != prompt_digest or
                checks['checkpoint'] != item['checkpoint'] or
                checks['checkpoint'] != read(directory / 'checkpoint/random_checkpoint.json')['checkpoint_sha256']):
            raise ValueError('C2 input or checkpoint mismatch')
        checkpoints.add(item['checkpoint'])
        tokens.add(item['tokens_sha256'])
        expected = {('candidate', False), ('candidate', True)}
        if c2['artifacts']['candidate']['path'] != c2['artifacts']['reference']['path']:
            expected |= {('reference', False), ('reference', True)}
        if {(v['implementation'], v['decode_loop_used']) for v in checks['checks']} != expected:
            raise ValueError('incomplete C2 loop/reference coverage')
        values = []
        for check in checks['checks']:
            artifact = c2['artifacts'][check['implementation']]
            if not check['passed'] or check['binary_sha256'] != artifact['sha256'] or sha(artifact['path']) != artifact['sha256']:
                raise ValueError('C2 artifact changed')
            path = directory / 'checks' / f'{check["implementation"]}-loop{int(check["decode_loop_used"])}.tokens.json'
            if sha(path) != check['tokens_sha256'] or sha(path) != item['tokens_sha256']:
                raise ValueError('C2 tokens changed')
            value = read(path)
            if len(value) != cell['batch'] or any(len(v) != 1024 for v in value):
                raise ValueError('incomplete C2 token arrays')
            values.append(value)
            hashes[str(path)] = sha(path)
        if any(value != values[0] for value in values[1:]):
            raise ValueError('C2 token mismatch')
        for suffix in ('result.json', 'checks/result.json', 'guard/guard_result.json', 'guard/pgid', 'command.json'):
            path = directory / suffix
            hashes[str(path)] = sha(path)
    if len(checkpoints) != 1 or len(tokens) != 1 or len(pgids) != 5:
        raise ValueError('C2 fresh-process consistency failed')
    c1 = read(arm['c1'])
    if len(c1['rows']) != 1:
        raise ValueError('expected one 256-position C1 reference process')
    row = c1['rows'][0]
    job = Path(row['tokens']).parent.parent
    record = read(job / 'sm90r3-process.json')
    if (read(job.parent / 'guard_result.json')['code'] or record['exit_code'] or
            record['decode_sha256'] != digest or row['c1']['max_new_tokens'] != 256 or
            row['c1']['positions'] != 256 * cell['batch'] or
            row['c1']['batch'] != cell['batch'] or not row['c1']['pass']):
        raise ValueError('C1 raw evidence failed')
    for path in (job / 'sm90r3-process.json', job.parent / 'guard_result.json', job.parent / 'pgid'):
        hashes[str(path)] = sha(path)
    return dict(c2_processes=5, c2_passed=5, c2_checkpoint=next(iter(checkpoints)),
                c2_tokens_sha256=next(iter(tokens)), c1_processes=1, c1_positions=256,
                c1_total_positions=row['c1']['positions'],
                c1_pass=True, c1_gap_ratio=row['c1']['gap_le_0_5_ratio'], c1_max_gap=row['c1']['max_gap'],
                c1=str(arm['c1']), c2=str(arm['c2']), hashes=hashes)


def merge(root):
    cells = read(root / 'cells.json')
    if ({c['model'] for c in cells} != MODELS or len(cells) != 4 or
            any(c['batch'] != 4 for c in cells)):
        raise ValueError('expected exactly the four B4 model cells')
    source = read(root / 'source.json')
    if any(sha(p) != digest for p, digest in source['source_hashes'].items()):
        raise ValueError('retained toolchain changed')
    rows, process_rows, correctness_rows, audits = [], [], [], []
    sequence = ['AB4b', 'AB4c', 'AB4c', 'AB4b', 'AB4b', 'AB4c']
    for cell in cells:
        directory = root / cell['id']
        arms = read(directory / 'arms.json')
        validate(arms, cell['model'], 4)
        if read(directory / 'complete.json')['status'] != 'complete':
            raise ValueError('incomplete cell')
        receipt_path = directory / 'receipt.json'
        receipt = read(receipt_path)
        if receipt['input'] != cell:
            raise ValueError('search input changed')
        for field in ('seed', 'seed_source', 'export'):
            if sha(cell[field]) != cell[field + '_sha256']:
                raise ValueError('search input artifact changed')
        build = directory / 'build'
        build_source = read(build / 'source.json')
        if build_source['receipt_sha256'] != sha(receipt_path):
            raise ValueError('compiled receipt changed')
        winner = read(directory / 'psel/result.json')
        selected = receipt['finalists'][winner['winner_rank']]
        if winner['winner_binary_sha256'] != sha(arms['AB4b']['binary']):
            raise ValueError('selected binary changed')
        labels = [f'c{v["rank"]}' for v in read(build / 'candidates.json') if v['status'] == 'compiled']
        if len(labels) != 4:
            raise ValueError('P-SEL requires all four candidates')
        psel_audit = audit_directory(directory / 'psel', labels + labels[::-1], cell['model'], 4)
        timing_audit = audit_directory(directory / 'measurement', sequence, cell['model'], 4)
        measured_path = directory / 'measurement/result.json'
        measured = read(measured_path)
        if measured['protocol'] != 'P-ABL two-arm supplement' or measured['order'] != sequence:
            raise ValueError('supplement timing protocol changed')
        raw = read(directory / 'measurement/unit/result.json')
        if any(read(v['tokens']) != read(raw['rows'][0]['tokens']) for v in raw['rows'][1:]):
            raise ValueError('timing cross-arm token mismatch')
        audits.append(dict(cell=cell['id'], psel=psel_audit, timing=timing_audit))
        for index, value in enumerate(raw['rows']):
            process_rows.append(dict(model=cell['model'], batch=4, process_order=index,
                arm=value['arm'], tpot_ms=value['tpot_ms'], ttft_ms=value['ttft_ms'],
                e2e_ms=value['e2e_ms'], tokens_per_s=value['tokens_per_s'], gpu=value['gpu'],
                tokens=value['tokens'], tokens_sha256=sha(value['tokens']),
                checkpoint=timing_audit['checkpoint'], protocol=measured['protocol']))
        for label in ARMS:
            arm = arms[label]
            binary = Path(arm['binary'])
            identity = read(str(binary) + '.identity.json')
            coordinate = identity['plan']['native_dense']
            if coordinate != selected['coordinates']:
                raise ValueError('arm coordinates differ from P-SEL winner')
            outcome = read(str(binary) + '.compile-result.json')
            if outcome['status'] != 'compiled' or not outcome['inputs_unchanged']:
                raise ValueError('unstable selected build')
            limits = Path(outcome['bounded_receipt'])
            if sha(limits) != outcome['bounded_receipt_sha256']:
                raise ValueError('build limit receipt changed')
            build_time = read(limits)
            geometry, structure = compact(coordinate)
            value = measured['arms'][label]
            change = measured['comparisons'][label]
            expected_change = (value['tpot_ms'] / measured['arms']['AB4b']['tpot_ms'] - 1) * 100
            threshold = max(.5, 2 * max(value['range_pct'], measured['arms']['AB4b']['range_pct']))
            if change['change_pct'] != expected_change or change['threshold_pct'] != threshold or value['processes'] != 3:
                raise ValueError('comparison or fresh-process count changed')
            corr = audit_correctness(arm, cell)
            correctness_rows.append(dict(model=cell['model'], batch=4, arm=label,
                **{k: v for k, v in corr.items() if k != 'hashes'}))
            audits[-1][label] = corr
            rows.append(dict(model=cell['model'], batch=4, arm=label, **value,
                change_vs_ab4b_pct=change['change_pct'], threshold_pct=threshold, decision=change['decision'],
                ab4b_over_arm_speedup=measured['arms']['AB4b']['tpot_ms'] / value['tpot_ms'],
                pool_ns=selected['pool_ns'], replay_ns=selected['replay_ns'],
                objective=receipt['objective'] + ('; AB4b program prediction, no kernel-launch cost model' if label == 'AB4c' else ''),
                selected_rank=winner['winner_rank'], solve_seconds=receipt['seconds'],
                search_evaluations=receipt['evaluations'], psel_gap_pct=winner['top_gap_pct'],
                psel_threshold_pct=winner['threshold_pct'], psel_tie_broken_by_prediction=winner['tie_broken_by_prediction'],
                winner_build_seconds=build_time['ended'] - build_time['started'],
                winner_build_queue_seconds=build_time['started'] - build_time['queued'],
                geometry_tn_tk_wn_wk_slots_rs_mc_mma=geometry, structure=structure,
                structural_class=selected['structural_class'], coordinates_sha256=identity['plan']['coordinates_sha256'],
                binary=str(binary), binary_sha256=sha(binary), gpu=measured['gpu'], checkpoint=measured['checkpoint'],
                concurrency='trial', evidence='verified', measurement_protocol=measured['protocol'],
                search_receipt=str(receipt_path), search_receipt_sha256=sha(receipt_path),
                seed_source=cell['seed_source'], seed_source_sha256=cell['seed_source_sha256'],
                c1=arm['c1'], c1_sha256=sha(arm['c1']), c2=arm['c2'], c2_sha256=sha(arm['c2']),
                raw=str(measured_path), raw_sha256=sha(measured_path)))
    return rows, process_rows, correctness_rows, audits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='output directory')
    args = parser.parse_args()
    rows, process_rows, correctness_rows, audits = merge(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'SM90R6_ab4_b4.csv', rows)
    write_csv(args.out / 'SM90R6_ab4_b4_processes.csv', process_rows)
    write_csv(args.out / 'SM90R6_ab4_b4_correctness.csv', correctness_rows)
    write_json(args.out / 'audit.json', dict(verified=True, cells=4, arms=8,
        timing_processes=len(process_rows), psel_processes=sum(v['psel']['processes'] for v in audits),
        c2_processes=sum(v['c2_processes'] for v in correctness_rows),
        c1_processes=sum(v['c1_processes'] for v in correctness_rows),
        script_sha256=sha(__file__), records=audits))
    print(json.dumps(dict(cells=4, arms=len(rows), timing_processes=len(process_rows), correctness_checks=len(correctness_rows))))
