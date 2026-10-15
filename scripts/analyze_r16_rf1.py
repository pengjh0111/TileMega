#!/usr/bin/env python3
"""Validate the recorded R15 L1 semantic ablation and compare fresh requests."""
import argparse
import json
from pathlib import Path
import statistics

from r16_measurement_validation import completed_guard, file_sha, validate_llm


def comparison(on, off):
    median_on, median_off = statistics.median(on), statistics.median(off)
    gap = abs(median_on-median_off)
    change = (median_on/median_off-1)*100
    ranges = {'on': max(on)-min(on), 'off': max(off)-min(off)}
    distinguishable = abs(change) > 0.5 and gap > max(ranges.values())
    return dict(on_median=median_on, off_median=median_off,
        on_change_percent=change, absolute_gap=gap, process_ranges=ranges,
        distinguishable=distinguishable,
        on_regression=distinguishable and change > 0)


def directory(root, state, name):
    record = state[name]
    path = Path(record['out'])
    path = path if path.is_absolute() else root/path
    return path, completed_guard(path, record)


def smoke(root, state, candidate, batch):
    path, _ = directory(root, state, candidate['smoke_job'])
    c1 = json.loads((path/'measurements/c1.json').read_text())
    c2 = json.loads((path/'measurements/c2.json').read_text())
    if c1.get('pass') is not True or c1.get('positions') != 16*batch:
        raise ValueError('16-token shared-checkpoint numerical check did not pass')
    passed = {(row['mode'], int(row['decode_loop']))
              for row in c2.get('checks', []) if row.get('pass') is True}
    if c2.get('pass') is not True or c2.get('same_binary') is not True or passed != {
            ('L1', 0), ('L2', 0), ('L1', 1)}:
        raise ValueError('same-binary executor token checks did not all pass')


def process(root, state, name, candidate, batch, plan):
    path, guard = directory(root, state, name)
    measured, _, own = validate_llm(path/'measurements'/f'B{batch}'/'measurements.json', guard, root)
    if (measured['system'] != candidate['system'] or measured['batch'] != batch or
            measured['gpu_index'] != plan['gpu'] or measured['model_config_sha256'] != plan['config_sha256'] or
            measured['prompt_ids_sha256'] != plan['prompt_sha256']):
        raise ValueError('RF1 system, model, prompt, batch or GPU differs')
    config = measured['configuration']
    if config['mode'] != 'L1' or config['decode_loop'] != 1:
        raise ValueError('RF1 formal run is not L1 device-loop')
    for phase in ('prefill', 'decode'):
        if Path(config[phase+'_binary']).resolve() != Path(candidate[phase]).resolve():
            raise ValueError('RF1 measurement belongs to another binary')
    actual = json.loads((path/'measurements/actual-execution.json').read_text())
    if actual['driver_sha256'] != plan['driver_sha256'] or len(actual['calls']) != 8:
        raise ValueError('missing frozen actual-execution evidence')
    for count in (1, 1024):
        rows = [row for row in actual['calls'] if row['tokens'] == count]
        if len(rows) != 4 or any(row['executor'] != 'L1' or
                row['device_loop_used'] is not (count > 1) for row in rows):
            raise ValueError('RF1 runtime fell back from the requested L1 loop')
    checkpoint = json.loads(Path(config['checkpoint_manifest']).read_text())
    if checkpoint['seed'] != plan['seed']:
        raise ValueError('RF1 checkpoint seed differs')
    return dict(pid=measured['pid'], round=measured['round_index'],
        e2e_seconds=measured['e2e_seconds'], ttft_seconds=measured['ttft_seconds'],
        tpot_seconds=measured['tpot_seconds'], versions=measured['versions'],
        checkpoint_sha256=checkpoint['checkpoint_sha256'], acceptance=own['acceptance'],
        measurement=str(path/'measurements'/f'B{batch}'/'measurements.json'))


def analyze(plan):
    root = Path(plan['root'])
    if file_sha(plan['source_comparison']) != plan['source_comparison_sha256']:
        raise ValueError('RF1 source comparison changed after queue preparation')
    state = {}
    for path in (root/'runs/r16/scheduler').glob('gpu*/state.json'):
        rows = json.loads(path.read_text())
        if set(rows) & set(state):
            raise ValueError('duplicate scheduler job names')
        state.update(rows)
    cells = []
    for cell in plan['cells']:
        output = dict(batch=cell['batch'], arms={}, rejected=[], pending=[])
        for arm, candidate in cell['arms'].items():
            rows = []
            if state.get(candidate['smoke_job'], {}).get('status') not in ('done', 'failed', 'skipped', 'not_run', 'cancelled'):
                output['pending'].append(candidate['smoke_job'])
                continue
            try:
                smoke(root, state, candidate, cell['batch'])
                for name in candidate['measurement_jobs']:
                    if state.get(name, {}).get('status') not in ('done', 'failed', 'skipped', 'not_run', 'cancelled'):
                        output['pending'].append(name)
                        continue
                    rows.append(process(root, state, name, candidate, cell['batch'], plan))
            except (OSError, ValueError, KeyError, TypeError) as error:
                output['rejected'].append(dict(arm=arm, error=str(error)))
            if rows:
                output['arms'][arm] = rows
        rows = [row for arm in output['arms'].values() for row in arm]
        if output['pending'] or output['rejected'] or any(len(output['arms'].get(arm, [])) != 3 for arm in ('on', 'off')):
            output['status'] = 'incomplete'
        elif (len({row['pid'] for row in rows}) != 6 or
              any({row['round'] for row in output['arms'][arm]} != {0, 1, 2} for arm in ('on', 'off')) or
              len({row['checkpoint_sha256'] for row in rows}) != 1 or
              len({json.dumps(row['versions'], sort_keys=True) for row in rows}) != 1):
            output['status'] = 'inconsistent_process_or_checkpoint_or_version_identity'
        else:
            output['metrics'] = {key: comparison([row[key] for row in output['arms']['on']],
                                                 [row[key] for row in output['arms']['off']])
                for key in ('e2e_seconds', 'ttft_seconds', 'tpot_seconds')}
            output['status'] = 'regression_requires_investigation' if any(
                output['metrics'][key]['on_regression'] for key in ('e2e_seconds', 'tpot_seconds')) else 'no_distinguishable_regression_in_this_scope'
        cells.append(output)
    return dict(evidence='verified completed requests only', scope=plan['protocol'],
        source_comparison=plan['source_comparison'], cells=cells,
        original_paged_rf1_closed=False,
        limitation='This experiment uses nonpaged L1. Provisional ownership remains provisional; no automatic default change.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({str(cell['batch']): cell['status'] for cell in result['cells']}))


if __name__ == '__main__':
    main()
