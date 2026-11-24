"""Measure the frozen SQ2 population with independent guarded processes."""

import collections
import concurrent.futures
import csv
from pathlib import Path
import statistics
import time

from calibration_queue import CASE, REPRO, prior_attempt
from guarded import ROOT, PYTHON, gpu_attempt, read, sha, write
from phase0 import PROMPT, timing_value


MEAS = CASE / 'measurements'


def verify_freeze():
    frozen = read(CASE / 'frozen_search.json')
    if not frozen['model_frozen_at'] < frozen['frozen_at'] < time.time():
        raise RuntimeError('TPOT must follow the model and population freezes')
    for info in frozen['files'].values():
        assert sha(info['path']) == info['sha256'], info['path']
    for name, expected in frozen['source_hashes'].items():
        assert sha(ROOT / name) == expected, name
    for path in MEAS.glob('**/command.json'):
        assert read(path)['started'] > frozen['frozen_at'], path
    return frozen


def attempt_process(phase, round_index, q, card, actual, frozen, check):
    folder = MEAS / phase / q
    receipt_path = folder / 'result.json'
    row = actual[q]
    if receipt_path.is_file():
        result = read(receipt_path)
        assert result['binary_sha256'] == row['binary_sha256']
        assert result['gpu_uuid'] == card['uuid'] and result['round_index'] == round_index
        return True
    binary = Path(row['binary'])
    assert sha(binary) == row['binary_sha256']
    command = [PYTHON, '-m', 'tilemega.serving.r16_run', '--model', frozen['checkpoint'],
               '--prefill', 'prompt', '--decode', binary, '--prompt-ids', PROMPT,
               '--batch', 1, '--max-tokens', 1024, '--warmup', 1, '--repeats', 3,
               '--mode', 'L1', '--decode-loop', 1, '--prefill-mode', 'L1',
               '--round-index', round_index, '--out', '@ATTEMPT@/job/measurements']
    if check:
        command.extend(['--check-step-loop', '--hf-check', '--hf-positions', 256])
    previous = prior_attempt(folder)
    if previous:
        path, guard = previous
    else:
        next_attempt = max((int(d.name[7:]) for d in folder.glob('attempt*')), default=-1) + 1
        path = folder / f'attempt{next_attempt:04d}'
        assert time.time() > frozen['frozen_at']
        guard = gpu_attempt(command, path, REPRO / 'guard_policy.json', card)
    assert read(path / 'command.json')['started'] > frozen['frozen_at']
    if guard['code'] == 75:
        return False
    result = {
        'qid': q, 'phase': phase, 'round_index': round_index, 'gpu_uuid': card['uuid'],
        'binary_sha256': row['binary_sha256'], 'realization_id': row['realization_id'],
        'attempt': str(path), 'returncode': guard['code'], 'status': 'runtime_failed',
        'checks_requested': check,
    }
    if guard['code'] == 0:
        spec = {'checkpoint_sha256': frozen['checkpoint_sha256'],
                'prompt_sha256': frozen['files']['prompt']['sha256'], 'reproduction_card': card}
        result.update(timing_value(path, binary, spec, check), status='valid')
    elif check:
        for name in ('c1', 'c2'):
            check_path = path / f'job/measurements/{name}.json'
            if check_path.exists() and not read(check_path)['pass']:
                result.update(status='numerical_failed', failed_check=name)
    write(receipt_path, result)
    print(phase, q, card['uuid'], result['status'], result.get('tpot_ms'), flush=True)
    return True


def run_round(phase, index, order, lanes, actual, frozen, checks=False):
    def worker(card):
        pending = collections.deque(q for q in order if lanes[q] == card['uuid'])
        while pending:
            q = pending.popleft()
            if not attempt_process(phase, index, q, card, actual, frozen, checks):
                pending.append(q)
                time.sleep(5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(frozen['gpu_cards'])) as pool:
        futures = [pool.submit(worker, card) for card in frozen['gpu_cards']]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def csv_dump(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def candidate_results(frozen, actual):
    pop = read(frozen['files']['population']['path'])
    cells = {row['id']: row for row in pop['all_original_coordinates']}
    representatives = frozen['representatives']
    result = []
    for q in pop['P']:
        v = actual[q]
        scores = frozen['scores'][q]
        rid = v.get('realization_id')
        rep = representatives.get(rid)
        values = []
        states = []
        for phase in ('round0', 'round1'):
            p = MEAS / phase / (rep or 'absent') / 'result.json'
            if p.exists():
                r = read(p)
                states.append(r['status'])
                if r['status'] == 'valid':
                    values.append(r['tpot_ms'])
        good = len(values) == 2
        compile_receipt = Path(v['build_dir']) / 'compile/exit.json' if v.get('build_dir') else None
        result.append({
            'id': q, 'coordinate_sha256': cells[q]['coordinate_sha256'], 'realization_id': rid,
            'representative': rep, 'E': 'E' in cells[q]['sets'], 'U': 'U' in cells[q]['sets'],
            'Q': 'Q' in cells[q]['sets'], 'S': 'S' in cells[q]['sets'],
            'baselines': '|'.join(cells[q]['baselines']),
            'pool_ns': v.get('realized_pool_ns', scores['pool_ns']),
            'operator_sum_ns': scores['operator_sum_ns'],
            'replay_ns': v.get('realized_replay_ns', cells[q].get('replay_ns')),
            'compile_status': v['status'], 'validation_status': '|'.join(states),
            'compile_command_seconds': read(compile_receipt)['seconds'] if compile_receipt else None,
            'compile_receipt': str(compile_receipt) if compile_receipt else None,
            'binary_sha256': v.get('binary_sha256'), 'tpot_ms': statistics.median(values) if good else None,
            'process0_tpot_ms': values[0] if values else None,
            'process1_tpot_ms': values[1] if good else None,
            'range_ms': max(values) - min(values) if good else None,
        })
    csv_dump(CASE / 'candidates.csv', result)
    return result


def write_attempt_index(frozen):
    rows = []
    for folder in MEAS.glob('*/*/attempt*'):
        command_file = folder / 'command.json'
        start = read(command_file)['started'] if command_file.exists() else None
        assert start is None or start > frozen['frozen_at']
        guard_file = folder / 'guard/guard_result.json'
        guard = read(guard_file) if guard_file.exists() else {}
        device = read(folder / 'gpu_before.json')
        record_file = folder.parent / 'result.json'
        record = read(record_file) if record_file.exists() else {}
        accepted = record.get('attempt') == str(folder)
        end = folder / 'exit.json'
        rows.append({
            'phase': folder.parents[1].name, 'id': folder.parent.name, 'attempt': str(folder),
            'started': start, 'wall_seconds': read(end)['seconds'] if end.exists() else None,
            'gpu_uuid': device['uuid'], 'guard_code': guard.get('code'), 'guard_reason': guard.get('reason'),
            'status': record['status'] if accepted else 'guard_rejected' if guard.get('code') == 75 else 'unfinished',
            'measurement_sha256': record.get('measurement_sha256') if accepted else None,
            'tpot_ms': record.get('tpot_ms') if accepted else None,
        })
    csv_dump(CASE / 'timings.csv', sorted(rows, key=lambda r: (r['started'] is None, r['started'] or 0)))


def main():
    frozen = verify_freeze()
    actual = read(frozen['files']['materializations']['path'])
    first, second = frozen['round_order']
    started = time.time()
    run_round('round0', 0, first, frozen['round_gpu_uuid'][0], actual, frozen, True)
    eligible = [q for q in second if read(MEAS / 'round0' / q / 'result.json')['status'] == 'valid']
    run_round('round1', 1, eligible, frozen['round_gpu_uuid'][1], actual, frozen)
    result = candidate_results(frozen, actual)
    index = {r['id']: r for r in result}
    valid = [r for r in result if r['id'] == r['representative'] and r['tpot_ms'] is not None]
    best = sorted(valid, key=lambda r: (r['tpot_ms'], r['id']))[:3]
    valid_finalists = [index[q] for q in frozen['finalists'] if index[q]['tpot_ms'] is not None]
    measured = min(valid_finalists, key=lambda r: r['tpot_ms']) if valid_finalists else None
    selected = [frozen['predicted_winner'], *(r['id'] for r in best), *frozen['baseline_ids'].values()]
    if measured:
        selected.append(measured['id'])
    confirm = list(dict.fromkeys(index[q]['representative'] for q in selected if index[q]['tpot_ms'] is not None))
    write(CASE / 'confirmation_plan.json', {
        'predicted': frozen['predicted_winner'], 'measured': measured['id'] if measured else None,
        'observed_top3': [r['id'] for r in best], 'representatives': confirm,
        'started': started, 'audit_finished': time.time(),
        'selection_rule': 'measured sees only S; confirmation runs do not change selection or the audit oracle',
    })
    confirmations = []
    for repeat in range(5):
        phase = f'confirmation{repeat}'
        lanes = {q: frozen['gpu_cards'][repeat % len(frozen['gpu_cards'])]['uuid'] for q in confirm}
        run_round(phase, repeat + 2, confirm, lanes, actual, frozen)
        confirmations.extend(read(MEAS / phase / q / 'result.json') for q in confirm)
    csv_dump(CASE / 'confirmation.csv', confirmations)
    write_attempt_index(frozen)
    write(CASE / 'measurement_complete.json', {'finished': time.time(), 'valid_representatives': len(valid),
                                               'audit_rows': len(result), 'confirmation_processes': len(confirmations)})


if __name__ == '__main__':
    main()
