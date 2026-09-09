#!/usr/bin/env python3
"""Fit per-task/per-wait event costs from archived steady-state four arms."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def schedule(path):
    lines = path.read_text().splitlines()
    timing = next(x for x in lines if x.startswith('E2E_TIMING '))
    if 'cold=0 warmup=5 repeat=11' not in timing:
        raise ValueError(f'not the required steady-state protocol: {path}')
    fields = next(x for x in lines if x.startswith('E2E_SCHEDULE '))
    return dict(x.split('=', 1) for x in fields.split()[1:])


def fit(cells, response, count):
    return sum(c[count]*c[response] for c in cells)/sum(c[count]**2 for c in cells)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'calibration')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for arch, dirname in [('sm_89', 'raw'), ('sm_120', 'raw_sm120')]:
        raw = a.repo/'docs/experiments/OCCUPANCY'/dirname
        rows = list(csv.DictReader((raw/'attrib.tsv').open(), delimiter='\t'))
        rows = [r for r in rows if r['variant'] == 'occ1']
        cells = []
        for model in ('gqa2', 'mha4'):
            for seq in (4, 128):
                sample = [r for r in rows if r['model'] == model and int(r['seq']) == seq]
                groups = {}
                for r in sample:
                    if r['arm'] == 'full' and r['pass'] != '1':
                        raise ValueError('failed synchronized full arm')
                    groups.setdefault(int(r['round']), {})[r['arm']] = r
                if sorted(groups) != list(range(25)):
                    raise ValueError('expected all 25 rounds')
                notify, wait, counts = [], [], set()
                for round_id, g in groups.items():
                    if set(g) != {'full', 'nowait', 'neither', 'l1nosync'}:
                        raise ValueError('missing four-arm pair')
                    notify.append(1e6*(float(g['nowait']['l2_ms'])-float(g['neither']['l2_ms'])))
                    wait.append(1e6*(float(g['full']['l2_ms'])-float(g['nowait']['l2_ms'])))
                    for arm in g:
                        s = schedule(raw/'log'/f'attrib_{model}_occ1_{arm}_{seq}_3_{round_id}.txt')
                        counts.add((int(s['task_refs']), int(s['waits'])))
                if len(counts) != 1: raise ValueError('schedule counts differ between arms')
                tasks, waits = counts.pop()
                cells.append(dict(model=model, seq=seq, task_refs=tasks, waits=waits,
                                  notify_ns=statistics.median(notify), poll_total_ns=statistics.median(wait)))
        rates = {'notify': fit(cells, 'notify_ns', 'task_refs'),
                 'poll': fit(cells, 'poll_total_ns', 'waits')}
        if any(v <= 0 for v in rates.values()): raise ValueError('nonpositive fitted event rate')
        for i, cell in enumerate(cells):
            train = cells[:i]+cells[i+1:]
            for label, response, count in [('notify','notify_ns','task_refs'),
                                           ('poll','poll_total_ns','waits')]:
                predicted = fit(train, response, count)*cell[count]
                cell[f'{label}_loo_predicted_ns'] = predicted
                cell[f'{label}_loo_relative_error'] = (predicted-cell[response])/cell[response]
        with (a.out/f'{arch}_cross_validation.tsv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(cells[0]), delimiter='\t', lineterminator='\n')
            w.writeheader(); w.writerows(cells)
        record = {'dtype': 'bf16', 'method': 'three-cell leave-one-out; final origin-constrained least squares on four cell medians',
                  'source': str((raw/'attrib.tsv').relative_to(a.repo)),
                  'source_sha256': hashlib.sha256((raw/'attrib.tsv').read_bytes()).hexdigest(),
                  'notify': {'ns': rates['notify'], 'reason': 'measured', 'unit': 'ns/runtime_task_ref'},
                  'poll': {'ns': rates['poll'], 'reason': 'measured', 'unit': 'ns/runtime_wait_entry'},
                  'fence': {'ns': None, 'reason': 'not_calibrated'}}
        (a.out/f'{arch}.json').write_text(json.dumps(record, indent=2)+'\n')
        print(arch, rates)
        for c in cells:
            print(c['model'], c['seq'], 'notify_LOO', c['notify_loo_relative_error'],
                  'poll_LOO', c['poll_loo_relative_error'])


if __name__ == '__main__':
    main()
