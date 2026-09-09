#!/usr/bin/env python3
"""A9 counts-times-rates fit; steady-state paired cells, never cold archives."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURES = {
    'notify': ('variant_stages','max_worker_task_refs','task_refs'),
    'poll': ('variant_stages','max_worker_task_refs','waits'),
}


def fit(x, y):
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('not_calibrated: feature matrix is rank deficient')
    scale = np.linalg.norm(x, axis=0)
    normalized = x/scale
    # Three regressors: enumerate every nonnegative least-squares face,
    # including the all-zero face. No fitting hyperparameter or ridge term.
    best = np.zeros(x.shape[1])
    best_error = float(y@y)
    for bits in range(1,1 << x.shape[1]):
        active = [i for i in range(x.shape[1]) if bits & (1 << i)]
        values = np.linalg.lstsq(normalized[:,active],y,rcond=None)[0]
        if np.any(values<0):
            continue
        candidate = np.zeros(x.shape[1])
        candidate[active] = values
        residual = normalized@candidate-y
        error = float(residual@residual)
        if error<best_error:
            best,best_error = candidate,error
    return best/scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument('raw',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('refusing to overwrite fit evidence')
    a.out.mkdir(parents=True)
    manifest = json.loads((a.raw/'calibration_command.json').read_text())
    if (manifest['warmup'],manifest['repeat']) != (5,11):
        raise RuntimeError('fit requires the recorded steady-state timing policy')
    rows = list(csv.DictReader((a.raw/'attrib.tsv').open(),delimiter='\t'))
    paired = {}
    for row in rows:
        if tuple(row[name] for name in ('cold','warmup','repeat')) != ('0','5','11'):
            raise RuntimeError('runtime timing policy is not warmup=5/repeat=11 steady state')
        key = tuple(row[name] for name in ('model','variant','seq','past','round'))
        if row['arm'] in paired.setdefault(key,{}):
            raise RuntimeError('duplicate arm in a process round')
        paired[key][row['arm']] = row
    cells = {}
    for key, arms in paired.items():
        if set(arms) != {'full','nowait','neither','l1nosync'}:
            raise RuntimeError(f'incomplete four-arm round: {key}')
        full = arms['full']
        if full['pass'] != '1':
            raise RuntimeError('full-arm correctness failure is not a calibration point')
        for field in ('grid','block','variant_stages','task_refs','waits','max_worker_task_refs'):
            if len({row[field] for row in arms.values()}) != 1:
                raise RuntimeError(f'four-arm structure/occupancy confound: {key} {field}')
        nowait = float(arms['nowait']['l2_ms'])
        neither = float(arms['neither']['l2_ms'])
        item = dict(full)
        item['notify_ns'] = (nowait-neither)*1e6
        item['poll_ns'] = (float(full['l2_ms'])-nowait)*1e6
        cells.setdefault(key[:-1],[]).append(item)
    if len(cells)<12 or any(len(rounds)<25 for rounds in cells.values()):
        raise RuntimeError('requires at least twelve cells and 25 complete paired rounds each')
    output = dict(model='stages + longest_worker_queue + total_work',
                  fit='nonnegative least squares; no intercept or regularization',
                  hardware_publication='pending A2 and functional gate',
                  fence_ns=dict(reason='not_calibrated'),cells=len(cells),rates={})
    with (a.out/'cross_validation.tsv').open('w') as stream:
        writer = csv.writer(stream,delimiter='\t',lineterminator='\n')
        writer.writerow(['term','model','variant','seq','past','observed_ns',
                         'predicted_ns','relative_error','reason'])
        for term,names in FEATURES.items():
            keys = list(cells)
            for rounds in cells.values():
                for name in names:
                    if len({row[name] for row in rounds})!=1:
                        raise RuntimeError('feature changed across rounds of one cell')
            x = np.array([[float(cells[key][0][name]) for name in names] for key in keys])
            y = np.array([np.median([row[term+'_ns'] for row in cells[key]]) for key in keys])
            coefficients = fit(x,y)
            output['rates'][term] = dict(zip(names,coefficients.tolist()))
            for i,key in enumerate(keys):
                mask = np.arange(len(keys))!=i
                prediction = float(x[i]@fit(x[mask],y[mask]))
                relative = (prediction-y[i])/abs(y[i]) if y[i] else ''
                writer.writerow([term,*key,y[i],prediction,relative,
                                 '' if y[i] else 'zero_observed_cost'])
    (a.out/'coefficients.json').write_text(json.dumps(output,indent=2)+'\n')
    (a.out/'status.txt').write_text('FIT COMPLETE; residuals do not close or block A9.3\n')


if __name__ == '__main__':
    main()
