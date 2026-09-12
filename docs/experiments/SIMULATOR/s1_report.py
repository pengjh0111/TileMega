#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""EX-S1 gates S1-a, S1-b and S1-c, computed from the raw artifacts only.

S1-a compares per-task start times against the trace.  Both sides are shifted
to their own first task: the trace carries absolute %globaltimer values and the
simulator starts at zero, so only the offsets within a launch are comparable.
The gate sets no threshold on the absolute error and requires it to be
reported, so this prints the distribution and never a verdict.

S1-b is the hard gate, and the ranking is what it is about.  Measured l2_ms
comes from the untraced arm; predicted makespan from the `proportional` arm,
the one the real models run.
"""
import argparse, collections, csv, math, os, statistics, sys

CANDIDATE_MODE = {'legacy_grid_stride': 0, 'balanced': 4, 'rotate': 5}


def read_tsv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def quantile(values, q):
    if not values:
        return float('nan')
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def spearman(xs, ys):
    """Rank correlation with midranks, so ties do not silently become order."""
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    if len(xs) < 2:
        return float('nan')
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    raw, out = args.raw, args.out

    predicted = read_tsv(os.path.join(raw, 'predicted.tsv'))
    timing = read_tsv(os.path.join(raw, 'time', 'l2.tsv'))

    # --- S1-a -------------------------------------------------------------
    rows = []
    for record in sorted({(r['model'], int(r['seq'])) for r in predicted}):
        model, seq = record
        # Real width is built for the S1-c budget and carries no trace, so it
        # is absent here by design rather than by omission.
        if model == 'real':
            continue
        for place in (0, 4, 5):
            task_path = os.path.join(raw, f'tasks_{model}_s{seq}_p{place}.tsv')
            slot_path = os.path.join(raw, 'dump', f'{model}_s{seq}_p{place}', 'slots.tsv')
            if not (os.path.exists(task_path) and os.path.exists(slot_path)):
                print(f'S1-a  {model:5s} s{seq:<4d} p{place}  MISSING', file=sys.stderr)
                continue
            pred = {(int(r['stage']), int(r['logical'])): float(r['start_ns'])
                    for r in read_tsv(task_path)}
            meas = {(int(r['stage']), int(r['logical_task'])): float(r['run_begin'])
                    for r in read_tsv(slot_path)}
            shared = sorted(set(pred) & set(meas))
            if not shared:
                continue
            p0 = min(pred[k] for k in shared)
            m0 = min(meas[k] for k in shared)
            errors = [(pred[k] - p0) - (meas[k] - m0) for k in shared]
            span = max(meas[k] for k in shared) - m0
            absolute = [abs(e) for e in errors]
            rows.append({
                'model': model, 'seq': seq, 'place': place, 'tasks': len(shared),
                'measured_span_ns': span,
                'signed_p50': quantile(errors, 0.50),
                'abs_p50': quantile(absolute, 0.50),
                'abs_p90': quantile(absolute, 0.90),
                'abs_max': max(absolute),
                'abs_p90_pct_span': 100.0 * quantile(absolute, 0.90) / span if span else float('nan'),
            })
    with open(os.path.join(out, 's1_start_error.tsv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('S1-a  per-task start error, both sides shifted to their first task')
    print(f'  {"cell":16s} {"tasks":>6s} {"span_ns":>11s} {"bias":>10s} '
          f'{"|p50|":>10s} {"|p90|":>10s} {"|max|":>11s} {"p90/span":>9s}')
    for r in rows:
        print(f'  {r["model"]+" s"+str(r["seq"])+" p"+str(r["place"]):16s} '
              f'{r["tasks"]:6d} {r["measured_span_ns"]:11.0f} {r["signed_p50"]:10.0f} '
              f'{r["abs_p50"]:10.0f} {r["abs_p90"]:10.0f} {r["abs_max"]:11.0f} '
              f'{r["abs_p90_pct_span"]:8.1f}%')
    print('  real width carries no trace arm; it appears under S1-c only')
    models = len({r['model'] for r in rows})
    seqs = len({r['seq'] for r in rows})
    places = len({r['place'] for r in rows})
    print(f'  coverage: {models} models x {seqs} seq x {places} placements '
          f'= {len(rows)} cells (gate asks >= 2 x 3 x 3)')

    # --- S1-b -------------------------------------------------------------
    measured = collections.defaultdict(list)
    for r in timing:
        measured[(r['model'], int(r['seq']), int(r['place']))].append(float(r['l2_ms']))
    scan = []
    for r in predicted:
        if r['arm'] != 'proportional' or r['status'] != 'ok':
            continue
        mode = CANDIDATE_MODE.get(r['candidate'])
        if mode is None:
            continue
        key = (r['model'], int(r['seq']), mode)
        if key not in measured:
            continue
        scan.append({
            'model': r['model'], 'seq': int(r['seq']), 'candidate': r['candidate'],
            'place': mode, 'predicted_ns': float(r['makespan_ns']),
            'measured_ns': statistics.median(measured[key]) * 1e6,
            'rounds': len(measured[key]),
        })
    with open(os.path.join(out, 's1_ranking.tsv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=list(scan[0]))
        writer.writeheader()
        writer.writerows(scan)

    print('\nS1-b  ranking over the placement x config scan')
    print(f'  {"cell":16s} {"candidate":20s} {"predicted_us":>13s} {"measured_us":>12s} {"ratio":>7s}')
    for r in sorted(scan, key=lambda r: (r['model'], r['seq'], r['place'])):
        print(f'  {r["model"]+" s"+str(r["seq"]):16s} {r["candidate"]:20s} '
              f'{r["predicted_ns"]/1e3:13.1f} {r["measured_ns"]/1e3:12.1f} '
              f'{r["predicted_ns"]/r["measured_ns"]:7.3f}')

    per_config = []
    for cell in sorted({(r['model'], r['seq']) for r in scan}):
        group = [r for r in scan if (r['model'], r['seq']) == cell]
        rho = spearman([r['predicted_ns'] for r in group], [r['measured_ns'] for r in group])
        best_pred = min(group, key=lambda r: r['predicted_ns'])['candidate']
        best_meas = min(group, key=lambda r: r['measured_ns'])['candidate']
        per_config.append((cell, len(group), rho, best_pred, best_meas))
    print(f'\n  {"cell":16s} {"n":>3s} {"rho":>7s}  {"argmin predicted":20s} {"argmin measured":20s}')
    for cell, n, rho, bp, bm in per_config:
        mark = 'same' if bp == bm else 'DIFFERENT'
        print(f'  {cell[0]+" s"+str(cell[1]):16s} {n:3d} {rho:7.3f}  {bp:20s} {bm:20s} {mark}')

    agree = sum(1 for _, _, _, bp, bm in per_config if bp == bm)
    pooled = spearman([r['predicted_ns'] for r in scan], [r['measured_ns'] for r in scan])
    # Within-config z-scores: the pooled raw correlation is dominated by the
    # spread between configs, which no placement decision can change, so the
    # question "does it rank placements" needs the config mean removed.
    centred_x, centred_y = [], []
    for cell in {(r['model'], r['seq']) for r in scan}:
        group = [r for r in scan if (r['model'], r['seq']) == cell]
        mx = statistics.fmean(r['predicted_ns'] for r in group)
        my = statistics.fmean(r['measured_ns'] for r in group)
        centred_x += [r['predicted_ns'] - mx for r in group]
        centred_y += [r['measured_ns'] - my for r in group]
    within = spearman(centred_x, centred_y)
    print(f'\n  pooled Spearman (raw, n={len(scan)}):            {pooled:.3f}')
    print(f'  within-config Spearman (config mean removed): {within:.3f}')
    print(f'  argmin agreement: {agree}/{len(per_config)} configs pick the same placement')

    # "model top-3 contains a member of the measured top 3%" -- on a scan of
    # n cells, top 3% is max(1, floor(0.03n)) cells, so at n = 18 it is the
    # single fastest cell.  Reported as the gate defines it, not softened.
    top_k = max(1, int(0.03 * len(scan)))
    measured_top = {id(r) for r in sorted(scan, key=lambda r: r['measured_ns'])[:top_k]}
    predicted_top3 = sorted(scan, key=lambda r: r['predicted_ns'])[:3]
    hit = any(id(r) in measured_top for r in predicted_top3)
    print(f'  measured top 3% of the {len(scan)}-cell scan = {top_k} cell(s): '
          + ', '.join(f'{r["model"]} s{r["seq"]} {r["candidate"]}'
                      for r in sorted(scan, key=lambda r: r['measured_ns'])[:top_k]))
    print('  predicted top 3: '
          + ', '.join(f'{r["model"]} s{r["seq"]} {r["candidate"]}' for r in predicted_top3))
    print(f'  S1-b top-3 criterion: {"PASS" if hit else "FAIL"}')

    # --- S1-c -------------------------------------------------------------
    # S1-c states two budgets, so it is checked against two.  Real width being
    # absent is a FAIL of the real-width half, never a silent pass.
    print('\nS1-c  evaluation time per Plan (proportional arm)')
    failures, seen_real = [], False
    for cell in sorted({(r['model'], int(r['seq'])) for r in predicted}):
        group = [float(r['eval_us']) for r in predicted
                 if r['model'] == cell[0] and int(r['seq']) == cell[1]
                 and r['status'] == 'ok' and r['arm'] == 'proportional']
        if not group:
            continue
        real = cell[0] == 'real'
        seen_real = seen_real or real
        budget = 10000.0 if real else 1000.0
        verdict = 'PASS' if max(group) < budget else 'FAIL'
        if verdict == 'FAIL':
            failures.append((cell, max(group), budget))
        print(f'  {cell[0]+" s"+str(cell[1]):16s} n={len(group)} '
              f'median={statistics.median(group):9.1f}us max={max(group):9.1f}us  '
              f'vs {budget:7.0f}us  {verdict}')
    for cell, worst, budget in failures:
        print(f'  over budget: {cell[0]} s{cell[1]} at {worst/budget:.1f}x '
              f'({worst:.0f}us vs {budget:.0f}us)')
    if not seen_real:
        print('  real width: MISSING -- the < 10 ms half of S1-c is unchecked')
    print(f'  S1-c: {"PASS" if not failures and seen_real else "FAIL"} '
          '(gate fixed before implementation, not moved -- H7)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
