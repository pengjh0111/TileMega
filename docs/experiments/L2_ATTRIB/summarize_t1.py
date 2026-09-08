#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Paired reductions of run_t1.py's per-process rows (times in ms)."""
import argparse
import csv
from pathlib import Path
import math
import random
import statistics

p = argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--baseline', default='base')
p.add_argument('--out', required=True)
p.add_argument('--expected-rounds', type=int, default=25)
a = p.parse_args()
rows = list(csv.DictReader(open(a.input), delimiter='\t'))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
data = {}
for row in rows:
    key = row['model'], int(row['seq']), row['variant'], int(row['round'])
    data.setdefault(key, {})[row['arm']] = tuple(float(row[c]) for c in ['l1_ms', 'l2_ms'])
expected_arms = set(row['arm'] for row in rows)
for key, arms in data.items():
    if set(arms) != expected_arms:
        raise ValueError(f'incomplete paired round: {key}')
for group in {k[:3] for k in data}:
    if sum(k[:3] == group for k in data) != a.expected_rounds:
        raise ValueError(f'incomplete group: {group}')

def stats(values):
    rng = random.Random(20260908)
    n = len(values)
    boot = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                  for _ in range(20000))
    return [statistics.median(values), boot[500], boot[19499]]

def signed_rank(values):
    x = sorted((abs(v), v > 0) for v in values if v)
    n = len(x)
    if not n:
        return 1.0
    positive = ties = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[j][0] == x[i][0]:
            j += 1
        positive += .5 * (i + 1 + j) * sum(sign for _, sign in x[i:j])
        ties += (j-i)**3 - (j-i)
        i = j
    variance = n*(n+1)*(2*n+1)/24 - ties/48
    z = max(0.0, abs(positive-n*(n+1)/4)-.5)/math.sqrt(variance)
    return math.erfc(z/math.sqrt(2))

derived = {}
for (model, seq, variant, r), arms in data.items():
    if 'full' not in arms:
        continue
    l1, l2 = arms['full']
    vals = {'l1_ms': l1, 'l2_ms': l2, 'ratio': l2/l1}
    if set(['nowait', 'neither', 'l1nosync']).issubset(arms):
        neither, nowait, nosync = arms['neither'][1], arms['nowait'][1], arms['l1nosync'][0]
        vals.update(notify_ms=nowait-neither, wait_ms=l2-nowait,
                    barrier_ms=l1-nosync, loop_ms=neither-nosync,
                    gap_ms=l2-l1, closure_us=(l2-(neither+(nowait-neither)+(l2-nowait)))*1000)
    derived[model, seq, variant, r] = vals
with (out / 'summary.tsv').open('w') as h:
    w = csv.writer(h, delimiter='\t', lineterminator='\n')
    w.writerow(['model', 'seq', 'variant', 'metric', 'rounds', 'median', 'ci_low', 'ci_high'])
    for model, seq, variant in sorted({k[:3] for k in derived}):
        values = [v for k, v in derived.items() if k[:3] == (model, seq, variant)]
        for metric in values[0]:
            w.writerow([model, seq, variant, metric, len(values), *stats([v[metric] for v in values])])
with (out / 'paired.tsv').open('w') as h:
    w = csv.writer(h, delimiter='\t', lineterminator='\n')
    w.writerow(['model', 'seq', 'variant', 'baseline', 'metric', 'rounds',
                'delta_median', 'ci_low', 'ci_high', 'wilcoxon_p'])
    for model, seq, variant in sorted({k[:3] for k in derived}):
        if variant == a.baseline:
            continue
        pairs = [(v, derived[model, seq, a.baseline, k[3]]) for k, v in derived.items()
                 if k[:3] == (model, seq, variant) and (model, seq, a.baseline, k[3]) in derived]
        if not pairs:
            continue
        for metric in pairs[0][0]:
            differences = [x[metric] - b[metric] for x, b in pairs]
            prob = signed_rank(differences)
            w.writerow([model, seq, variant, a.baseline, metric, len(pairs),
                        *stats(differences), prob])
print(out / 'summary.tsv')
print(out / 'paired.tsv')
