#!/usr/bin/env python3
"""Paired L1-vs-L2 statistics: each fresh process times both, so the pairing
is within-round and the ratio is taken per round before it is aggregated."""
import glob, math, random, re, sys

def read(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            if line.startswith('E2E_TIME'):
                f = dict(kv.split('=') for kv in line.split()[1:])
                rows.append((float(f['l1_ms']), float(f['l2_ms'])))
    return rows

def median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def wilcoxon(d):
    """Two-sided signed-rank on the paired differences, normal approximation
    with tie and continuity correction; n=25 is well inside its validity."""
    d = [x for x in d if x != 0.0]
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    i = 0
    ties = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    w_pos = sum(rank[i] for i in range(n) if d[i] > 0)
    w_neg = sum(rank[i] for i in range(n) if d[i] < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0)
    z = (abs(w - mu) - 0.5) / sd
    p = math.erfc(z / math.sqrt(2))
    return w_pos, w_neg, z, p

def boot(xs, reps=20000, seed=20260904):
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(median([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]

for label, pattern in [a.split('=', 1) for a in sys.argv[1:]]:
    rows = read(pattern)
    l1 = [a for a, _ in rows]; l2 = [b for _, b in rows]
    ratios = [b / a for a, b in rows]
    diffs = [b - a for a, b in rows]
    lo, hi = boot(ratios)
    wp, wn, z, p = wilcoxon(diffs)
    print(f"MODEL {label} n={len(rows)}")
    print(f"  l1_median_ms={median(l1):.6f} l2_median_ms={median(l2):.6f}")
    print(f"  median_of_ratios={median(ratios):.6f} "
          f"ratio_of_medians={median(l2)/median(l1):.6f}")
    print(f"  bootstrap95_median_ratio=[{lo:.6f}, {hi:.6f}] reps=20000")
    print(f"  median_paired_diff_ms={median(diffs):+.6f} "
          f"min_ratio={min(ratios):.6f} max_ratio={max(ratios):.6f}")
    print(f"  wilcoxon W+={wp:.1f} W-={wn:.1f} z={z:.3f} p={p:.3e} "
          f"faster_rounds={sum(1 for r in ratios if r < 1.0)}/{len(ratios)}")
