#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Paired medians, bootstrap CI and signed-rank test for queue L2 vs L1."""
import glob
import math
import os
import random
import re
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def read(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            match = TIME.match(line)
            if match:
                rows.append(tuple(map(float, match.groups())))
    return rows


def bootstrap(values, draws=20000, seed=20260906):
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(median([values[rng.randrange(n)] for _ in range(n)])
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def signed_rank(deltas):
    values = [d for d in deltas if d]
    n = len(values)
    ordered = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[ordered[j + 1]]) == abs(values[ordered[i]]):
            j += 1
        rank = .5 * (i + j) + 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = rank
        count = j - i + 1
        tie_term += count ** 3 - count
        i = j + 1
    positive = sum(ranks[i] for i, value in enumerate(values) if value > 0)
    negative = sum(ranks[i] for i, value in enumerate(values) if value < 0)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    z = (abs(min(positive, negative) - mean) - .5) / math.sqrt(variance)
    return positive, negative, math.erfc(z / math.sqrt(2))


raw = sys.argv[1]
print("model\tseq\tn\tl1_median_ms\tl2_median_ms\tmedian_ratio\tci95_low\tci95_high\tw_plus\tw_minus\tp\tl2_faster")
for model in ("gqa2", "mha4"):
    for seq in (4, 128, 512):
        rows = read(os.path.join(raw, "final", f"{model}_s{seq}", "run_*.log"))
        l1 = [row[0] for row in rows]
        l2 = [row[1] for row in rows]
        ratios = [b / a for a, b in rows]
        low, high = bootstrap(ratios)
        wp, wm, p = signed_rank([b - a for a, b in rows])
        print(f"{model}\t{seq}\t{len(rows)}\t{median(l1):.6f}\t{median(l2):.6f}\t"
              f"{median(ratios):.6f}\t{low:.6f}\t{high:.6f}\t{wp:.1f}\t{wm:.1f}\t"
              f"{p:.3e}\t{sum(value < 1 for value in ratios)}/{len(ratios)}")
