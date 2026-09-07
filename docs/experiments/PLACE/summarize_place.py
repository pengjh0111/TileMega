#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Within-round comparison of the two materialized queue orders."""
import glob
import math
import os
import random
import re
import statistics
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def read(raw, model, arm):
    result = {}
    for path in glob.glob(os.path.join(raw, "final", f"{model}_{arm}", "r*", "run_*.log")):
        round_number = int(ROUND.search(path).group(1))
        for line in open(path):
            match = TIME.match(line)
            if match:
                result[round_number] = float(match.group(1))
    return result


def bootstrap(values, draws=20000):
    rng = random.Random(20260906)
    n = len(values)
    medians = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                     for _ in range(draws))
    return medians[int(.025 * draws)], medians[int(.975 * draws) - 1]


def signed_rank(values):
    values = [value for value in values if value]
    ordered = sorted(values, key=abs)
    ranks = list(range(1, len(ordered) + 1))
    positive = sum(rank for rank, value in zip(ranks, ordered) if value > 0)
    n = len(values)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    z = (abs(positive - mean) - .5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


raw = sys.argv[1]
print("model\tn\tcritical_path_ms\tround_robin_ms\trr_over_cp\tci95_low\tci95_high\tp\tcp_faster")
for model in ("gqa2", "mha4"):
    cp = read(raw, model, "critical_path")
    rr = read(raw, model, "round_robin")
    rounds = sorted(set(cp) & set(rr))
    ratios = [rr[r] / cp[r] for r in rounds]
    low, high = bootstrap(ratios)
    print(f"{model}\t{len(rounds)}\t{statistics.median(cp.values()):.6f}\t"
          f"{statistics.median(rr.values()):.6f}\t{statistics.median(ratios):.6f}\t"
          f"{low:.6f}\t{high:.6f}\t{signed_rank([rr[r]-cp[r] for r in rounds]):.3e}\t"
          f"{sum(rr[r] > cp[r] for r in rounds)}/{len(rounds)}")
