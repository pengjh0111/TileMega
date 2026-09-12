#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Paired mode-5 against mode-0 latency, arm by arm.

Round r contributes one sample per (arm, placement), so the ratio is formed
within a round and the medians never cross a session boundary.
"""
import glob
import math
import os
import random
import statistics
import re
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")
ARMS = ("neither", "nowait", "full", "l1nosync")
CELLS = [(model, seq) for model in ("gqa2", "mha4") for seq in (4, 128)]


def samples(raw, model, seq, arm, place):
    result = {}
    pattern = os.path.join(raw, "final", f"{model}_s{seq}_{arm}_p{place}",
                           "r*", "run_*.log")
    for path in glob.glob(pattern):
        index = int(ROUND.search(path).group(1))
        for line in open(path, errors="replace"):
            match = TIME.match(line)
            if match:
                result[index] = float(match.group(2))
    return result


def bootstrap(values, draws=20000):
    rng = random.Random(20260906)
    n = len(values)
    estimates = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def signed_rank(values):
    """Wilcoxon on the deviation from an unchanged ratio."""
    values = [value for value in values if value]
    n = len(values)
    if n < 2:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.] * n
    tie_term = 0.
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        rank = .5 * (i + j) + 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        count = j - i + 1
        tie_term += count ** 3 - count
        i = j + 1
    positive = sum(ranks[i] for i, value in enumerate(values) if value > 0)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    z = (abs(positive - mean) - .5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def main():
    raw = sys.argv[1]
    print("model\tseq\tarm\tn\tp0_median_ms\tp5_median_ms\tratio\tci_lo\tci_hi\tp_wilcoxon")
    pooled = {arm: [] for arm in ARMS}
    cells = {arm: 0 for arm in ARMS}
    for model, seq in CELLS:
        for arm in ARMS:
            zero = samples(raw, model, seq, arm, 0)
            five = samples(raw, model, seq, arm, 5)
            rounds = sorted(set(zero) & set(five))
            if not rounds:
                continue
            ratios = [five[r] / zero[r] for r in rounds]
            pooled[arm].extend(ratios)
            cells[arm] += 1
            low, high = bootstrap(ratios)
            print(f"{model}\t{seq}\t{arm}\t{len(rounds)}"
                  f"\t{statistics.median(zero[r] for r in rounds):.6f}"
                  f"\t{statistics.median(five[r] for r in rounds):.6f}"
                  f"\t{statistics.median(ratios):.4f}\t{low:.4f}\t{high:.4f}"
                  f"\t{signed_rank([x - 1 for x in ratios]):.3e}")
    for arm in ARMS:
        if not pooled[arm]:
            continue
        low, high = bootstrap(pooled[arm])
        print(f"ALL\tALL\t{arm}\t{len(pooled[arm])}\t-\t-"
              f"\t{statistics.median(pooled[arm]):.4f}\t{low:.4f}\t{high:.4f}"
              f"\t{signed_rank([x - 1 for x in pooled[arm]]):.3e}"
              f"\tcells={cells[arm]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
