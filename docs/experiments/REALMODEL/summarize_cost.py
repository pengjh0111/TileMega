#!/usr/bin/env python3
"""Summarize the real-width control/candidate paired measurements."""

import csv
import math
import random
import statistics
import sys
from pathlib import Path


def wilcoxon(deltas):
    values = sorted((abs(value), value) for value in deltas if value != 0)
    n = len(values)
    if n < 6:
        return float("nan")
    ranks = [0.0] * n
    ties = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1][0] == values[i][0]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = rank
        run = j - i + 1
        ties += run**3 - run
        i = j + 1
    positive = sum(rank for rank, (_, value) in zip(ranks, values)
                   if value > 0)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "docs/experiments/REALMODEL/raw/cost_paired.tsv")
    rows = list(csv.DictReader(source.open(encoding="utf-8"), delimiter="\t"))
    if len(rows) != 25:
        raise RuntimeError(f"expected 25 paired rounds, got {len(rows)}")
    print("metric\trounds\tcontrol_median_ms\tcandidate_median_ms\t"
          "candidate_vs_control_pct\tci_low\tci_high\twilcoxon_p")
    for metric in ("l1", "l2"):
        control = [float(row[f"control_{metric}_ms"]) for row in rows]
        candidate = [float(row[f"candidate_{metric}_ms"]) for row in rows]
        deltas = [100 * (new / old - 1)
                  for new, old in zip(candidate, control)]
        rng = random.Random(20260906)
        bootstrap = sorted(
            statistics.median(deltas[rng.randrange(len(deltas))]
                              for _ in deltas)
            for _ in range(20000))
        print(f"{metric}\t{len(rows)}\t{statistics.median(control):.6f}\t"
              f"{statistics.median(candidate):.6f}\t"
              f"{statistics.median(deltas):.4f}\t{bootstrap[500]:.4f}\t"
              f"{bootstrap[19499]:.4f}\t{wilcoxon(deltas):.9g}")


if __name__ == "__main__":
    main()
