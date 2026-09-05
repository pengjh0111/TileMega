#!/usr/bin/env python3
"""Deterministic paired bootstrap and Wilcoxon report for ORACLE finalists."""
import csv
import math
import random
import statistics
import sys
from pathlib import Path


def wilcoxon(deltas):
    values = sorted((abs(x), x) for x in deltas if x != 0)
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
    positive = sum(rank for rank, (_, value) in zip(ranks, values) if value > 0)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def main():
    root = Path(sys.argv[1])
    print("model\tmetric\trounds\tcontrol_median_ms\tbest_median_ms\t"
          "best_vs_control_pct\tci_low\tci_high\twilcoxon_p\tbest_tag")
    for model in ("gqa2", "mha4"):
        rows = list(csv.DictReader((root / f"paired_{model}.tsv").open(),
                                   delimiter="\t"))
        by_arm = {arm: {int(r["round"]): r for r in rows if r["arm"] == arm}
                  for arm in ("control", "best")}
        rounds = sorted(set(by_arm["control"]) & set(by_arm["best"]))
        if len(rounds) != 25:
            raise RuntimeError(f"{model}: expected 25 paired rounds, got {len(rounds)}")
        best_tag = by_arm["best"][rounds[0]]["tag"]
        for metric in ("l05_ms", "l1_ms", "l2_ms"):
            control = [float(by_arm["control"][r][metric]) for r in rounds]
            best = [float(by_arm["best"][r][metric]) for r in rounds]
            deltas = [100 * (b / c - 1) for b, c in zip(best, control)]
            rng = random.Random(20260905)
            boot = sorted(statistics.median(
                deltas[rng.randrange(len(deltas))] for _ in deltas)
                for _ in range(20000))
            print(f"{model}\t{metric[:-3]}\t{len(rounds)}\t"
                  f"{statistics.median(control):.6f}\t"
                  f"{statistics.median(best):.6f}\t"
                  f"{statistics.median(deltas):.4f}\t"
                  f"{boot[500]:.4f}\t{boot[19499]:.4f}\t"
                  f"{wilcoxon(deltas):.9g}\t{best_tag}")


if __name__ == "__main__":
    main()
