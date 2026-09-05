#!/usr/bin/env python3
"""Paired bootstrap/Wilcoxon summary for RoPE ownership arms."""
import glob
import math
import os
import random
import re
import sys


TIME = re.compile(
    r"^E2E_TIME l05_ms=([0-9.]+) l1_ms=([0-9.]+).* l2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def load(root, model, arm, field):
    values = {}
    for path in glob.glob(os.path.join(root, f"{model}_{arm}", "r*", "run_*.log")):
        match_round = ROUND.search(path)
        if not match_round:
            continue
        for line in open(path, encoding="utf-8"):
            match = TIME.match(line)
            if match:
                values[int(match_round.group(1))] = float(match.group(field))
    return values


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def wilcoxon(deltas):
    values = sorted((abs(x), x) for x in deltas if x != 0)
    n = len(values)
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
        ties += run ** 3 - run
        i = j + 1
    if n < 6:
        return float("nan")
    positive = sum(rank for rank, (_, value) in zip(ranks, values) if value > 0)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def paired(base, changed, draws=20000):
    rounds = sorted(set(base) & set(changed))
    deltas = [100 * (changed[r] / base[r] - 1) for r in rounds]
    rng = random.Random(20260905)
    bootstrap = sorted(median([deltas[rng.randrange(len(deltas))]
                               for _ in deltas]) for _ in range(draws))
    return (median(deltas), bootstrap[int(.025 * draws)],
            bootstrap[int(.975 * draws) - 1], wilcoxon(deltas), len(rounds))


def main():
    root = sys.argv[1]
    print("model\tmetric\trounds\tchunk_median_ms\ttile_median_ms\t"
          "tile_vs_chunk_pct\tci_low\tci_high\twilcoxon_p")
    for model in ("gqa2", "mha4"):
        for metric, field in (("l05", 1), ("l1", 2), ("l2", 3)):
            base, changed = load(root, model, "chunk", field), load(root, model, "tile", field)
            point, low, high, p, n = paired(base, changed)
            print(f"{model}\t{metric}\t{n}\t{median(base.values()):.6f}\t"
                  f"{median(changed.values()):.6f}\t{point:.4f}\t{low:.4f}\t"
                  f"{high:.4f}\t{p:.6g}")


if __name__ == "__main__":
    main()
