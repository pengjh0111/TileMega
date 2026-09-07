#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
import glob
import math
import os
import random
import re
import statistics
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def samples(raw, model, seq, arm):
    result = {}
    for path in glob.glob(os.path.join(raw, "final", f"{model}_s{seq}_{arm}", "r*", "run_*.log")):
        round_number = int(ROUND.search(path).group(1))
        for line in open(path):
            match = TIME.match(line)
            if match:
                result[round_number] = float(match.group(1))
    return result


def bootstrap(values, draws=20000):
    rng = random.Random(20260906)
    n = len(values)
    estimates = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def signed_rank(values):
    values = [value for value in values if value]
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = .5 * (i + j) + 1
        i = j + 1
    positive = sum(ranks[i] for i, value in enumerate(values) if value > 0)
    n = len(values)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    z = (abs(positive - mean) - .5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


raw = sys.argv[1]
print("model\tseq\tn\texact_ms\tall_ms\tall_over_exact\tci95_low\tci95_high\tp")
for model in ("gqa2", "mha4"):
    for seq in (4, 128, 512):
        exact = samples(raw, model, seq, "exact")
        all_deps = samples(raw, model, seq, "all")
        rounds = sorted(set(exact) & set(all_deps))
        ratios = [all_deps[r] / exact[r] for r in rounds]
        low, high = bootstrap(ratios)
        print(f"{model}\t{seq}\t{len(rounds)}\t{statistics.median(exact.values()):.6f}\t"
              f"{statistics.median(all_deps.values()):.6f}\t{statistics.median(ratios):.6f}\t"
              f"{low:.6f}\t{high:.6f}\t{signed_rank([all_deps[r]-exact[r] for r in rounds]):.3e}")
