#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Close the queue-era L2 and L1 additive decompositions round by round."""
import glob
import math
import os
import random
import re
import statistics
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def samples(raw, model, seq, arm):
    result = {}
    pattern = os.path.join(raw, "final", f"{model}_s{seq}_{arm}", "r*", "run_*.log")
    for path in glob.glob(pattern):
        round_match = ROUND.search(path)
        for line in open(path):
            match = TIME.match(line)
            if match:
                result[int(round_match.group(1))] = tuple(map(float, match.groups()))
    return result


def bootstrap(values, draws=20000):
    rng = random.Random(20260906)
    n = len(values)
    estimates = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def signed_rank(values):
    values = [value for value in values if value]
    n = len(values)
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


raw = sys.argv[1]
print("model\tseq\tn\tl1\tl2\tgap\twait\tnotify\tbarrier\tloop\tclosure_error_us")
details = []
for model in ("gqa2", "mha4"):
    for seq in (4, 128):
        data = {arm: samples(raw, model, seq, arm)
                for arm in ("neither", "nowait", "full", "l1nosync")}
        rounds = sorted(set.intersection(*(set(rows) for rows in data.values())))
        columns = {name: [] for name in ("l1", "l2", "gap", "wait", "notify",
                                          "barrier", "loop", "closure")}
        for round_number in rounds:
            full_l1, full_l2 = data["full"][round_number]
            neither_l2 = data["neither"][round_number][1]
            nowait_l2 = data["nowait"][round_number][1]
            nosync_l1 = data["l1nosync"][round_number][0]
            values = {
                "l1": full_l1, "l2": full_l2, "gap": full_l2 - full_l1,
                "wait": full_l2 - nowait_l2,
                "notify": nowait_l2 - neither_l2,
                "barrier": full_l1 - nosync_l1,
                "loop": neither_l2 - nosync_l1,
            }
            values["closure"] = values["gap"] - (values["wait"] + values["notify"]
                                                   + values["loop"] - values["barrier"])
            for name, value in values.items():
                columns[name].append(value)
        med = {name: statistics.median(values) for name, values in columns.items()}
        print(f"{model}\t{seq}\t{len(rounds)}\t{med['l1']:.6f}\t{med['l2']:.6f}\t"
              f"{med['gap']:.6f}\t{med['wait']:.6f}\t{med['notify']:.6f}\t"
              f"{med['barrier']:.6f}\t{med['loop']:.6f}\t{med['closure'] * 1000:.6f}")
        for name in ("gap", "wait", "notify", "barrier", "loop"):
            low, high = bootstrap(columns[name])
            details.append((model, seq, len(rounds), name, med[name], low, high,
                            signed_rank(columns[name])))

print("\nmodel\tseq\tn\tcomponent\tmedian_ms\tci95_low\tci95_high\tp_vs_zero")
for model, seq, n, name, value, low, high, p in details:
    print(f"{model}\t{seq}\t{n}\t{name}\t{value:.6f}\t{low:.6f}\t{high:.6f}\t{p:.3e}")
