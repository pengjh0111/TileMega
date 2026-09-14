#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""EX-E3 step 1: what dropping three of the five per-task barriers is worth.

Two tables.  The first is the four-arm decomposition of L2_ATTRIB/summarize.py,
computed once per barrier arm, so the protocol terms can be compared under both
shapes rather than only the totals.  The second is the paired v2-on/v2-off
ratio on the `full` arm -- paired because both were run in the same session and
the same round, alternating which went first (H6).
"""
import glob
import math
import os
import random
import re
import statistics
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")
ARMS = ("neither", "nowait", "full", "l1nosync")


def samples(raw, model, seq, v2, arm):
    result = {}
    pattern = os.path.join(raw, "final", f"{model}_s{seq}_v{v2}_{arm}",
                           "r*", "run_*.log")
    for path in glob.glob(pattern):
        round_match = ROUND.search(path)
        for line in open(path):
            match = TIME.match(line)
            if match:
                result[int(round_match.group(1))] = tuple(map(float, match.groups()))
    return result


def bootstrap(values, draws=20000):
    # Same seed and draw count as the round-two summarizers, so a CI here is
    # comparable with the ones already in FINDINGS.
    rng = random.Random(20260906)
    n = len(values)
    estimates = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def signed_rank(values):
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


raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "raw_barrier")

print("model\tseq\tv2\tn\tl1\tl2\tgap\twait\tnotify\tbarrier\tloop\tclosure_us")
paired = []
for model in ("gqa2", "mha4"):
    for seq in (4, 128):
        per_v2 = {}
        for v2 in (0, 1):
            data = {arm: samples(raw, model, seq, v2, arm) for arm in ARMS}
            if not all(data.values()):
                continue
            rounds = sorted(set.intersection(*(set(rows) for rows in data.values())))
            columns = {name: [] for name in ("l1", "l2", "gap", "wait", "notify",
                                             "barrier", "loop", "closure")}
            for r in rounds:
                full_l1, full_l2 = data["full"][r]
                values = {
                    "l1": full_l1, "l2": full_l2, "gap": full_l2 - full_l1,
                    "wait": full_l2 - data["nowait"][r][1],
                    "notify": data["nowait"][r][1] - data["neither"][r][1],
                    "barrier": full_l1 - data["l1nosync"][r][0],
                    "loop": data["neither"][r][1] - data["l1nosync"][r][0],
                }
                values["closure"] = values["gap"] - (
                    values["wait"] + values["notify"] + values["loop"]
                    - values["barrier"])
                for name, value in values.items():
                    columns[name].append(value)
            med = {name: statistics.median(v) for name, v in columns.items()}
            per_v2[v2] = (rounds, columns, med)
            print(f"{model}\t{seq}\t{v2}\t{len(rounds)}\t{med['l1']:.6f}\t"
                  f"{med['l2']:.6f}\t{med['gap']:.6f}\t{med['wait']:.6f}\t"
                  f"{med['notify']:.6f}\t{med['barrier']:.6f}\t{med['loop']:.6f}\t"
                  f"{med['closure'] * 1000:.6f}")
        if 0 in per_v2 and 1 in per_v2:
            shared = sorted(set(per_v2[0][0]) & set(per_v2[1][0]))
            off = {r: v for r, v in zip(per_v2[0][0], per_v2[0][1]["l2"])}
            on = {r: v for r, v in zip(per_v2[1][0], per_v2[1][1]["l2"])}
            ratios = [on[r] / off[r] for r in shared]
            deltas = [on[r] - off[r] for r in shared]
            low, high = bootstrap(ratios)
            paired.append((model, seq, len(shared), statistics.median(ratios),
                           low, high, statistics.median(deltas),
                           signed_rank(deltas)))

print("\nmodel\tseq\tn\tratio_on_over_off\tci95_low\tci95_high\tdelta_ms\tp")
for row in paired:
    print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.6f}\t{row[4]:.6f}\t"
          f"{row[5]:.6f}\t{row[6]:.6f}\t{row[7]:.3e}")
