#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""S2-b/S2-c/S2-d: paired ratios of every candidate against mode 5.

Round r contributes one sample per arm, so a ratio is formed inside a round and
no median crosses a session boundary.  Mode 5 is the denominator because that is
the gate this round is held to (R2 §6.3); the ratios against mode 0 and against
L1 are printed from the same rounds so the report can state all three without a
second measurement.
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
ARMS = ("legacy_grid_stride", "balanced", "rotate", "eft", "band", "wavefront")
REAL_ARMS = ("legacy_grid_stride", "rotate", "eft", "band", "wavefront")
PROBES = ("neither", "nowait", "full", "l1nosync")


def samples(pattern):
    """round -> (l1_ms, l2_ms) for one arm of one cell."""
    result = {}
    for path in glob.glob(pattern):
        index = int(ROUND.search(path).group(1))
        for line in open(path, errors="replace"):
            match = TIME.match(line)
            if match:
                result[index] = (float(match.group(1)), float(match.group(2)))
    return result


def bootstrap(values, draws=20000):
    rng = random.Random(20260912)
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


def row(tag, model, seq, arm, base, arm_s, n=None):
    rounds = sorted(set(base) & set(arm_s))
    if not rounds:
        return
    ratios = [arm_s[r][1] / base[r][1] for r in rounds]
    low, high = bootstrap(ratios)
    print(f"{tag}\t{model}\t{seq}\t{arm}\t{len(rounds)}"
          f"\t{statistics.median(base[r][1] for r in rounds):.6f}"
          f"\t{statistics.median(arm_s[r][1] for r in rounds):.6f}"
          f"\t{statistics.median(ratios):.4f}\t{low:.4f}\t{high:.4f}"
          f"\t{signed_rank([x - 1 for x in ratios]):.3e}")


def main():
    raw = sys.argv[1]
    print("gate\tmodel\tseq\tarm\tn\tbase_median_ms\tarm_median_ms"
          "\tratio\tci_lo\tci_hi\tp_wilcoxon")

    cells = [(m, s, ARMS) for s in (4, 128) for m in ("gqa2", "mha4")]
    cells += [("real", s, REAL_ARMS) for s in (4, 128)]
    pooled = {arm: [] for arm in ARMS}
    for model, seq, arms in cells:
        def cell(arm):
            return samples(os.path.join(raw, "final", f"{model}_s{seq}_{arm}",
                                        "r*", "run_*.log"))
        five = cell("rotate")
        if not five:
            continue
        gate = "S2-c" if model == "real" else "S2-b"
        for arm in arms:
            arm_s = cell(arm)
            row(gate, model, seq, arm, five, arm_s)
            rounds = sorted(set(five) & set(arm_s))
            if model != "real":
                pooled[arm].extend(arm_s[r][1] / five[r][1] for r in rounds)
        # The other two denominators the report has to quote, for the plan this
        # round proposes: mode 0 and L1 of the same rounds.
        zero, eft = cell("legacy_grid_stride"), cell("eft")
        rounds = sorted(set(zero) & set(eft))
        if rounds:
            row("vs0", model, seq, "eft", zero, eft)
            l1 = {r: (0.0, eft[r][0]) for r in rounds}
            row("vsL1", model, seq, "eft", l1, {r: eft[r] for r in rounds})

    for arm in ARMS:
        if not pooled[arm]:
            continue
        low, high = bootstrap(pooled[arm])
        print(f"S2-b\tALL\tALL\t{arm}\t{len(pooled[arm])}\t-\t-"
              f"\t{statistics.median(pooled[arm]):.4f}\t{low:.4f}\t{high:.4f}"
              f"\t{signed_rank([x - 1 for x in pooled[arm]]):.3e}")

    # S2-d: the four-arm decomposition, mode 5 beside the eft plan.  Reported as
    # medians rather than ratios: the unsafe arms are timing probes and their
    # numerical answer is not an acceptance condition.
    print()
    print("gate\tmodel\tseq\tarm\tprobe\tn\tl2_median_ms\tl1_median_ms")
    for seq in (4, 128):
        for model in ("gqa2", "mha4"):
            for arm in ("rotate", "eft"):
                for probe in PROBES:
                    got = samples(os.path.join(
                        raw, "final", f"dec_{model}_s{seq}_{arm}_{probe}",
                        "r*", "run_*.log"))
                    if not got:
                        continue
                    print(f"S2-d\t{model}\t{seq}\t{arm}\t{probe}\t{len(got)}"
                          f"\t{statistics.median(v[1] for v in got.values()):.6f}"
                          f"\t{statistics.median(v[0] for v in got.values()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
