#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Paired report for the kappa sweep.

Same protocol as SOLVER/summarize_per_operator.py and for the same reason
(F-46): the effects here are of the same size as the session drift, so the
statistic is the within-round ratio and never a difference of two medians.

Two families of contrast, and each is the other's control:

  l2_ms vs `stage`   what event granularity costs.  L1 is untouched by kappa.
  l1_ms vs `stage`   nothing, for every kappa arm -- nine null controls that
                     measure the protocol's own floor inside this experiment.
  l1_ms of `nosync`  the ceiling: what deleting the grid barrier is worth.
  l2_ms of `nosync`  nothing -- the tenth null control.
"""
import glob
import math
import os
import random
import re
import sys

ARMS = ("stage", "k1", "k2", "k4", "k8", "k16", "k32", "k64", "k128", "k256",
        "nosync")
L1 = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+)")
L2 = re.compile(r"^E2E_TIME .*?\sl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def samples(raw, model, arm, pattern):
    out = {}
    glob_pattern = os.path.join(raw, "final", "%s_%s" % (model, arm), "r*",
                                "run_*.log")
    for path in sorted(glob.glob(glob_pattern)):
        found_round = ROUND.search(path)
        if not found_round:
            continue
        with open(path) as handle:
            for line in handle:
                found = pattern.match(line)
                if found:
                    out[int(found_round.group(1))] = float(found.group(1))
    return out


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def signed_rank_p(deltas):
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n < 6:
        return float("nan")
    ordered = sorted(nonzero, key=abs)
    ranks, i, ties = [0.0] * n, 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(ordered[j + 1]) == abs(ordered[i]):
            j += 1
        shared = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = shared
        run = j - i + 1
        ties += run ** 3 - run
        i = j + 1
    w = sum(r for r, d in zip(ranks, ordered) if d > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0
    if var <= 0:
        return 1.0
    z = (abs(w - mu) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2.0))


def paired(base, other, draws=20000, seed=20260903):
    rounds = sorted(set(base) & set(other))
    deltas = [100.0 * (other[r] / base[r] - 1.0) for r in rounds]
    if not deltas:
        return None
    rng = random.Random(seed)
    boot = sorted(median([deltas[rng.randrange(len(deltas))]
                          for _ in range(len(deltas))]) for _ in range(draws))
    return (median(deltas), boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1],
            signed_rank_p(deltas), len(deltas))


def report(title, data, note_for):
    base = data["stage"]
    if not base:
        return
    print("   %s" % title)
    print("   %-8s %-10s %-27s %-13s %s"
          % ("arm", "median", "vs stage (95% CI)", "signed-rank p", "note"))
    for arm in ARMS:
        values = data.get(arm)
        if not values:
            continue
        if arm == "stage":
            print("   %-8s %.6f  %-27s %-13s %s"
                  % (arm, median(list(values.values())), "--", "--", ""))
            continue
        stats = paired(base, values)
        if stats is None:
            continue
        point, low, high, p, n = stats
        print("   %-8s %.6f  %+7.3f%% [%+6.3f,%+6.3f]      %10.2e    %s"
              % (arm, median(list(values.values())), point, low, high, p,
                 note_for(arm)))
    print()


def main():
    raw = sys.argv[1]
    for model in ("gqa2", "mha4"):
        l2 = {arm: samples(raw, model, arm, L2) for arm in ARMS}
        l1 = {arm: samples(raw, model, arm, L1) for arm in ARMS}
        if not l2["stage"]:
            continue
        print("== %s   %d interleaved rounds, one fresh process per arm per round"
              % (model, len(l2["stage"])))
        report("l2_ms -- the kappa curve", l2,
               lambda a: "NULL CONTROL: kappa does not touch L2 here"
               if a == "nosync" else "")
        report("l1_ms -- untouched by kappa", l1,
               lambda a: "CEILING: grid barrier deleted (output wrong)"
               if a == "nosync" else "NULL CONTROL")


if __name__ == "__main__":
    main()
