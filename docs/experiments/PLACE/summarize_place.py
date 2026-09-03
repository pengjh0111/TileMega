#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Paired report for the placement sweep.

Same protocol as COARSEN/summarize_kappa.py and for the same reason (F-46):
the effect being looked for is of the same size as the session drift, so the
statistic is the within-round ratio and never a difference of two medians.

The arms are read as a two-sided bound rather than a search:

  ident2   a second binary compiled exactly like `ident`.  Its measured
           "effect" is this experiment's own noise floor; no other arm can be
           believed below it.
  pair     the only permutation the affinity measurement argues for -- CTAs
           co-resident on one SM take consecutive task indices, and
           consecutive GEMM tasks differ in N and share the whole A panel.
  reverse  a bijection that keeps adjacency and only relabels.
  scatter  multiplication by a unit of Z/g, which destroys index adjacency
           entirely.  If even this costs nothing, no placement can gain
           anything: the objective is flat over the whole permutation group,
           not merely near the identity.
"""
import glob
import math
import os
import random
import re
import sys

ARMS = ("ident", "ident2", "pair", "reverse", "scatter")
L05 = re.compile(r"^E2E_TIME .*?\bl05_ms=([0-9.]+)")
L1 = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+)")
L2 = re.compile(r"^E2E_TIME .*?\sl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def samples(raw, model, arm, pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(
            raw, "final", "%s_%s" % (model, arm), "r*", "run_*.log"))):
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


def report(title, data):
    base = data.get("ident")
    if not base:
        return None
    print("   %s" % title)
    print("   %-8s %-10s %-27s %-13s %s"
          % ("arm", "median", "vs ident (95% CI)", "signed-rank p", "note"))
    notes = {"ident2": "NOISE FLOOR: identical binary",
             "pair": "the affinity heuristic",
             "scatter": "adjacency destroyed"}
    best = None
    for arm in ARMS:
        values = data.get(arm)
        if not values:
            continue
        if arm == "ident":
            print("   %-8s %.6f  %-27s %-13s %s"
                  % (arm, median(list(values.values())), "--", "--", "baseline"))
            continue
        stats = paired(base, values)
        if stats is None:
            continue
        point, low, high, p, _ = stats
        if arm != "ident2" and (best is None or point < best[1]):
            best = (arm, point)
        print("   %-8s %.6f  %+7.3f%% [%+6.3f,%+6.3f]      %10.2e    %s"
              % (arm, median(list(values.values())), point, low, high, p,
                 notes.get(arm, "")))
    print()
    return best


def main():
    raw = sys.argv[1]
    for model in ("gqa2", "mha4"):
        series = {name: {arm: samples(raw, model, arm, pattern)
                         for arm in ARMS}
                  for name, pattern in (("l05_ms", L05), ("l1_ms", L1),
                                        ("l2_ms", L2))}
        if not series["l2_ms"]["ident"]:
            continue
        print("== %s   %d interleaved rounds, one fresh process per arm per round"
              % (model, len(series["l2_ms"]["ident"])))
        floor = paired(series["l2_ms"]["ident"], series["l2_ms"]["ident2"])
        best = {}
        for name in ("l05_ms", "l1_ms", "l2_ms"):
            best[name] = report("%s -- %s" % (name, {
                "l05_ms": "one kernel per stage",
                "l1_ms": "persistent grid, barrier per stage",
                "l2_ms": "persistent grid, event DAG"}[name]), series[name])
        # The oracle over this family is the best arm, and it is only a result
        # at all if it clears the noise floor the identical binary measured.
        for name in ("l05_ms", "l1_ms", "l2_ms"):
            if not best.get(name):
                continue
            arm, point = best[name]
            print("   ORACLE model=%s metric=%s arm=%s gain=%+.3f%% floor=%+.3f%%"
                  % (model, name, arm, -point,
                     abs(floor[0]) if floor else float("nan")))
        print()


if __name__ == "__main__":
    main()
