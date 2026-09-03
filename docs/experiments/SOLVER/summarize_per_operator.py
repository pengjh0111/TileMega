#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Turn run_per_operator.sh's fresh-process logs into the (b) answer.

A 1 % end-to-end delta is smaller than the run-to-run spread §0.3 documents, so
a median on its own does not decide anything.  Worse, an earlier block-per-arm
protocol made two byte-identical mha4 binaries separate by 0.64 % at rank-sum
p = 1.8e-4: within-session drift, not sampling noise, dominated the contrast.
The arms are therefore measured interleaved, one fresh process per arm per
round, and reported *paired* -- the statistic is the within-round ratio, so any
drift shared by a round cancels instead of loading onto whichever arm ran late.
Arms whose compiled plan is identical are flagged as null controls: whatever
separation they show is the protocol's own floor, and no claim below that floor
is a claim.
"""
import glob
import math
import os
import random
import re
import sys

ARMS = ("uniform", "split_only", "per_op", "best_uniform")
TIME = re.compile(r"^E2E_TIME .*?\bl05_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")


def samples(raw, model, arm):
    """l05 times keyed by measurement round, so arms can be paired."""
    out = {}
    pattern = os.path.join(raw, "final", "%s_%s" % (model, arm), "r*", "run_*.log")
    for path in sorted(glob.glob(pattern)):
        found_round = ROUND.search(path)
        if not found_round:
            continue
        with open(path) as handle:
            for line in handle:
                found = TIME.match(line)
                if found:
                    out[int(found_round.group(1))] = float(found.group(1))
    return out


DEFINE = re.compile(r"#define\s+(TILEMEGA_GEMM_\S+)\s+(.*)")
ARRAY = re.compile(r"static constexpr int (kTileMega\w+)\[\] = \{([^}]*)\}")


def plan_digest(raw, model, arm):
    """What the plan *means*, not how it was spelled.

    The uniform arm is emitted by the plan generator and best_uniform by a fixed
    template, so two arms can compile the same kernel from different text.  A
    text diff would miss exactly the null controls this report exists to expose.
    """
    path = os.path.join(raw, "plan_%s_%s.h" % (model, arm))
    if not os.path.exists(path):
        return None
    plan = {"TILEMEGA_GEMM_VARIANT_COUNT": "1"}
    with open(path) as handle:
        for line in handle:
            found = DEFINE.match(line.strip())
            if found and "(i)" not in found.group(1):
                plan[found.group(1)] = found.group(2).strip()
            found = ARRAY.search(line)
            if found:
                plan[found.group(1)] = re.sub(r"\s+", "", found.group(2))
    return repr(sorted(plan.items()))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def signed_rank_p(deltas):
    """Two-sided Wilcoxon signed-rank, normal approximation with tie correction."""
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
    """Median within-round percent difference, with a bootstrap over rounds."""
    rounds = sorted(set(base) & set(other))
    deltas = [100.0 * (other[r] / base[r] - 1.0) for r in rounds]
    if not deltas:
        return None
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        boot.append(median([deltas[rng.randrange(len(deltas))]
                            for _ in range(len(deltas))]))
    boot.sort()
    return (median(deltas), boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1],
            signed_rank_p(deltas), len(deltas))


def report(model, data, digests, base_arm, label):
    base = data[base_arm]
    print("   %-13s %-9s %-10s %-26s %s"
          % ("arm", "median", "min..max", "vs %s (95%% CI)" % label, "signed-rank p"))
    for arm in ARMS:
        values = data[arm]
        if not values:
            continue
        series = list(values.values())
        note = ""
        if digests[arm] is not None and arm != base_arm and \
                digests[arm] == digests[base_arm]:
            note = "   <- NULL CONTROL: identical compiled plan"
        if arm == base_arm:
            print("   %-13s %.6f  %.6f..%.6f  %-26s %s"
                  % (arm, median(series), min(series), max(series), "--", "--"))
            continue
        stats = paired(base, values)
        if stats is None:
            continue
        point, low, high, p, n = stats
        print("   %-13s %.6f  %.6f..%.6f  %+7.3f%% [%+6.3f,%+6.3f]  %10.2e%s"
              % (arm, median(series), min(series), max(series), point, low, high,
                 p, note))


def main():
    raw = sys.argv[1]
    for model in ("gqa2", "mha4"):
        data = {arm: samples(raw, model, arm) for arm in ARMS}
        digests = {arm: plan_digest(raw, model, arm) for arm in ARMS}
        if not data["uniform"]:
            continue
        print("== %s   l05, interleaved rounds, one fresh process per arm per round,"
              " %d rounds" % (model, len(data["uniform"])))
        report(model, data, digests, "uniform", "DP uniform")
        if data["best_uniform"] and data["per_op"]:
            print()
            report(model, data, digests, "best_uniform", "oracle best uniform")
        nulls = [(a, b) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]
                 if digests[a] is not None and digests[a] == digests[b]
                 and data[a] and data[b]]
        if nulls:
            print()
            print("   null controls -- arms below compile the same kernel, so their"
                  " separation is the")
            print("   protocol's own floor and no smaller claim is a claim:")
            for a, b in nulls:
                point, low, high, p, n = paired(data[a], data[b])
                print("   %-13s vs %-13s %+7.3f%% [%+6.3f,%+6.3f]  p = %.2e"
                      % (a, b, point, low, high, p))
        print()


if __name__ == "__main__":
    main()
