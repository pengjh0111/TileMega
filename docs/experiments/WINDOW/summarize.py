#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""The four EX-E2 gates, each from the raw logs of `run.sh` and nothing else.

E2-a and E2-c are counts of processes, so no statistic applies to them.  E2-b is
a measurement and is formed the way S2c-d is -- a ratio taken inside a round,
against the arm run in the same session, never a median across a session
boundary -- so the two gates' intervals mean the same thing.

E2-d is not recomputed here.  `hol_reclaimable_ns` and `cp_lb_nosync_ms` come
from TRACE_V2/analyze.py, which is the tool that measured F-134; a second
definition of head-of-line time would not be comparable to the number this round
has to move (H7).  This reads its `analysis.tsv` and relates it to makespan.
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
CELL = re.compile(r"^trace_w([0-9]+)_([a-z0-9]+)_s([0-9]+)$")
# Same seed and draw count as CHAIN/summarize.py so a W interval and a chain
# interval are produced by the same procedure, not merely the same formula.
SEED, DRAWS = 20260906, 20000
# E2-b passes when the interval covers an unchanged ratio, or when the effect is
# smaller than this either way: W = 1 is meant to *be* today's executor, so the
# claim is equivalence, and an equivalence claim needs a band.
EQUIVALENCE = 0.005


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


def bootstrap(values, draws=DRAWS):
    rng = random.Random(SEED)
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


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    if not rows:
        return []
    header = rows[0]
    return [dict(zip(header, row)) for row in rows[1:]]


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def gate_a(raw):
    """E2-a: every W runs the whole SEQSCAN matrix, every process passing."""
    print("gate\tw\tcells\tpasses\tprocesses\tverdict")
    rows = read_tsv(os.path.join(raw, "matrix.tsv"))
    windows = sorted({int(row["w"]) for row in rows})
    for w in windows:
        cells = [row for row in rows if int(row["w"]) == w]
        passes = sum(int(row["passes"]) for row in cells)
        total = sum(int(row["processes"]) for row in cells)
        verdict = "PASS" if cells and passes == total else "FAIL"
        print(f"E2-a\t{w}\t{len(cells)}\t{passes}\t{total}\t{verdict}")
    if not windows:
        print("E2-a\t-\t0\t0\t0\tFAIL")


def gate_b(raw, runs_expected):
    """E2-b: W = 1 against the executor as it stands, paired and rotated."""
    print()
    print("gate\tmodel\tseq\tn\ttoday_median_ms\tw1_median_ms"
          "\tratio\tci_lo\tci_hi\tp_wilcoxon\tverdict")
    pooled = []
    for model in ("gqa2", "mha4"):
        for seq in (4, 128):
            def cell(arm):
                return samples(os.path.join(raw, "final", f"{model}_s{seq}_{arm}",
                                            "r*", "run_*.log"))
            base, arm_s = cell("today"), cell("w1")
            rounds = sorted(set(base) & set(arm_s))
            if not rounds:
                print(f"E2-b\t{model}\t{seq}\t0\t-\t-\t-\t-\t-\t-\tFAIL")
                continue
            ratios = [arm_s[r][1] / base[r][1] for r in rounds]
            low, high = bootstrap(ratios)
            ratio = statistics.median(ratios)
            ok = (low <= 1. <= high or abs(ratio - 1.) < EQUIVALENCE) \
                and len(rounds) >= runs_expected
            print(f"E2-b\t{model}\t{seq}\t{len(rounds)}"
                  f"\t{statistics.median(base[r][1] for r in rounds):.6f}"
                  f"\t{statistics.median(arm_s[r][1] for r in rounds):.6f}"
                  f"\t{ratio:.4f}\t{low:.4f}\t{high:.4f}"
                  f"\t{signed_rank([x - 1 for x in ratios]):.3e}"
                  f"\t{'PASS' if ok else 'FAIL'}")
            pooled.extend(ratios)
    if pooled:
        low, high = bootstrap(pooled)
        ratio = statistics.median(pooled)
        ok = low <= 1. <= high or abs(ratio - 1.) < EQUIVALENCE
        print(f"E2-b\tALL\tALL\t{len(pooled)}\t-\t-\t{ratio:.4f}\t{low:.4f}"
              f"\t{high:.4f}\t{signed_rank([x - 1 for x in pooled]):.3e}"
              f"\t{'PASS' if ok else 'FAIL'}")


def gate_c(raw):
    """E2-c: the W = 1 lifting rules under a W = 2 executor have to fail.

    A clean sweep here is §10's stop condition.  It is never evidence that the
    rule can be dropped -- it means the reference models do not reach the
    reordering the rule exists to make safe, and H4 then requires a graph that
    does.
    """
    print()
    print("gate\tmodel\tseq\tpasses\tprocesses\tfailure_rate\tverdict")
    rows = read_tsv(os.path.join(raw, "negative.tsv"))
    for row in rows:
        passes, total = int(row["passes"]), int(row["processes"])
        rate = (total - passes) / total if total else 0.
        print(f"E2-c\t{row['model']}\t{row['seq']}\t{passes}\t{total}"
              f"\t{rate:.4f}\t{'PASS' if total and passes != total else 'FAIL'}")
    if not rows:
        print("E2-c\t-\t-\t0\t0\t0.0000\tFAIL")


def gate_d(raw):
    """E2-d: reclaimed head-of-line time against the makespan it bought.

    These rows come from traced builds, which carry the trace's own cost, so a
    column here is not comparable with E2-b's untraced milliseconds; the
    comparison that matters is across W within this table.  `cp_lb_nosync_ms`
    is each configuration's own ceiling, recomputed from its own trace (H8).
    """
    print()
    print("gate\tmodel\tseq\tw\thol_reclaimable_ns\thol_fraction_of_kernel"
          "\thol_workers_nonzero\tmeasured_l2_ms\tcp_lb_nosync_ms"
          "\tmeasured_over_ceiling\thol_delta_vs_w1_ns\tl2_delta_vs_w1_ms")
    cells = {}
    for row in read_tsv(os.path.join(raw, "analysis", "analysis.tsv")):
        match = CELL.match(row.get("cell", ""))
        if match:
            w, model, seq = int(match.group(1)), match.group(2), int(match.group(3))
            cells[(model, seq, w)] = row
    for model, seq in sorted({(m, s) for m, s, _ in cells}):
        base = cells.get((model, seq, 1))
        for w in sorted(w for m, s, w in cells if (m, s) == (model, seq)):
            row = cells[(model, seq, w)]
            hol, l2 = number(row, "hol_reclaimable_ns"), number(row, "measured_l2_ms")
            ceiling = number(row, "cp_lb_nosync_ms")
            hol_delta = (number(base, "hol_reclaimable_ns") - hol) if base else float("nan")
            l2_delta = (l2 - number(base, "measured_l2_ms")) if base else float("nan")
            print(f"E2-d\t{model}\t{seq}\t{w}\t{hol:.1f}"
                  f"\t{number(row, 'hol_fraction_of_kernel'):.4f}"
                  f"\t{row.get('hol_workers_nonzero', '')}\t{l2:.6f}\t{ceiling:.6f}"
                  f"\t{(l2 / ceiling if ceiling else float('nan')):.4f}"
                  f"\t{hol_delta:.1f}\t{l2_delta:+.6f}")
    if not cells:
        print("E2-d\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-")


def main():
    raw = sys.argv[1]
    runs_expected = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    gate_a(raw)
    gate_b(raw, runs_expected)
    gate_c(raw)
    gate_d(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
