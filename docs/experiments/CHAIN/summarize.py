#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""S2c-b and S2c-d: what placing the critical chain on one worker is worth.

Two questions, kept apart because they are answered by different evidence.
S2c-b is a property of the plans and is read out of `predicted.tsv`, which the
emitter computed offline -- no GPU is involved and no statistics apply to it.
S2c-d is a measurement, so it is a paired ratio against `rotate` formed inside
a round, never a median taken across a session boundary.

`rotate` is the denominator because §1.2 records it as the best candidate under
configuration A, so it is the arm chaining has to beat to mean anything.
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
ARMS = ("legacy_grid_stride", "rotate", "chain")
# Seed and draw count are fixed so a rerun of this script on the same logs
# prints the same interval; they are not tuned per cell.
SEED, DRAWS = 20260906, 20000


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


def main():
    raw = sys.argv[1]

    # --- S2c-b: the plan metrics, straight from the emitter.  Reported for
    # every candidate so the expected ordering -- rotate's critical_path_hops
    # largest, chain's smallest -- can be read off rather than asserted.
    print("gate\tmodel\tseq\tcandidate\tstatus\tcritical_path_hops"
          "\tcritical_path_same_worker_edges\tcritical_path_queue_edges"
          "\tspine_length_ns\tmax_queue_ns\tmakespan_ns\tcross_worker_edges")
    for row in read_tsv(os.path.join(raw, "predicted.tsv")):
        print(f"S2c-b\t{row['model']}\t{row['seq']}\t{row['candidate']}\t{row['status']}"
              f"\t{row.get('critical_path_hops', '')}"
              f"\t{row.get('critical_path_same_worker_edges', '')}"
              f"\t{row.get('critical_path_queue_edges', '')}"
              f"\t{row.get('spine_length_ns', '')}\t{row.get('max_queue_ns', '')}"
              f"\t{row.get('makespan_ns', '')}\t{row.get('cross_worker_edges', '')}")

    # --- §6.2: the same schedule priced both ways.  Printed rather than
    # averaged: a traced weight for a stage whose tasks measure one globaltimer
    # tick is quantised at the clock, not more precise than the cost model.
    print()
    print("gate\tmodel\tseq\tsource\tnodes\tobserved\tspine_length_ns"
          "\tmax_queue_ns\tchain_count\tchain_interleaves\tsplit_count"
          "\tmakespan_ns\tfill_overflows\tworker_diff")
    for row in read_tsv(os.path.join(raw, "chain_weights.tsv")):
        print(f"S2c-w\t{row['model']}\t{row['seq']}\t{row['source']}\t{row['nodes']}"
              f"\t{row['observed']}\t{row['spine_length_ns']}\t{row['max_queue_ns']}"
              f"\t{row['chain_count']}\t{row['chain_interleaves']}"
              f"\t{row['split_count']}\t{row['makespan_ns']}"
              f"\t{row.get('fill_overflows', '')}"
              f"\t{row['worker_diff']}")

    # --- S2c-c: how often a chain had to be split so the contracted graph
    # stays acyclic.  It is always zero, and provably so: every queue is
    # ordered by topological index, so the union of task and queue edges is a
    # subset of one topological order and no chain placement can close an L-a
    # cycle.  The unit test carries the constructed case -- two chains with
    # cross edges both ways -- which closes a cycle in the *contracted* graph
    # and still needs no split, plus a task-DAG cycle, which is refused.
    print()
    print("gate\tmodel\tseq\tchain_count\tchain_interleaves\tsplit_count"
          "\tfill_overflows")
    for row in read_tsv(os.path.join(raw, "chain_weights.tsv")):
        if row["source"] == "cost_model":
            print(f"S2c-c\t{row['model']}\t{row['seq']}\t{row['chain_count']}"
                  f"\t{row['chain_interleaves']}\t{row['split_count']}"
                  f"\t{row.get('fill_overflows', '')}")

    # --- S2c-d: the measurement.
    print()
    print("gate\tmodel\tseq\tarm\tn\trotate_median_ms\tarm_median_ms"
          "\tratio\tci_lo\tci_hi\tp_wilcoxon")
    pooled = {arm: [] for arm in ARMS}
    cells = [(m, s) for s in (4, 128) for m in ("gqa2", "mha4")]
    cells += [("real", s) for s in (4, 128)]
    for model, seq in cells:
        def cell(arm):
            return samples(os.path.join(raw, "final", f"{model}_s{seq}_{arm}",
                                        "r*", "run_*.log"))
        base = cell("rotate")
        if not base:
            continue
        for arm in ARMS:
            arm_s = cell(arm)
            rounds = sorted(set(base) & set(arm_s))
            if not rounds:
                continue
            ratios = [arm_s[r][1] / base[r][1] for r in rounds]
            low, high = bootstrap(ratios)
            print(f"S2c-d\t{model}\t{seq}\t{arm}\t{len(rounds)}"
                  f"\t{statistics.median(base[r][1] for r in rounds):.6f}"
                  f"\t{statistics.median(arm_s[r][1] for r in rounds):.6f}"
                  f"\t{statistics.median(ratios):.4f}\t{low:.4f}\t{high:.4f}"
                  f"\t{signed_rank([x - 1 for x in ratios]):.3e}")
            if model != "real":
                pooled[arm].extend(ratios)
    for arm in ARMS:
        if not pooled[arm]:
            continue
        low, high = bootstrap(pooled[arm])
        print(f"S2c-d\tALL\tALL\t{arm}\t{len(pooled[arm])}\t-\t-"
              f"\t{statistics.median(pooled[arm]):.4f}\t{low:.4f}\t{high:.4f}"
              f"\t{signed_rank([x - 1 for x in pooled[arm]]):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
