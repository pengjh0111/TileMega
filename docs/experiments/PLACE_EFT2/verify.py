#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Every R3 §9 gate, recomputed from raw logs and dumps.

Nothing here reads a summary another script concluded from: the paired ratios
are re-derived from the `E2E_TIME` lines of individual runs, the pass counts
from `RESULT status=` lines, the barrier counts from the SASS census, and the
SASS identity from the dumps themselves.  A gate whose evidence has not been
produced yet is a FAIL that names the missing path, never a skip -- an absent
directory is the one failure mode a verifier must not be quiet about.

Two definitions are deliberately imported rather than restated (H7).  The
head-of-line and ceiling columns come from TRACE_V2/analyze.py's `analysis.tsv`,
because `hol_reclaimable_ns` and `cp_lb_nosync_ms` are the quantities F-134 and
§1.1 measured and a second definition of either would not be comparable; and
the bootstrap seed and draw count match CHAIN/summarize.py and WINDOW/
summarize.py so that an interval here means what an interval there means.

    python3 verify.py [repo_root]

Exit status is non-zero if any hard gate fails.  The research gate S2r-b is
reported with the same arithmetic but does not set the exit status: §0 item 3
makes a missed research gate something to localize and act on, not a build
break.
"""
import glob
import hashlib
import os
import random
import re
import statistics
import subprocess
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")
ROUND = re.compile(r"/r([0-9]+)/")
SEED, DRAWS = 20260906, 20000

# §7.3, the round's fixed reference, in micro-seconds: ceiling_A, measured_A,
# target.  H8 has every configuration recompute its own ceiling from its own
# trace, but the *gate* is anchored here and is never recomputed from whichever
# configuration happens to flatter it (H7).
TARGETS = {
    ("gqa2", 4): (143., 293., 218.),
    ("gqa2", 128): (197., 458., 328.),
    ("mha4", 4): (144., 579., 362.),
    ("mha4", 128): (199., 856., 528.),
}

# The executor's per-task barrier count as built.  Skeleton §5.5.1 says at most
# five `__syncthreads()` per task; the R3 prompt §1.2 says six.  SASS is the
# authority and the difference is recorded rather than reconciled: the census
# below measures the *drop*, which is the same under either baseline, and the
# per-task figure is reported against the skeleton's five.
BARRIERS_PER_TASK_BASE = 5
BARRIERS_PER_TASK_GATE = 2

HARD, RESEARCH, REPORT = "hard", "research", "report"
results = []


def record(gate, kind, ok, detail, evidence):
    results.append((gate, kind, ok, detail, evidence))


def missing(gate, kind, path, note=""):
    record(gate, kind, False, "missing evidence" + (": " + note if note else ""), path)


def read_tsv(path):
    """Rows of a headed TSV as dicts; [] when the file is absent or empty."""
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle
                if line.strip() and not line.startswith("#")]
    if len(rows) < 2:
        return []
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def samples(pattern):
    """round -> (l1_ms, l2_ms), read out of the run logs themselves."""
    result = {}
    for path in glob.glob(pattern):
        match = ROUND.search(path)
        if not match:
            continue
        for line in open(path, errors="replace"):
            found = TIME.match(line)
            if found:
                result[int(match.group(1))] = (float(found.group(1)),
                                               float(found.group(2)))
    return result


def bootstrap(values, draws=DRAWS):
    rng = random.Random(SEED)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    estimates = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def counts(rows, pass_key="passes", total_key="processes"):
    """(cells, passes, processes, short rows) for a pass-count table."""
    passes = sum(int(row[pass_key]) for row in rows)
    total = sum(int(row[total_key]) for row in rows)
    short = [row for row in rows if int(row[pass_key]) != int(row[total_key])]
    return len(rows), passes, total, short


def label(row):
    return "_".join(str(row.get(key)) for key in ("model", "seq", "cell", "arm")
                    if row.get(key) is not None)


# --- H2: with every switch off, the build is the baseline ---------------------
def gate_h2(sync, repo):
    root = os.path.join(sync, "sass_identity")
    meta = os.path.join(root, "meta.tsv")
    if not os.path.exists(meta):
        return missing("H2-sass-identity", HARD, meta)
    fields = dict(line.rstrip("\n").split("\t", 1)
                  for line in open(meta) if "\t" in line)
    verdicts, detail = [], []
    for model in ("gqa2", "mha4"):
        base = os.path.join(root, model + "_base.sass")
        head = os.path.join(root, model + "_head.sass")
        if not (os.path.exists(base) and os.path.exists(head)):
            verdicts.append(False)
            detail.append(model + ": dump missing")
            continue
        # Recomputed from the dumps rather than trusting the stored `.diff`,
        # which is an artifact of the run and not evidence by itself.
        same = digest(base) == digest(head)
        lines = sum(1 for _ in open(base, errors="replace"))
        verdicts.append(same)
        detail.append("%s: identical=%d lines=%d" % (model, int(same), lines))
    stamped = fields.get("head_commit", "")
    current = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             check=False, capture_output=True,
                             text=True).stdout.strip()
    # The gate printed these two and compared neither, so identity dumps taken
    # at an older commit certified today's tree: a violation introduced after
    # the dump would have passed silently.
    fresh = bool(current) and stamped == current
    verdicts.append(fresh)
    detail.append("base=%s head=%s%s"
                  % (fields.get("base_commit", "?")[:8], stamped[:8] or "?",
                     "" if fresh else " STALE, HEAD is %s" % (current[:8] or "?")))
    record("H2-sass-identity", HARD, all(verdicts), "; ".join(detail), root)


# --- the shape shared by every E3 correctness gate ----------------------------
def correctness_gate(name, raw, need_cells=4, need_matrix=30):
    correctness = os.path.join(raw, "correctness.tsv")
    rows = read_tsv(correctness)
    if not rows:
        return missing(name, HARD, correctness, "no correctness table")
    cells, passes, total, short = counts(rows)
    matrix = read_tsv(os.path.join(raw, "matrix.tsv"))
    _, m_pass, m_total, m_short = counts(matrix) if matrix else (0, 0, 0, [])
    # Coverage by equality, not truthiness.  The matrix is 2 models x 5 seq x
    # 3 past; a run still in flight or killed leaves a short table whose every
    # present row is 50/50, and `bool(matrix)` read that as a pass -- E3-3
    # reported 29 cells as PASS while its seqscan was still running.
    ok = (cells >= need_cells and not short and len(matrix) == need_matrix
          and not m_short)
    detail = "cells=%d %d/%d; seqscan=%d cells %d/%d" % (
        cells, passes, total, len(matrix), m_pass, m_total)
    if len(matrix) != need_matrix:
        detail += "; seqscan covers %d of %d cells" % (len(matrix), need_matrix)
    if short:
        detail += "; short: " + ",".join(label(row) for row in short)
    if m_short:
        detail += "; seqscan short: " + ",".join(label(row) for row in m_short)
    record(name, HARD, ok, detail, raw)


def census_delta(name, raw, expect_drop):
    """The BAR.SYNC drop in the L2 kernel, with the fence/backoff controls."""
    rows = [row for row in read_tsv(os.path.join(raw, "census.tsv"))
            if "tilemega_l2_kernel" in row.get("kernel", "")
            and row.get("variant") == "plain"]
    if not rows:
        return missing(name, HARD, os.path.join(raw, "census.tsv"), "no census")
    detail, ok = [], True
    for model in sorted({row["model"] for row in rows}):
        off = next((r for r in rows if r["model"] == model and r["v2"] == "0"), None)
        on = next((r for r in rows if r["model"] == model and r["v2"] == "1"), None)
        if not (off and on):
            ok = False
            detail.append(model + ": arm missing")
            continue
        drop = int(off["bar_sync"]) - int(on["bar_sync"])
        per_task = BARRIERS_PER_TASK_BASE - drop
        # The controls: the switch must move BAR.SYNC and nothing else.  A fence
        # or a backoff that moved with it would mean the edit reached further
        # than the barriers it claims to drop.
        moved = [key for key in ("membar", "nanosleep")
                 if key in off and off[key] != on[key]]
        good = drop >= expect_drop and per_task <= BARRIERS_PER_TASK_GATE and not moved
        ok = ok and good
        detail.append("%s: %s->%s drop=%d per_task=%d%s" % (
            model, off["bar_sync"], on["bar_sync"], drop, per_task,
            (" moved=" + ",".join(moved)) if moved else ""))
    record(name, HARD, ok, "; ".join(detail), os.path.join(raw, "census.tsv"))


# --- E3-0: the calibrated wait policy -----------------------------------------
def gate_e3_0(sync, repo):
    curve = os.path.join(sync, "backoff.tsv")
    policy = os.path.join(sync, "backoff_policy.tsv")
    spin = os.path.join(sync, "spin_interference.tsv")
    rows = read_tsv(curve)
    interference = read_tsv(spin)
    # §5's two mandatory points on the curve: pure spin, and the status quo.
    pure_spin = any(row.get("backoff_ns") == "0" for row in rows)
    status_quo = any((row.get("spin_iters"), row.get("backoff_ns")) == ("0", "64")
                     for row in rows)
    compute_modes = {row.get("compute_mode") for row in interference}
    # H5: the values come from the target, never from this script.
    tool = os.path.join(repo, "build-portable", "tools", "tilemega-wait-policy")
    flags = ""
    if os.path.exists(tool):
        flags = subprocess.run([tool, "sm_89", "bf16", repo], check=False,
                               capture_output=True, text=True).stdout.strip()
    ok = (bool(rows) and pure_spin and status_quo and len(compute_modes) >= 2
          and "TILEMEGA_WAIT_SPIN_ITERS" in flags
          and os.path.exists(policy))
    record("E3-0-calibration", HARD, ok,
           "curve_points=%d pure_spin=%d status_quo=%d spin_compute_modes=%d; policy=%s"
           % (len(rows), int(pure_spin), int(status_quo), len(compute_modes),
              flags or "tool absent"), policy)
    correctness_gate("E3-0-correctness", os.path.join(sync, "raw_wait"))


# --- E3-4: the litmus ---------------------------------------------------------
def gate_e3_4(sync):
    path = os.path.join(sync, "raw_litmus", "litmus.tsv")
    rows = read_tsv(path)
    if not rows:
        return missing("E3-4-litmus", HARD, path)
    if "acquire" not in rows[0]:
        record("E3-4-litmus", HARD, False,
               "litmus.tsv predates the acquire axis: every arm ran the"
               " consumer fence, so no cell could detect a missing release",
               path)
        return
    arms = {}
    for row in rows:
        arms.setdefault(row["release"], []).append(row)
    # grid in {64,128,256} x tile in {1024,4096,16384} x acquire in {1,0}
    expected = 18
    complete = all(len(arms.get(arm, [])) == expected for arm in
                   ("per_writer", "thread0_fence", "no_barrier", "no_fence"))
    runs_ok = all(int(row["runs"]) >= 50 for row in rows)
    # Soundness is asserted only where the protocol is fully present.  With the
    # acquire fence dropped the protocol under test is incomplete, so
    # per_writer failing there is a measurement about that fence rather than an
    # unsound harness.
    sound = all(int(row["pass"]) == int(row["runs"])
                for row in arms.get("per_writer", []) if row["acquire"] == "1")
    key = lambda r: (r["acquire"], r["grid"], r["tile"])
    # A cell is evidence only where the detector is awake (`no_fence`
    # mismatches: it cannot see a missing release otherwise, F-3/F-10) and the
    # reference protocol still holds there (`per_writer` passes).
    awake = {key(r) for r in arms.get("no_fence", []) if int(r["mismatch"]) > 0}
    holds = {key(r) for r in arms.get("per_writer", [])
             if int(r["pass"]) == int(r["runs"])}
    readable_keys = awake & holds
    negative = [r for r in arms.get("no_barrier", [])
                if key(r) in readable_keys and int(r["mismatch"]) > 0]
    ok = (complete and runs_ok and sound and bool(readable_keys)
          and bool(negative))
    record("E3-4-litmus", HARD, ok,
           "cells=%d/72 runs>=50=%d per_writer_sound(acquire=1)=%d;"
           " no_fence awake in %d cells; per_writer holds in %d;"
           " readable=%d; no_barrier failed in %d of those" % (
               len(rows), int(runs_ok), int(sound), len(awake), len(holds),
               len(readable_keys), len(negative)), path)
    candidate = arms.get("thread0_fence", [])
    readable = [r for r in candidate if key(r) in readable_keys]
    held = [r for r in readable if int(r["pass"]) == int(r["runs"])]
    detail = "readable cells=%d; thread0_fence held in %d of them" % (
        len(readable), len(held))
    if not readable:
        detail += "; NO readable cell: the run says nothing about the candidate"
    elif len(held) == len(readable):
        detail += "; the candidate is indistinguishable from per_writer where"
        detail += " the harness can tell them apart"
    else:
        detail += "; the candidate FAILED where per_writer holds"
    record("E3-4-conclusion", REPORT, True, detail, path)


# --- EX-S2c: critical-path chaining -------------------------------------------
def gate_s2c(chain, repo):
    correctness = os.path.join(chain, "correctness.tsv")
    rows = read_tsv(correctness)
    if not rows:
        missing("S2c-a-correctness", HARD, correctness)
    else:
        cells, passes, total, short = counts(rows)
        detail = "arm-cells=%d %d/%d" % (cells, passes, total)
        if short:
            detail += "; short: " + ",".join(label(row) for row in short)
        record("S2c-a-correctness", HARD, not short, detail, correctness)

    predicted = os.path.join(chain, "predicted.tsv")
    rows = read_tsv(predicted)
    if not rows:
        missing("S2c-b-hops", HARD, predicted)
    else:
        wins, detail = [], []
        for model in ("gqa2", "mha4"):
            for seq in ("4", "128"):
                cell = [r for r in rows if r["model"] == model and r["seq"] == seq]
                hops = {r["candidate"]: int(r["critical_path_hops"])
                        for r in cell if r.get("critical_path_hops")}
                if "chain" not in hops or "rotate" not in hops:
                    continue
                best = min(hops, key=lambda name: hops[name])
                wins.append(hops["chain"] < hops["rotate"])
                detail.append("%s_s%s: chain=%d rotate=%d best=%s(%d) candidates=%d"
                              % (model, seq, hops["chain"], hops["rotate"], best,
                                 hops[best], len(hops)))
        record("S2c-b-hops", HARD, bool(wins) and all(wins),
               "; ".join(detail) or "no comparable cells", predicted)

    weights = os.path.join(chain, "chain_weights.tsv")
    rows = [r for r in read_tsv(weights) if r.get("source") == "cost_model"]
    test = os.path.join(repo, "test", "unit", "chain_placement_test.cpp")
    has_case = (os.path.exists(test)
                and "cycle" in open(test, errors="replace").read().lower())
    if not rows:
        missing("S2c-c-legality", HARD, weights)
    else:
        splits = sum(int(row["split_count"]) for row in rows)
        record("S2c-c-legality", HARD, has_case,
               "split_count=%d over %d cells; constructed cycle case present=%d."
               " Ordering each queue by topological index makes the union of"
               " task and queue edges a subset of one topological order, so"
               " split_count is provably 0 rather than merely observed 0"
               % (splits, len(rows), int(has_case)),
               weights + "; " + test)


# --- EX-E2: the slot window ---------------------------------------------------
def gate_e2(window):
    matrix = os.path.join(window, "matrix.tsv")
    rows = read_tsv(matrix)
    if not rows:
        missing("E2-a-matrix", HARD, matrix)
    else:
        detail, ok = [], True
        for w in sorted({int(row["w"]) for row in rows}):
            cell = [row for row in rows if int(row["w"]) == w]
            _, passes, total, short = counts(cell)
            ok = ok and not short and len(cell) == 30
            detail.append("W=%d: %d/30 cells %d/%d" % (w, len(cell), passes, total))
        record("E2-a-matrix", HARD, ok, "; ".join(detail), matrix)

    detail, verdicts = [], []
    for model in ("gqa2", "mha4"):
        for seq in (4, 128):
            def cell(arm):
                return samples(os.path.join(window, "final",
                                            "%s_s%d_%s" % (model, seq, arm),
                                            "r*", "run_*.log"))
            base, arm_s = cell("today"), cell("w1")
            rounds = sorted(set(base) & set(arm_s))
            if not rounds:
                continue
            ratios = [arm_s[r][1] / base[r][1] for r in rounds]
            low, high = bootstrap(ratios)
            ratio = statistics.median(ratios)
            # W = 1 is meant to *be* today's executor, so the claim is
            # equivalence and an equivalence claim needs a band.
            good = low <= 1. <= high or abs(ratio - 1.) < .005
            verdicts.append(good)
            detail.append("%s_s%d: n=%d ratio=%.4f [%.4f,%.4f]"
                          % (model, seq, len(rounds), ratio, low, high))
    if not verdicts:
        missing("E2-b-degeneracy", HARD, os.path.join(window, "final"),
                "no paired W=1 runs")
    else:
        record("E2-b-degeneracy", HARD, all(verdicts), "; ".join(detail),
               os.path.join(window, "final"))

    negative = os.path.join(window, "negative.tsv")
    rows = read_tsv(negative)
    if not rows:
        missing("E2-c-negative", HARD, negative)
    else:
        # A clean sweep here is §10's stop condition.  It is never evidence that
        # the rule can be dropped: it means the reference models do not reach
        # the reordering the rule exists to make safe, and H4 then requires a
        # graph that does.
        clean = [r for r in rows if int(r["passes"]) == int(r["processes"])]
        rates = ", ".join(
            "%s_s%s: %.2f" % (r["model"], r["seq"],
                              (int(r["processes"]) - int(r["passes"]))
                              / int(r["processes"]))
            for r in rows)
        record("E2-c-negative", HARD, not clean,
               "failure rate " + rates
               + ("; STOP CONDITION: passed cleanly in "
                  + ",".join("%s_s%s" % (r["model"], r["seq"]) for r in clean)
                  if clean else ""), negative)


# --- S2r-b: the round's research gate -----------------------------------------
def gate_s2r(place):
    final = os.path.join(place, "final")
    if not os.path.isdir(final):
        return missing("S2r-b-ceiling", RESEARCH, final)
    met, detail = [], []
    for (model, seq), (ceiling, measured_a, target) in sorted(TARGETS.items()):
        prefix = "%s_s%d_" % (model, seq)
        best = None
        for path in sorted(glob.glob(os.path.join(final, prefix + "*"))):
            arm = os.path.basename(path)[len(prefix):]
            found = samples(os.path.join(path, "r*", "run_*.log"))
            if len(found) < 2:
                continue
            values = [value[1] * 1000. for value in found.values()]  # ms -> us
            median = statistics.median(values)
            if best is None or median < best[1]:
                low, high = bootstrap(values)
                best = (arm, median, low, high, len(values))
        if best is None:
            met.append(False)
            detail.append("%s_s%d: no runs" % (model, seq))
            continue
        arm, median, low, high, n = best
        good = median <= target and high < target
        met.append(good)
        closed = (measured_a - median) / (measured_a - ceiling)
        detail.append("%s_s%d: best=%s median=%.1fus ci=[%.1f,%.1f] target=%.0f"
                      " ceiling=%.0f gap_closed=%.0f%% n=%d %s"
                      % (model, seq, arm, median, low, high, target, ceiling,
                         100. * closed, n, "MET" if good else "missed"))
    passed = sum(1 for value in met if value)
    record("S2r-b-ceiling", RESEARCH, passed >= 3,
           "%d/4 reference cells met; " % passed + "; ".join(detail), final)


def main():
    # Four levels up, not three: the first strips verify.py itself, then
    # PLACE_EFT2, experiments and docs.  Three landed on `<repo>/docs` and every
    # gate read `docs/docs/...`, which reports a finished round as 0/14.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(here, "..", "..", ".."))
    experiments = os.path.join(repo, "docs", "experiments")
    sync = os.path.join(experiments, "SYNC_V2")

    gate_h2(sync, repo)
    gate_e3_0(sync, repo)
    correctness_gate("E3-1-correctness", os.path.join(sync, "raw_barrier"))
    census_delta("E3-1-barriers", os.path.join(sync, "raw_barrier"), 3)
    correctness_gate("E3-2-correctness", os.path.join(sync, "raw_solo"))
    correctness_gate("E3-3-correctness", os.path.join(sync, "raw_red"))
    gate_e3_4(sync)
    gate_s2c(os.path.join(experiments, "CHAIN", "raw"), repo)
    gate_e2(os.path.join(experiments, "WINDOW", "raw"))
    gate_s2r(os.path.join(experiments, "PLACE_EFT2", "raw"))

    width = max(len(name) for name, *_ in results)
    failed = 0
    print("%s  kind      verdict  detail" % "gate".ljust(width))
    for name, kind, ok, detail, evidence in results:
        verdict = "PASS" if ok else ("MISS" if kind == RESEARCH else "FAIL")
        if not ok and kind == HARD:
            failed += 1
        print("%s  %s  %-7s  %s" % (name.ljust(width), kind.ljust(8), verdict, detail))
        print("%s  evidence: %s" % (" " * width, evidence))
    hard = sum(1 for _, kind, *_ in results if kind == HARD)
    print("\n%d/%d hard gates pass" % (hard - failed, hard))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
