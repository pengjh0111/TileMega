#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# EX-E3 step 0 (R3 §5.1): fit hop_ns(N, R) once per wait policy and recommend
# the calibrated policy for a target.
#
# The model is F-145's, unchanged, so the arms are comparable to the round-two
# number they have to beat:
#
#     hop_ns(N, R) = c0 + c1*log2(1 + N/R) + c2*log2(R)
#
# c0 is the per-hop constant EX-E3 exists to reduce; c1 and c2 were zero inside
# one standard error on both architectures and are refitted per arm only to
# check that a different policy does not wake the contention term up.
#
# Recommendation rule, fixed here before the numbers were read: take the policy
# with the smallest c0.  If that policy carries no backoff, keep it only when
# spinning costs a co-resident compute worker no more than the idle arm's own
# round-to-round spread (its p90/p50); otherwise recommend the smallest-c0
# policy that retains a backoff.  Both are printed either way, so choosing
# differently is possible and has to be argued in the report rather than by
# editing this rule afterwards.
import math
import sys

def read(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f]

def invert(a):
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[pivot] = m[pivot], m[i]
        d = m[i][i]
        m[i] = [v / d for v in m[i]]
        for r in range(n):
            if r != i and m[r][i] != 0.0:
                f = m[r][i]
                m[r] = [v - f * w for v, w in zip(m[r], m[i])]
    return [row[n:] for row in m]

def fit(samples, column):
    design, target, weight = [], [], []
    for r in samples:
        n, k = float(r["consumers"]), float(r["rows"])
        se = max(float(r["hop_se_ns"]), 0.1)
        design.append([1.0, math.log2(1.0 + n / k), math.log2(k)])
        target.append(float(r[column]))
        weight.append(1.0 / (se * se))
    p = len(design[0])
    ata = [[sum(w * x[i] * x[j] for x, w in zip(design, weight)) for j in range(p)]
           for i in range(p)]
    atb = [sum(w * x[i] * y for x, y, w in zip(design, target, weight)) for i in range(p)]
    inv = invert(ata)
    c = [sum(inv[i][j] * atb[j] for j in range(p)) for i in range(p)]
    resid = [y - sum(ci * xi for ci, xi in zip(c, x)) for x, y in zip(design, target)]
    dof = max(len(design) - p, 1)
    chi2 = sum(w * e * e for e, w in zip(resid, weight))
    scale = chi2 / dof
    se_c = [math.sqrt(max(inv[i][i] * scale, 0.0)) for i in range(p)]
    rms = math.sqrt(sum(e * e for e in resid) / len(resid))
    return c, se_c, rms, max(abs(e) for e in resid), scale

def main():
    tsv = sys.argv[1] if len(sys.argv) > 1 else "backoff.tsv"
    spin_tsv = sys.argv[2] if len(sys.argv) > 2 else "spin_interference.tsv"
    emit = sys.argv[3] if len(sys.argv) > 3 else None

    rows = read(tsv)
    arms = []
    seen = []
    for r in rows:
        if r["arm"] not in seen:
            seen.append(r["arm"])
    for arm in seen:
        s = [r for r in rows if r["arm"] == arm]
        c, se, rms, mx, scale = fit(s, "hop_trim_mean_ns")
        clast = fit(s, "last_trim_mean_ns")[0]
        lo = min(float(r["hop_trim_mean_ns"]) for r in s)
        hi = max(float(r["hop_trim_mean_ns"]) for r in s)
        arms.append({
            "arm": arm,
            "spin_iters": int(s[0]["spin_iters"]),
            "backoff_ns": int(s[0]["backoff_ns"]),
            "grow": int(s[0]["grow"]),
            "cap_ns": int(s[0]["cap_ns"]),
            "c0": c[0], "c1": c[1], "c2": c[2],
            "se0": se[0], "se1": se[1], "se2": se[2],
            "rms": rms, "max": mx, "chi2": scale,
            "cells": len(s), "lo": lo, "hi": hi, "last_c0": clast[0],
            "inversions": sum(int(r["inversions"]) for r in s),
        })
        print("FIT arm=%-13s cells=%2d c0=%7.1f+-%5.1f c1=%6.2f+-%5.2f "
              "c2=%6.2f+-%5.2f resid_rms_ns=%5.1f resid_max_ns=%5.1f "
              "chi2_red=%8.1f hop_min_ns=%7.1f hop_max_ns=%7.1f inversions=%d"
              % (arm, len(s), c[0], se[0], c[1], se[1], c[2], se[2], rms, mx,
                 scale, lo, hi, sum(int(r["inversions"]) for r in s)))

    # Per-rotation means expose an arm-order or warm-up effect, which the
    # pooled fit would otherwise absorb into a wide standard error.
    rots = []
    for r in rows:
        rot = r.get("rotation", "0")
        if rot not in rots:
            rots.append(rot)
    if len(rots) > 1:
        for arm in seen:
            means = []
            for rot in rots:
                cells = [float(r["hop_trim_mean_ns"]) for r in rows
                         if r["arm"] == arm and r.get("rotation", "0") == rot]
                means.append(sum(cells) / len(cells) if cells else float("nan"))
            spread = max(means) - min(means)
            print("ORDER arm=%-13s %s spread_ns=%6.1f"
                  % (arm, " ".join("rot%s=%7.1f" % (rot, m)
                                   for rot, m in zip(rots, means)), spread))

    # `hop` is when the row was first observed; `last` is when the slowest
    # consumer of that row got out of its wait.  A runtime task cannot start on
    # the first observation, so both are reported and a policy that only wins
    # on `hop` has not won anything the megakernel can spend.
    for a in arms:
        print("LAST arm=%-13s hop_c0=%7.1f last_c0=%7.1f last_over_hop=%.3f"
              % (a["arm"], a["c0"], a["last_c0"],
                 a["last_c0"] / a["c0"] if a["c0"] else float("nan")))

    base = next((a for a in arms if a["arm"] == "literal64"), None)
    if base:
        for a in arms:
            print("RATIO arm=%-13s c0_over_literal64=%.4f delta_ns=%+.1f"
                  % (a["arm"], a["c0"] / base["c0"], a["c0"] - base["c0"]))

    interference = {}
    modes = []
    try:
        for r in read(spin_tsv):
            mode = r.get("compute_mode", "fma")
            if mode not in modes:
                modes.append(mode)
            interference[(mode, r["arm"])] = r
    except OSError:
        print("SPIN no %s; the interference question is unanswered" % spin_tsv)

    tolerance = None
    worst_spin = None
    for mode in modes:
        idle = interference.get((mode, "idle"))
        if not idle:
            continue
        p50, p90 = float(idle["compute_p50_cycles"]), float(idle["compute_p90_cycles"])
        if not p50:
            continue
        for name in ("idle", "spin", "backoff64"):
            r = interference.get((mode, name))
            if not r:
                continue
            ratio = float(r["compute_p50_cycles"]) / p50
            print("SPIN compute=%-4s arm=%-9s compute_p50_cycles=%9.1f "
                  "ratio_to_idle=%.4f paired_fraction=%.4f samples=%s"
                  % (mode, name, float(r["compute_p50_cycles"]), ratio,
                     float(r["paired_fraction"]), r["samples"]))
            if name == "spin":
                worst_spin = ratio if worst_spin is None else max(worst_spin, ratio)
        # The tolerance is the idle arm's own round-to-round spread, taken from
        # whichever compute shape is noisiest: a policy has to clear the worst.
        t = p90 / p50
        tolerance = t if tolerance is None else max(tolerance, t)
        if tolerance is None:
            print("SPIN idle arm has no samples; the interference question is "
                  "unanswered and the recommendation falls back to keeping a backoff")
        else:
            print("SPIN tolerance_from_idle_p90_over_p50=%.4f" % tolerance)

    ranked = sorted(arms, key=lambda a: a["c0"])
    best = ranked[0]
    with_backoff = [a for a in ranked if a["backoff_ns"] > 0]
    chosen = best
    reason = "smallest c0"
    if best["backoff_ns"] == 0 and tolerance is not None and worst_spin is not None:
        spin_ratio = worst_spin
        if spin_ratio > tolerance:
            chosen = with_backoff[0] if with_backoff else best
            reason = ("smallest c0 that keeps a backoff; pure spin costs a "
                      "co-resident compute worker %.4f against a %.4f tolerance"
                      % (spin_ratio, tolerance))
        else:
            reason = ("smallest c0; pure spin costs a co-resident compute worker "
                      "%.4f within a %.4f tolerance" % (spin_ratio, tolerance))
    elif best["backoff_ns"] == 0:
        chosen = with_backoff[0] if with_backoff else best
        reason = "smallest c0 that keeps a backoff; interference unmeasured"

    print("RECOMMEND arm=%s spin_iters=%d backoff_ns=%d grow=%d cap_ns=%d c0=%.1f "
          "reason=%s" % (chosen["arm"], chosen["spin_iters"], chosen["backoff_ns"],
                         chosen["grow"], chosen["cap_ns"], chosen["c0"], reason))
    if base:
        print("RECOMMEND_GAIN c0_ns %.1f -> %.1f (%.1f%% of the status quo)"
              % (base["c0"], chosen["c0"], 100.0 * chosen["c0"] / base["c0"]))

    if emit:
        with open(emit, "w") as f:
            f.write("# EX-E3 step 0: the calibrated wait policy and the curve it was chosen from.\n")
            f.write("# hop_ns(N, R) = c0 + c1*log2(1 + N/R) + c2*log2(R), 0.1%% trimmed mean.\n")
            f.write("# rule: %s\n" % reason)
            f.write("key\tvalue\n")
            for k in ("arm", "spin_iters", "backoff_ns", "grow", "cap_ns"):
                f.write("%s\t%s\n" % (k, chosen[k]))
            for k in ("c0", "c1", "c2"):
                f.write("%s\t%.6f\n" % (k, chosen[k]))
            f.write("status_quo_c0\t%.6f\n" % (base["c0"] if base else float("nan")))
        print("EMIT %s" % emit)

main()
