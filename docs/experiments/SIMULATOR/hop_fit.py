#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# EX-S1 §5.2: fit hop_ns(N, R) from contention.tsv and emit it as the
# simulator's edge-weight function.
#
# N = consumers polling, R = distinct event rows under contention, so N/R is how
# many CTAs share one cache line.  The fit is deliberately the simplest form the
# data supports:
#
#     hop_ns(N, R) = c0 + c1*log2(1 + N/R) + c2*log2(R)
#
# Anything richer would be fitting noise: the measured surface varies by under
# 5% of c0 across the whole 1 <= N <= 256, 1 <= R <= 64 sweep, so c1 and c2 are
# reported with their standard errors and the residual spread, and the README
# says plainly that they are near zero.  Never move an expected value to match
# an implementation -- the flat surface is the result, not a fit failure.
import math
import sys

def rows(path, mode, backoff):
    out = []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            if r["poll_mode"] == mode and int(r["backoff_ns"]) == backoff:
                out.append(r)
    return out

def fit(samples, column):
    # Weighted least squares; weight = 1/se^2 on the cell mean, floored so a
    # single-cell se of 0 cannot take over the fit.
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
    # Scale the parameter covariance by the reduced chi-square: the cell se
    # measures sampling noise only, and the model misfit is larger than that.
    scale = chi2 / dof
    se_c = [math.sqrt(max(inv[i][i] * scale, 0.0)) for i in range(p)]
    rms = math.sqrt(sum(e * e for e in resid) / len(resid))
    return c, se_c, rms, max(abs(e) for e in resid), scale

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

def main():
    tsv = sys.argv[1] if len(sys.argv) > 1 else "docs/experiments/SIMULATOR/contention.tsv"
    load = sys.argv[2] if len(sys.argv) > 2 else "docs/experiments/SIMULATOR/contention_load.tsv"
    emit = sys.argv[3] if len(sys.argv) > 3 else None

    fits = {}
    for label, path, mode in (("rmw", tsv, "rmw"), ("load", load, "load")):
        for backoff in (64, 0):
            s = rows(path, mode, backoff)
            if not s:
                continue
            for column in ("hop_trim_mean_ns", "hop_mean_ns", "last_trim_mean_ns", "last_mean_ns"):
                c, se, rms, mx, scale = fit(s, column)
                key = (label, backoff, column)
                fits[key] = c
                print("FIT mode=%s backoff=%d column=%-12s cells=%d "
                      "c0=%.1f+-%.1f c1=%.2f+-%.2f c2=%.2f+-%.2f "
                      "resid_rms_ns=%.1f resid_max_ns=%.1f chi2_red=%.1f"
                      % (label, backoff, column, len(s), c[0], se[0], c[1], se[1],
                         c[2], se[2], rms, mx, scale))
            lo = min(float(r["hop_trim_mean_ns"]) for r in s)
            hi = max(float(r["hop_trim_mean_ns"]) for r in s)
            print("SPREAD mode=%s backoff=%d hop_min_ns=%.1f hop_max_ns=%.1f "
                  "range_over_c0=%.1f%%" % (label, backoff, lo, hi,
                                            100.0 * (hi - lo) / fits[(label, backoff, "hop_trim_mean_ns")][0]))

    # The default primitive is RMW with the generated `__nanosleep(64)` backoff,
    # so that is the curve the simulator uses; the other three arms are recorded
    # for comparison only (R2 §5.2: record load polling, do not change the default).
    c = fits[("rmw", 64, "hop_trim_mean_ns")]
    if emit:
        with open(emit, "w") as f:
            f.write("# hop_ns(N, R) = c0 + c1*log2(1 + N/R) + c2*log2(R)\n")
            f.write("# EX-S1 calibration, RMW poll with the generated nanosleep(64), 0.1% trimmed mean\n")
            f.write("arm\trmw_backoff64_hop_trim_mean\n")
            f.write("c0\t%.6f\nc1\t%.6f\nc2\t%.6f\n" % (c[0], c[1], c[2]))
        print("EMIT %s" % emit)
    print("HOP_NS_DEFAULT c0=%.3f c1=%.3f c2=%.3f" % (c[0], c[1], c[2]))

main()
