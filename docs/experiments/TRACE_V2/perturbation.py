#!/usr/bin/env python3
"""D1-c: does carrying trace v2 change what we are measuring?

Paired within one session -- round r contributes one `on` and one `off` sample
per cell and the two are divided against each other -- because absolute latency
is not comparable across sessions.
"""
import argparse
import glob
import os
import random
import re
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl2_ms=([0-9.]+)")


def samples(raw, cell, arm, runs):
    out = []
    for round_index in range(runs):
        logs = sorted(glob.glob(os.path.join(
            raw, "final", f"perturb_{cell}_{arm}", f"r{round_index}", "run_*.log")))
        values = []
        for log in logs:
            with open(log, errors="replace") as f:
                for line in f:
                    m = TIME.match(line)
                    if m:
                        values.append(float(m.group(1)))
        if len(values) != 1:
            raise SystemExit(f"{cell} {arm} r{round_index}: {len(values)} samples, expected 1")
        out.append(values[0])
    return out


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def bootstrap(values, draws=20000):
    rng = random.Random(20260906)
    n = len(values)
    seen = sorted(median([values[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return seen[int(0.025 * draws)], seen[int(0.975 * draws)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--limit", type=float, default=1.02)
    args = ap.parse_args()

    worst = 0.0
    print("cell\trounds\toff_median_ms\ton_median_ms\tratio\tci_lo\tci_hi")
    for model in ("gqa2", "mha4"):
        for seq in (4, 128):
            cell = f"{model}_s{seq}"
            off = samples(args.raw, cell, "off", args.runs)
            on = samples(args.raw, cell, "on", args.runs)
            ratios = [a / b for a, b in zip(on, off)]
            ratio = median(ratios)
            lo, hi = bootstrap(ratios)
            worst = max(worst, ratio)
            print(f"{cell}\t{args.runs}\t{median(off):.6f}\t{median(on):.6f}"
                  f"\t{ratio:.4f}\t{lo:.4f}\t{hi:.4f}")
    verdict = "PASS" if worst <= args.limit else "FAIL"
    print(f"PERTURBATION worst_ratio={worst:.4f} limit={args.limit:.2f} verdict={verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
