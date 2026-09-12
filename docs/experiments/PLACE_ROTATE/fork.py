#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Adjudicate §4.4's fork rule from the summarize output.

The rule was fixed before the measurement and is applied here by the script so
that no one has to be trusted to read the table the same way twice.
"""
import sys


def pooled(path, arm):
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 9 and parts[0] == "ALL" and parts[2] == arm:
            cells = 0
            for field in parts:
                if field.startswith("cells="):
                    cells = int(field.split("=", 1)[1])
            return float(parts[6]), float(parts[7]), float(parts[8]), cells
    raise SystemExit(f"no pooled row for arm {arm} in {path}")


def main():
    path = sys.argv[1]
    r_neither, n_lo, n_hi, cells = pooled(path, "neither")
    r_full, f_lo, f_hi, _ = pooled(path, "full")

    neither_gain = r_neither <= 0.95 and n_hi < 1.0
    full_contains_one = f_lo <= 1.0 <= f_hi
    if not neither_gain:
        rule = 3
    elif r_full >= 0.99 or full_contains_one:
        rule = 1
    else:
        rule = 2
    print(f"FORK rule={rule} r_neither={r_neither:.4f} ci=[{n_lo:.4f},{n_hi:.4f}] "
          f"r_full={r_full:.4f} ci=[{f_lo:.4f},{f_hi:.4f}] cells={cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
