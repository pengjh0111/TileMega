#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize single-forward and last-iteration timing from fresh processes."""
import glob
import os
import re
import statistics
import sys

TIME = re.compile(r"^E2E_TIME .*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")
ITER = re.compile(r"^E2E_ITER l2_iter1_ms=([0-9.]+)")

print("model\tarm\tn\tsingle_l2_over_l1\tlast_iter_over_l1")
for model in ("gqa2", "mha4"):
    for arm in ("monotone", "reset"):
        single, repeated = [], []
        for path in glob.glob(os.path.join(sys.argv[1], "final", f"{model}_{arm}", "run_*.log")):
            l1 = l2 = again = None
            for line in open(path):
                if match := TIME.match(line):
                    l1, l2 = map(float, match.groups())
                if match := ITER.match(line):
                    again = float(match.group(1))
            if l1 is not None and l2 is not None and again is not None:
                single.append(l2 / l1)
                repeated.append(again / l1)
        print(f"{model}\t{arm}\t{len(single)}\t{statistics.median(single):.6f}\t"
              f"{statistics.median(repeated):.6f}")
