#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Lift the per-sample measurements out of `raw/final/**/run_*.log` into one TSV.

`.gitignore:32` excludes `docs/experiments/**/*.log`, so the per-process logs the
paired statistics are computed from cannot be committed.  This writes the numbers
themselves -- one row per process, no aggregation, no verdict -- so that
`verify.py` can recompute every ratio, median and interval on a fresh clone.

It is deliberately not a summary: there is no median here and no PASS anywhere.
Running it on a tree that still has the logs and diffing against the committed
`raw/samples.tsv` is what proves the two agree.

    python3 docs/experiments/PLACE_EFT/collect_samples.py [RAW_DIR]
"""
import os
import re
import sys

TIME = re.compile(r"^E2E_TIME\b.*?\bl1_ms=([0-9.]+).*?\bl2_ms=([0-9.]+)")
STATUS = re.compile(r"^RESULT status=(\w+)")
# raw/final/<group>/[r<round>/]run_<n>.log
ROUND = re.compile(r"^r([0-9]+)$")
SAMPLE = re.compile(r"^run_([0-9]+)\.log$")


def scan(path):
    l1 = l2 = ""
    status = "MISSING"
    for line in open(path, errors="replace"):
        match = TIME.match(line)
        if match:
            l1, l2 = match.group(1), match.group(2)
        match = STATUS.match(line)
        if match:
            status = match.group(1)
    return l1, l2, status


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "raw")
    final = os.path.join(raw, "final")
    rows = []
    for group in sorted(os.listdir(final)):
        base = os.path.join(final, group)
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            here = os.path.join(base, entry)
            if os.path.isdir(here):
                match = ROUND.match(entry)
                rnd = match.group(1) if match else "-"
                logs = [(rnd, os.path.join(here, f)) for f in sorted(os.listdir(here))
                        if SAMPLE.match(f)]
            elif SAMPLE.match(entry):
                logs = [("-", here)]
            else:
                continue
            for rnd, path in logs:
                sample = SAMPLE.match(os.path.basename(path)).group(1)
                l1, l2, status = scan(path)
                rows.append((group, rnd, sample, l1, l2, status))
    print("group\tround\tsample\tl1_ms\tl2_ms\tstatus")
    for row in rows:
        print("\t".join(row))
    print(f"collect_samples: {len(rows)} processes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
