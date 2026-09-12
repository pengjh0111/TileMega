#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""D2-b: check the mode-5 rotation base against a hand computation.

The hand side takes nothing from the placement code.  A trace v2 dump of the
same model records, per slot, which stage it belongs to, and the queue is
stage-major in `model.stage_order`, so the order in which stages first appear
in `slots.tsv` is that stage order and the number of slots carrying a stage is
`active_tasks(stage)`.  From those two facts alone,

    base[s] = (sum of active_tasks over the stages before s in stage order) mod grid

is a prefix sum, computed here and compared against every `E2E_PLACE_BASE` row
the mode-5 binary printed.
"""
import argparse
import sys


def hand(dump):
    order, count = [], {}
    with open(f"{dump}/slots.tsv") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]
    for row in sorted(rows, key=lambda r: int(r["slot"])):
        stage = int(row["stage"])
        if stage not in count:
            order.append(stage)
            count[stage] = 0
        count[stage] += 1
    grid = 0
    with open(f"{dump}/meta.tsv") as f:
        for line in f:
            key, _, value = line.rstrip("\n").partition("\t")
            if key == "grid":
                grid = int(value)
    running = 0
    out = []
    for position, stage in enumerate(order):
        out.append((position, stage, count[stage], running % grid))
        running += count[stage]
    return grid, out


def code(log):
    rows = {}
    with open(log, errors="replace") as f:
        for line in f:
            if line.startswith("E2E_PLACE_BASE "):
                fields = dict(kv.split("=", 1) for kv in line.split()[1:])
                rows[int(fields["pos"])] = (int(fields["pos"]), int(fields["stage"]),
                                            int(fields["active"]), int(fields["base"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="a trace v2 dump of the same cell")
    ap.add_argument("--log", required=True, help="stdout of the mode-5 binary with "
                                                 "TILEMEGA_PLACEMENT_BASE_DUMP=1")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    grid, expected = hand(args.dump)
    actual = code(args.log)
    if not actual:
        raise SystemExit(f"no E2E_PLACE_BASE rows in {args.log}")
    print(f"grid={grid} stages={len(expected)}")
    print("pos\tstage\tactive\thand_base\tcode_base")
    bad = []
    for row in expected:
        got = actual.get(row[0])
        if got is None or got != row:
            bad.append((row, got))
        if row[0] < args.show or (got is not None and got != row):
            print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t"
                  f"{'missing' if got is None else got[3]}")
    print(f"BASE_EXAMPLE grid={grid} stages={len(expected)} mismatches={len(bad)} "
          f"verdict={'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
