#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""EX-E3 step 2: how many ArriveEvent calls are single-member at kappa=1.

The share bounds what E3-2 can be worth, and it is read off the host's own
tables rather than a new dump: TILEMEGA_PLAN_DUMP already writes events.tsv
(per-stage event_flags) and schedule.tsv (one row per task), and at kappa=1 a
fine group holds exactly one task, so members==1 for every fine event.  An
aggregate row is single-member only where a stage produced one task.
"""
import os
import sys

AGGREGATE, FINE = 1 << 0, 1 << 1
CELLS = ("gqa2_s4", "gqa2_s128", "mha4_s4", "mha4_s128")


def cell(root, name):
    flags, produced = {}, {}
    with open(os.path.join(root, name, "events.tsv")) as f:
        for line in f.read().splitlines()[1:]:
            stage, _offset, value = line.split("\t")
            flags[int(stage)] = int(value)
    with open(os.path.join(root, name, "schedule.tsv")) as f:
        for line in f.read().splitlines()[1:]:
            stage = int(line.split("\t")[2])
            produced[stage] = produced.get(stage, 0) + 1
    calls = single = 0
    for stage, count in produced.items():
        value = flags.get(stage, 0)
        if value & FINE:
            calls += count
            single += count
        if value & AGGREGATE:
            calls += count
            if count == 1:
                single += count
    return len(produced), sum(produced.values()), calls, single


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_notify", "plan")
    out = os.path.join(os.path.dirname(root), "single_member.tsv")
    with open(out, "w") as f:
        f.write("cell\tstages\ttasks\tarrive_calls\tsingle_member\tshare\n")
        for name in CELLS:
            stages, tasks, calls, single = cell(root, name)
            f.write("%s\t%d\t%d\t%d\t%d\t%.4f\n"
                    % (name, stages, tasks, calls, single, single / calls))
            print("%-10s stages=%-3d tasks=%-6d calls=%-6d single=%-6d share=%.1f%%"
                  % (name, stages, tasks, calls, single, 100 * single / calls))
    print("wrote", out)


if __name__ == "__main__":
    main()
