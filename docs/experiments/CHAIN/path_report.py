#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""§6.4 asks how many hops each candidate's critical path pays.  `predicted.tsv`
answers with counts; this answers with nanoseconds, from the path dumps, because
a count only matters multiplied by the per-hop cost.

Run after `run.sh` with TILEMEGA_CHAIN_PATH_DUMP pointed at `raw/path`:

    python3 path_report.py raw/path

Every gap on a critical path lands on an `h` edge -- `q` and `s` edges measure 0
ns of gap in all 42 dumps -- so `hop_gap_ns` is the whole schedule-dependent
synchronization cost on the path, and `hop_gap_ns / hops` recovers the per-hop
constant independently of F-145's microbenchmark.
"""
import csv
import os
import sys

CELLS = [("gqa2", 4), ("mha4", 4), ("gqa2", 128), ("mha4", 128),
         ("real", 4), ("real", 128)]
CANDIDATES = ["legacy_grid_stride", "rotate", "balanced", "eft", "band",
              "wavefront", "chain"]


def decompose(path):
    """Walk one dump source->sink and split its span into work and gap by edge."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))[::-1]
    for row in rows:
        for key in ("task_ns", "start_ns", "end_ns", "stretch"):
            row[key] = float(row[key])
    gap = {"h": 0.0, "q": 0.0, "s": 0.0}
    count = {"h": 0, "q": 0, "s": 0}
    for before, after in zip(rows, rows[1:]):
        # The dump is written sink-first, so `edge_to_next` on the later row is
        # the edge *into* it once the list is reversed.
        edge = after["edge_to_next"]
        if edge in gap:
            gap[edge] += after["start_ns"] - before["end_ns"]
            count[edge] += 1
    task = sum(r["task_ns"] for r in rows)
    stretched = sum(r["end_ns"] - r["start_ns"] for r in rows)
    return {
        "steps": len(rows),
        "hops": count["h"],
        "queue_edges": count["q"],
        "same_worker_edges": count["s"],
        "hop_gap_ns": gap["h"],
        "queue_gap_ns": gap["q"],
        "same_worker_gap_ns": gap["s"],
        "task_ns": task,
        "stretched_ns": stretched,
        "stretch_excess_ns": stretched - task,
        "span_ns": rows[-1]["end_ns"] - rows[0]["start_ns"],
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "raw/path"
    fields = ["model", "seq", "candidate", "steps", "hops", "queue_edges",
              "same_worker_edges", "hop_gap_ns", "queue_gap_ns",
              "same_worker_gap_ns", "task_ns", "stretched_ns",
              "stretch_excess_ns", "span_ns", "hop_ns_per_hop", "hop_share"]
    out = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t",
                         lineterminator="\n")
    out.writeheader()
    missing = []
    for model, seq in CELLS:
        for candidate in CANDIDATES:
            path = os.path.join(root, f"{model}_s{seq}_{candidate}_path.tsv")
            if not os.path.exists(path):
                missing.append(path)
                continue
            row = decompose(path)
            row.update(model=model, seq=seq, candidate=candidate)
            row["hop_ns_per_hop"] = (row["hop_gap_ns"] / row["hops"]
                                     if row["hops"] else 0.0)
            row["hop_share"] = (row["hop_gap_ns"] / row["span_ns"]
                                if row["span_ns"] else 0.0)
            for key in ("hop_gap_ns", "queue_gap_ns", "same_worker_gap_ns",
                        "task_ns", "stretched_ns", "stretch_excess_ns",
                        "span_ns", "hop_ns_per_hop"):
                row[key] = f"{row[key]:.1f}"
            row["hop_share"] = f"{row['hop_share']:.4f}"
            out.writerow(row)
    if missing:
        # A partial dump would quietly shrink the comparison §6.4 asks for, so
        # say which cells are absent instead of printing a shorter table.
        print(f"# missing {len(missing)} dumps, first: {missing[0]}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
