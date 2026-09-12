#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""EX-D2 offline arm: how much of L2's runtime is schedulable at all?

Reads trace v2 dumps and reports §3.6's four bounds beside the measured latency,
then simulates an earliest-finish-time list schedule over the same task DAG with
the same measured task durations and the measured hop as the cross-worker edge
cost.  The simulator is allowed to place tasks freely on the resident workers,
so its makespan is what a perfect scheduler could have reached with today's task
durations and today's synchronization cost -- not a claim that any such schedule
is legal under the current queue.
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "TRACE_V2"))
import analyze  # noqa: E402

WHOLE_STAGE = 0xFFFFFFFF


def build_dag(dump):
    meta, slots, waits, events = analyze.load(dump)
    kappa = int(meta["kappa"])
    by_index = {e["event_index"]: e for e in events}
    slot_of = {(r["stage"], r["logical_task"]): r["slot"] for r in slots}
    stage_tasks = defaultdict(list)
    for r in slots:
        stage_tasks[r["stage"]].append(r["logical_task"])

    def producers(event_index):
        e = by_index[event_index]
        if e["group"] == WHOLE_STAGE:
            return [slot_of[(e["stage"], t)] for t in stage_tasks[e["stage"]]]
        begin = e["group"] * kappa
        return [slot_of[(e["stage"], t)] for t in range(begin, begin + e["fanin"])
                if (e["stage"], t) in slot_of]

    succ = defaultdict(list)
    preds = defaultdict(list)
    for r in slots:
        for w in waits[r["wait_begin_idx"]: r["wait_begin_idx"] + r["wait_count"]]:
            for p in producers(w["event_index"]):
                if p == r["slot"]:
                    continue
                succ[p].append(r["slot"])
                preds[r["slot"]].append(p)
    return meta, slots, preds, succ


def simulate(meta, slots, preds, succ, hop):
    """EFT list scheduling, upward rank first, resident workers only."""
    grid = int(meta["grid"])
    weight = {r["slot"]: r["run_end"] - r["run_begin"] for r in slots}
    order = sorted(weight, key=lambda s: -s)  # reverse topological: slot ids rise with stage
    stage_of = {r["slot"]: r["stage"] for r in slots}
    stage_pos = {}
    for r in sorted(slots, key=lambda r: r["slot"]):
        stage_pos.setdefault(r["stage"], len(stage_pos))
    topo = sorted(weight, key=lambda s: (stage_pos[stage_of[s]], s))

    rank = {}
    for s in reversed(topo):
        rank[s] = weight[s] + max((hop + rank[t] for t in succ[s]), default=0)

    avail = [0] * grid
    finish = {}
    where = {}
    for s in sorted(topo, key=lambda s: (-rank[s], stage_pos[stage_of[s]], s)):
        ready_cross = max((finish[p] + hop for p in preds[s]), default=0)
        # Only a worker already holding a predecessor can beat the all-cross
        # ready time, so the rest of the grid is represented by its earliest
        # free worker.
        candidates = {where[p] for p in preds[s]}
        idle = min(range(grid), key=lambda w: avail[w])
        candidates.add(idle)
        best = None
        for worker in candidates:
            ready = max((finish[p] + (0 if where[p] == worker else hop)
                         for p in preds[s]), default=0)
            start = max(ready, avail[worker])
            if best is None or start + weight[s] < best[0]:
                best = (start + weight[s], worker, start)
        end, worker, _ = best
        finish[s] = end
        where[s] = worker
        avail[worker] = end
        del ready_cross
    return max(finish.values()), len(set(where.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for dump in args.dumps:
        stats = analyze.analyze(dump)
        meta, slots, preds, succ = build_dag(dump)
        makespan, used = simulate(meta, slots, preds, succ, stats["hop_p50_ns"])
        rows.append({
            "cell": os.path.basename(os.path.normpath(dump)),
            "model": stats["model"], "seq": stats["seq"], "grid": stats["grid"],
            "slots": len(slots),
            "measured_l2_ms": stats["measured_l2_ms"],
            "kernel_span_ms": stats["kernel_span_ms"],
            "work_lb_ms": stats["work_lb_ms"],
            "queue_lb_ms": stats["queue_lb_ms"],
            "cp_lb_nosync_ms": stats["cp_lb_nosync_ms"],
            "cp_lb_sync_ms": stats["cp_lb_sync_ms"],
            "hop_p50_ns": stats["hop_p50_ns"],
            "eft_makespan_ms": makespan / 1e6,
            "eft_workers_used": used,
            "eft_over_measured": makespan / 1e6 / stats["measured_l2_ms"],
            "hol_reclaimable_ms": stats["hol_reclaimable_ns"] / 1e6,
            "cross_worker_edge_fraction": stats["dag_cross_worker_fraction"],
        })

    keys = list(rows[0].keys())
    path = os.path.join(args.out, "headroom.tsv")
    exists = os.path.exists(path) and args.label
    with open(path, "a" if exists else "w") as f:
        if not exists:
            f.write("group\t" + "\t".join(keys) + "\n")
        for row in rows:
            f.write((args.label or "reference") + "\t"
                    + "\t".join(str(row[k]) for k in keys) + "\n")

    # The markdown is regenerated from the whole tsv every time, so a run that
    # appends one cell does not leave a section header per invocation behind.
    groups, order = {}, []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(head, line.rstrip("\n").split("\t")))
            if row["group"] not in groups:
                groups[row["group"]] = []
                order.append(row["group"])
            groups[row["group"]].append(row)

    with open(os.path.join(args.out, "headroom.md"), "w") as f:
        f.write("# EX-D2 offline headroom\n\n")
        f.write("`eft_makespan` is an earliest-finish-time list schedule over the\n")
        f.write("exact task DAG: measured task durations, measured hop p50 on every\n")
        f.write("cross-worker edge, free placement on the resident workers.  It is an\n")
        f.write("optimistic bound on what rescheduling alone could buy, and it says\n")
        f.write("nothing about whether such a schedule is legal under today's queue.\n")
        for group in order:
            f.write(f"\n## {group}\n\n")
            f.write("| cell | slots | measured l2 | work_lb | queue_lb | cp_lb_nosync | "
                    "cp_lb_sync | EFT | EFT/measured | workers |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for row in groups[group]:
                f.write("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | "
                        "{:.4f} | {:.2f} | {} |\n".format(
                            row["cell"], row["slots"],
                            float(row["measured_l2_ms"]), float(row["work_lb_ms"]),
                            float(row["queue_lb_ms"]), float(row["cp_lb_nosync_ms"]),
                            float(row["cp_lb_sync_ms"]), float(row["eft_makespan_ms"]),
                            float(row["eft_over_measured"]), row["eft_workers_used"]))

    for row in rows:
        print("HEADROOM cell={cell} measured_ms={measured_l2_ms:.4f} "
              "eft_ms={eft_makespan_ms:.4f} ratio={eft_over_measured:.3f} "
              "queue_lb_ms={queue_lb_ms:.4f}".format(**row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
