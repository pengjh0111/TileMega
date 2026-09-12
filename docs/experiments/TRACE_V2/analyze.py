#!/usr/bin/env python3
"""EX-D1 §3.6: reconstruct per-hop latency, head-of-line blocking, per-worker
occupancy, the critical path and the four bounds from one trace v2 dump.

Every definition here is the one fixed in the round-one prompt before the
measurement was taken.  Nothing is dropped silently: a hop that comes out
negative is counted and printed, never filtered.
"""
import argparse
import os
import sys
from collections import defaultdict

WHOLE_STAGE = 0xFFFFFFFF


def read_tsv(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def read_meta(path):
    out = {}
    for row in read_tsv(path):
        out[row["key"]] = row["value"]
    return out


def percentile(values, q):
    """Nearest-rank on the sorted sample; no interpolation, so every reported
    number is a value that was actually measured."""
    if not values:
        return 0
    s = sorted(values)
    at = int(round(q * (len(s) - 1)))
    return s[min(at, len(s) - 1)]


def load(dump):
    meta = read_meta(os.path.join(dump, "meta.tsv"))
    slots = read_tsv(os.path.join(dump, "slots.tsv"))
    waits = read_tsv(os.path.join(dump, "waits.tsv"))
    events = read_tsv(os.path.join(dump, "events.tsv"))
    for r in slots:
        for k in ("slot", "worker", "stage", "logical_task", "smid", "wait_begin",
                  "ready", "run_begin", "run_end", "publish_end", "wait_begin_idx",
                  "wait_count", "dependency_begin", "dependency_count",
                  "run_begin_clk", "run_end_clk"):
            r[k] = int(r[k])
    for r in waits:
        for k in ("wait_index", "producer", "group", "event_index"):
            r[k] = int(r[k])
    for r in events:
        for k in ("event_index", "stage", "group", "fanin", "publish_ns"):
            r[k] = int(r[k])
    return meta, slots, waits, events


def analyze(dump):
    meta, slots, waits, events = load(dump)
    kappa = int(meta["kappa"])
    grid = int(meta["grid"])
    l2_ms = float(meta["l2_ms"])
    out = {}

    by_index = {e["event_index"]: e for e in events}
    # Which concrete tasks publish an event row.  With the aggregate row it is
    # every task of the stage; with a fine row it is the kappa-sized group, and
    # the group id is logical_task // kappa by construction.
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

    kernel_start = min(r["wait_begin"] for r in slots)
    kernel_end = max(r["publish_end"] for r in slots)
    span_ns = kernel_end - kernel_start
    out["kernel_span_ns"] = span_ns
    out["kernel_span_ms"] = span_ns / 1e6
    out["measured_l2_ms"] = l2_ms

    # --- per-hop -----------------------------------------------------------
    # p(j) is the latest instant any event this slot waits on became visible;
    # a slot with no waits was releasable from the start.
    ready_from = {}
    hops = []
    negative = []
    for r in slots:
        rows = waits[r["wait_begin_idx"]: r["wait_begin_idx"] + r["wait_count"]]
        if not rows:
            ready_from[r["slot"]] = kernel_start
            continue
        p = max(by_index[w["event_index"]]["publish_ns"] for w in rows)
        ready_from[r["slot"]] = p
        hop = r["ready"] - p
        hops.append(hop)
        if hop < 0:
            negative.append((r["slot"], hop))
    out["hop_samples"] = len(hops)
    out["hop_p50_ns"] = percentile(hops, 0.50)
    out["hop_p90_ns"] = percentile(hops, 0.90)
    out["hop_max_ns"] = max(hops) if hops else 0
    out["hop_negative_count"] = len(negative)
    out["hop_negative_slots"] = ",".join(f"{s}:{h}" for s, h in negative[:20]) or "none"

    hop_p50 = out["hop_p50_ns"]

    # --- per-worker occupancy and head-of-line blocking --------------------
    queues = defaultdict(list)
    for r in sorted(slots, key=lambda r: r["slot"]):
        queues[r["worker"]].append(r)

    busy = {}
    wait_total = {}
    publish_total = {}
    hol_worker = {}
    hol_total = 0
    for worker, queue in queues.items():
        busy[worker] = sum(r["run_end"] - r["run_begin"] for r in queue)
        wait_total[worker] = sum(r["ready"] - r["wait_begin"] for r in queue)
        publish_total[worker] = sum(r["publish_end"] - r["run_end"] for r in queue)
        # Ready instants of the slots behind j, so "some later slot could have
        # run here" is a lookup rather than a rescan.
        later_ready = [0] * (len(queue) + 1)
        hol = 0
        prev_end = kernel_start
        for i, r in enumerate(queue):
            begin = max(prev_end, r["wait_begin"])
            end = r["ready"]
            if end > begin:
                # Reclaimable only if a slot further down this queue was already
                # released at the instant this one stopped blocking.
                if any(ready_from[k["slot"]] < end for k in queue[i + 1:]):
                    hol += end - begin
            prev_end = r["run_end"]
        hol_worker[worker] = hol
        hol_total += hol
        del later_ready

    out["workers_used"] = len(queues)
    out["grid"] = grid
    out["busy_total_ns"] = sum(busy.values())
    out["busy_max_worker_ns"] = max(busy.values())
    out["busy_mean_worker_ns"] = sum(busy.values()) / len(busy)
    out["idle_total_ns"] = span_ns * len(queues) - sum(busy.values())
    out["idle_fraction_of_worker_time"] = out["idle_total_ns"] / (span_ns * len(queues))
    out["wait_total_ns"] = sum(wait_total.values())
    out["publish_total_ns"] = sum(publish_total.values())
    out["hol_reclaimable_ns"] = hol_total
    out["hol_fraction_of_kernel"] = hol_total / (span_ns * len(queues))
    out["hol_worker_p50_ns"] = percentile(list(hol_worker.values()), 0.50)
    out["hol_worker_p90_ns"] = percentile(list(hol_worker.values()), 0.90)
    out["hol_worker_max_ns"] = max(hol_worker.values())
    out["hol_workers_nonzero"] = sum(1 for v in hol_worker.values() if v > 0)

    # --- the task DAG ------------------------------------------------------
    # (stage position, slot) is a topological order: dependency edges strictly
    # increase the stage position, and a queue edge either increases it or keeps
    # it while increasing the slot index.
    stage_first_slot = {}
    for r in sorted(slots, key=lambda r: r["slot"]):
        stage_first_slot.setdefault(r["stage"], r["slot"])
    preds = defaultdict(list)
    cross = 0
    same = 0
    slot_row = {r["slot"]: r for r in slots}
    for r in slots:
        rows = waits[r["wait_begin_idx"]: r["wait_begin_idx"] + r["wait_count"]]
        for w in rows:
            for producer in producers(w["event_index"]):
                if producer == r["slot"]:
                    continue
                if slot_row[producer]["worker"] == r["worker"]:
                    same += 1
                    preds[r["slot"]].append((producer, 0))
                else:
                    cross += 1
                    preds[r["slot"]].append((producer, hop_p50))
    for queue in queues.values():
        for prev, cur in zip(queue, queue[1:]):
            preds[cur["slot"]].append((prev["slot"], 0))
    out["dag_same_worker_edges"] = same
    out["dag_cross_worker_edges"] = cross
    out["dag_cross_worker_fraction"] = cross / (same + cross) if same + cross else 0.0

    stage_pos = {}
    for r in sorted(slots, key=lambda r: r["slot"]):
        stage_pos.setdefault(r["stage"], len(stage_pos))
    order = sorted((r["slot"] for r in slots),
                   key=lambda s: (stage_pos[slot_row[s]["stage"]], s))

    def longest(edge_weight, include_queue, node_publish=False):
        """Earliest-finish over the DAG.  `edge_weight` scales the cross-worker
        synchronization cost; `include_queue` decides whether one worker's
        serialization is part of the bound."""
        finish = {}
        parent = {}
        for s in order:
            best = 0
            best_from = None
            for pred, w in preds[s]:
                queue_edge = slot_row[pred]["worker"] == slot_row[s]["worker"]
                if queue_edge and not include_queue:
                    continue
                cost = finish.get(pred, 0) + (edge_weight if w else 0)
                if cost > best:
                    best = cost
                    best_from = pred
            node = slot_row[s]["run_end"] - slot_row[s]["run_begin"]
            if node_publish:
                node += slot_row[s]["publish_end"] - slot_row[s]["run_end"]
            finish[s] = best + node
            parent[s] = best_from
        end = max(finish, key=lambda s: finish[s])
        path = []
        at = end
        while at is not None:
            path.append(at)
            at = parent[at]
        return finish[end], list(reversed(path))

    cp_ns, cp_path = longest(hop_p50, include_queue=True)
    out["cp_reconstructed_ns"] = cp_ns
    out["cp_reconstructed_ms"] = cp_ns / 1e6
    out["cp_nodes"] = len(cp_path)
    out["cp_error_vs_l2_ms"] = abs(cp_ns / 1e6 - l2_ms) / l2_ms

    # Exact decomposition of the measured stamps along that chain.
    task = sum(slot_row[s]["run_end"] - slot_row[s]["run_begin"] for s in cp_path)
    waiting = sum(slot_row[s]["ready"] - slot_row[s]["wait_begin"] for s in cp_path)
    prerun = sum(slot_row[s]["run_begin"] - slot_row[s]["ready"] for s in cp_path)
    pub = sum(slot_row[s]["publish_end"] - slot_row[s]["run_end"] for s in cp_path)
    gap = sum(slot_row[b]["wait_begin"] - slot_row[a]["publish_end"]
              for a, b in zip(cp_path, cp_path[1:]))
    out["cp_chain_span_ns"] = (slot_row[cp_path[-1]]["publish_end"]
                               - slot_row[cp_path[0]]["wait_begin"])
    out["cp_split_task_ns"] = task
    out["cp_split_wait_ns"] = waiting
    out["cp_split_prerun_barrier_ns"] = prerun
    out["cp_split_publish_ns"] = pub
    out["cp_split_gap_ns"] = gap

    # Diagnostic, not the gate.  §3.6 fixes the node weight at the run interval
    # and the edge weight at the hop, which leaves the producer's own notify
    # cost out of the reconstruction even though it sits on the chain.  Adding
    # the measured `publish_end - run_end` back shows how much of the D1-d
    # residual that single omitted term accounts for.
    pub_ns, pub_path = longest(hop_p50, include_queue=True, node_publish=True)
    out["cp_with_publish_ns"] = pub_ns
    out["cp_with_publish_ms"] = pub_ns / 1e6
    out["cp_with_publish_error"] = abs(pub_ns / 1e6 - l2_ms) / l2_ms

    # --- the four bounds ---------------------------------------------------
    total_run = sum(r["run_end"] - r["run_begin"] for r in slots)
    out["work_lb_ns"] = total_run / grid
    out["work_lb_ms"] = total_run / grid / 1e6
    out["queue_lb_ns"] = max(busy.values())
    out["queue_lb_ms"] = max(busy.values()) / 1e6
    nosync, _ = longest(0, include_queue=False)
    withsync, _ = longest(hop_p50, include_queue=False)
    out["cp_lb_nosync_ns"] = nosync
    out["cp_lb_nosync_ms"] = nosync / 1e6
    out["cp_lb_sync_ns"] = withsync
    out["cp_lb_sync_ms"] = withsync / 1e6
    out["hop_p50_used_ns"] = hop_p50

    # clock64 is per SM, so it is only subtracted within one task, where both
    # reads are on the same SM and no calibration is involved.
    same_sm = [r["run_end_clk"] - r["run_begin_clk"] for r in slots]
    out["run_clk_p50_cycles"] = percentile(same_sm, 0.50)
    out["run_clk_max_cycles"] = max(same_sm)
    quantized = sum(1 for r in slots if r["run_end"] == r["run_begin"])
    out["run_globaltimer_zero_slots"] = quantized
    out["run_globaltimer_zero_fraction"] = quantized / len(slots)

    for key in ("model", "fixture", "seq", "past", "placement", "schedule_policy",
                "stage_count", "slot_count", "event_count", "traced_launch",
                "globaltimer_resolution_ns", "commit", "l1_ms"):
        out[key] = meta.get(key, "unset")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="trace v2 dump directories")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [analyze(d) for d in args.dumps]
    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "analysis.tsv"), "w") as f:
        f.write("cell\t" + "\t".join(keys) + "\n")
        for d, row in zip(args.dumps, rows):
            f.write(os.path.basename(os.path.normpath(d)) + "\t"
                    + "\t".join(str(row[k]) for k in keys) + "\n")

    bad = [r for r in rows if r["hop_negative_count"]]
    over = [r for r in rows if r["cp_error_vs_l2_ms"] > 0.05]
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write("# EX-D1 §3.6 reconstruction\n\n")
        f.write("Generated by `analyze.py`; every number is a measurement from the\n")
        f.write("dump named in the first column of `analysis.tsv`.\n\n")
        f.write("`ready` is stamped after `WaitTaskDependencies` returns and therefore\n")
        f.write("includes that function's own `__syncthreads()` and `__threadfence()`.\n")
        f.write("The globaltimer tick is 1024 ns, so a single hop is quantized to that\n")
        f.write("step; the distribution over many hops is still informative, an individual\n")
        f.write("hop of one or two ticks is not.\n\n")
        f.write("| cell | l2_ms | span_ms | cp_ms | cp err | +publish err | hop p50 | hop p90 | hop max | hop<0 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for d, r in zip(args.dumps, rows):
            f.write("| {} | {:.4f} | {:.4f} | {:.4f} | {:.2%} | {:.2%} | {} | {} | {} | {} |\n".format(
                os.path.basename(os.path.normpath(d)), r["measured_l2_ms"],
                r["kernel_span_ms"], r["cp_reconstructed_ms"], r["cp_error_vs_l2_ms"],
                r["cp_with_publish_error"],
                r["hop_p50_ns"], r["hop_p90_ns"], r["hop_max_ns"],
                r["hop_negative_count"]))
        f.write("\n## Bounds\n\n")
        f.write("| cell | work_lb | queue_lb | cp_lb_nosync | cp_lb_sync | measured l2_ms |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d, r in zip(args.dumps, rows):
            f.write("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                os.path.basename(os.path.normpath(d)), r["work_lb_ms"], r["queue_lb_ms"],
                r["cp_lb_nosync_ms"], r["cp_lb_sync_ms"], r["measured_l2_ms"]))
        f.write("\n## Occupancy and head-of-line blocking\n\n")
        f.write("| cell | workers | busy max | idle frac | HOL reclaimable | HOL frac | workers w/ HOL |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for d, r in zip(args.dumps, rows):
            f.write("| {} | {} | {} | {:.2%} | {} | {:.2%} | {} |\n".format(
                os.path.basename(os.path.normpath(d)), r["workers_used"],
                r["busy_max_worker_ns"], r["idle_fraction_of_worker_time"],
                r["hol_reclaimable_ns"], r["hol_fraction_of_kernel"],
                r["hol_workers_nonzero"]))
        f.write("\n## Gates\n\n")
        f.write("- D1-e, hop(j) < 0 count is 0: **{}**\n".format(
            "PASS" if not bad else "FAIL on " + ", ".join(
                str(r["hop_negative_slots"]) for r in bad)))
        f.write("- D1-d, reconstruction within 5% of measured l2_ms: **{}**\n".format(
            "PASS" if not over else "FAIL"))
        if over:
            f.write("\n  Cells over the threshold, reported rather than relaxed:\n\n")
            for d, r in zip(args.dumps, rows):
                if r["cp_error_vs_l2_ms"] > 0.05:
                    f.write("  - `{}`: cp {:.4f} ms vs l2 {:.4f} ms, error {:.2%}.\n".format(
                        os.path.basename(os.path.normpath(d)),
                        r["cp_reconstructed_ms"], r["measured_l2_ms"],
                        r["cp_error_vs_l2_ms"]))
                    f.write("    Measured along that chain: task {} ns, wait {} ns,\n"
                            "    pre-run barrier {} ns, publish {} ns, gap {} ns,\n"
                            "    chain span {} ns.  Restoring only the publish term to the\n"
                            "    node weight moves the error to {:.2%}.\n".format(
                                r["cp_split_task_ns"], r["cp_split_wait_ns"],
                                r["cp_split_prerun_barrier_ns"], r["cp_split_publish_ns"],
                                r["cp_split_gap_ns"], r["cp_chain_span_ns"],
                                r["cp_with_publish_error"]))

    print("ANALYZE cells={} negative_hops={} cp_error_max={:.4f}".format(
        len(rows), sum(r["hop_negative_count"] for r in rows),
        max(r["cp_error_vs_l2_ms"] for r in rows)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
