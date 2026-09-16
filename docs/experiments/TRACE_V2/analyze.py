#!/usr/bin/env python3
"""Trace v2 analysis with the R1 formulas preserved in analyze_legacy.

R4 reconstruction requires the unsimplified generated runtime DAG and the
materialized window. All historical metrics are retained with _legacy suffixes.
"""
import argparse
import os
import sys
import re
import heapq
import hashlib
from pathlib import Path
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


def analyze_legacy(dump):
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


def dependency_graph(slots, source, variant=0):
    """Recover the unsimplified DAG from the generated runtime table.

    R1/R3 dumps contain offsets into this table, not the table itself. Wait
    lifting is irreversible: a missing wait does not identify its producer.
    Require the actual generated source and validate every dependency slice.
    The window expansion matches MaterializeRuntimeTaskGraph.
    """
    text = Path(source).read_text()
    if re.search(r'constexpr\s+RuntimeExactDependencyDesc', text):
        raise ValueError('exact ISL dependencies require a task_dag.tsv export')
    match = re.search(r'constexpr StageDependency kDependencies' + str(variant)
                      + r'\[\] = \{(.*?)\n\};', text, re.S)
    if not match:
        raise ValueError(f'{source}: no dependency table for variant {variant}')
    pattern = re.compile(r'\{(\d+)u,\s*(\d+)u,\s*StageDependency::Map::'
                         r'(kAll|kWindow|kIdentity),\s*(\d+)u,\s*(-?\d+),'
                         r'\s*(-?\d+),\s*(\d+)u\}')
    deps = []
    for line in match[1].splitlines():
        if not line.strip():
            continue
        m = pattern.search(line)
        if not m:
            raise ValueError(f'{source}: unsupported dependency row {line!r}')
        p, c, mode, div, scale, offset, count = m.groups()
        deps.append((int(p), int(c), mode, int(div), int(scale), int(offset), int(count)))
    ids = {(r['stage'], r['logical_task']): r['slot'] for r in slots}
    if len(ids) != len(slots):
        raise ValueError('a trace contains duplicate task identities')
    counts = defaultdict(int)
    for stage, task in ids:
        counts[stage] = max(counts[stage], task + 1)
    if sum(counts.values()) != len(slots):
        raise ValueError('task coordinates are not contiguous')
    incoming = {r['slot']: set() for r in slots}
    for r in slots:
        begin, count = r['dependency_begin'], r['dependency_count']
        selected = deps[begin:begin + count]
        expected = [d for d in deps if d[1] == r['stage']]
        if selected != expected or len(selected) != count:
            raise ValueError(f'{source}: trace dependency slice disagrees at {r["slot"]}')
        for p, c, mode, div, scale, offset, count in selected:
            if p >= c or div <= 0 or p not in counts:
                raise ValueError('invalid runtime dependency')
            at = (r['logical_task'] // div) * scale + offset
            lo = 0 if mode == 'kAll' else max(0, at)
            hi = counts[p] if mode == 'kAll' else min(counts[p], at + count)
            incoming[r['slot']].update(ids[p, t] for t in range(lo, hi))
    return incoming


def graph_longest(rows, incoming, weight, edge=lambda p, s: 0):
    """Topologically evaluate all edges; never treat an unseen parent as zero."""
    successors = defaultdict(list)
    degree = {}
    for s in rows:
        degree[s] = len(incoming[s])
        for p in incoming[s]:
            if p not in rows or p == s:
                raise ValueError('invalid predecessor')
            successors[p].append(s)
    ready = [s for s in rows if not degree[s]]
    heapq.heapify(ready)
    finish, parent, lengths = {}, {}, {}
    while ready:
        s = heapq.heappop(ready)
        choices = [(finish[p] + edge(p, s), p) for p in incoming[s]]
        # Keep zero-duration (timer-quantized) nodes on tied paths. Task
        # coordinates settle remaining ties without using worker assignment.
        best, pred = max(choices, key=lambda x: (x[0], lengths[x[1]], -rows[x[1]]['stage'],
                          -rows[x[1]]['logical_task'])) if choices else (0, None)
        finish[s], parent[s] = best + weight(s), pred
        lengths[s] = (lengths[pred] if pred is not None else 0) + 1
        for t in successors[s]:
            degree[t] -= 1
            if not degree[t]:
                heapq.heappush(ready, t)
    if len(finish) != len(rows):
        raise ValueError('dependency/queue graph is cyclic')
    last = max(finish, key=lambda s: (finish[s], lengths[s], -rows[s]['stage'], -rows[s]['logical_task']))
    path, at = [], last
    while at is not None:
        path.append(at)
        at = parent[at]
    return finish[last], path[::-1]


def analyze(dump, source=None, window=None, variant=0):
    legacy = analyze_legacy(dump)
    if source is None:
        return {**legacy, **{k + '_legacy': v for k, v in legacy.items()},
                'corrected_status': 'unavailable: runtime DAG source required'}
    if window not in (1, 2, 4):
        raise ValueError('supply the materialized Plan window (1, 2, or 4)')
    meta, slots, _, _ = load(dump)
    rows = {r['slot']: r for r in slots}
    if len(rows) != len(slots):
        raise ValueError('duplicate slot')
    incoming = dependency_graph(slots, source, variant)
    queues = defaultdict(list)
    for r in sorted(slots, key=lambda r: r['slot']):
        queues[r['worker']].append(r['slot'])
    run = lambda s: rows[s]['run_end'] - rows[s]['run_begin']
    cp, path = graph_longest(rows, incoming, run)
    sync_cp, _ = graph_longest(rows, incoming, run,
        lambda p, s: legacy['hop_p50_ns'] if rows[p]['worker'] != rows[s]['worker'] else 0)
    plan_edges = {s: set(ps) for s, ps in incoming.items()}
    for queue in queues.values():
        for j, s in enumerate(queue):
            # W>1 guarantees all i <= j-W, not adjacent sigma slots.
            plan_edges[s].update(queue[:max(0, j - window + 1)])
    plan_cp, _ = graph_longest(rows, plan_edges, run)
    # Actual execution introduces additional serialization, observable in the
    # task stamps even when it differs from sigma. This is a reconstruction,
    # never an extra constraint on the zero-sync task-DAG bound above.
    executed_edges = {s: set(ps) for s, ps in incoming.items()}
    for queue in queues.values():
        ordered = sorted(queue, key=lambda s: (rows[s]['run_begin'], rows[s]['run_end'], s))
        for p, s in zip(ordered, ordered[1:]):
            executed_edges[s].add(p)
    def elapsed(p, s):
        delay = rows[s]['run_begin'] - rows[p]['run_end']
        if delay < 0:
            raise ValueError(f'task dependency overlaps: {p}->{s}, {delay} ns')
        return delay
    reconstructed, realized_path = graph_longest(rows, executed_edges, run, elapsed)
    split = dict(task=0, wait=0, prerun_barrier=0, publish=0, gap=0)
    for s in realized_path:
        split['task'] += run(s)
    for p, s in zip(realized_path, realized_path[1:]):
        # Partition only the interval AFTER the predecessor completes. Summing
        # every consumer's entire wait double-counts concurrent waiting CTAs.
        cursor, end = rows[p]['run_end'], rows[s]['run_begin']
        for name, lo, hi in (
            ('publish', cursor, rows[p]['publish_end']),
            ('wait', rows[s]['wait_begin'], rows[s]['ready']),
            ('prerun_barrier', rows[s]['ready'], end),
        ):
            a, b = max(cursor, lo), min(end, hi)
            if b > a:
                split['gap'] += a - cursor
                split[name] += b - a
                cursor = b
        split['gap'] += end - cursor
    tail = rows[realized_path[-1]]['publish_end'] - rows[realized_path[-1]]['run_end']
    split['publish'] += tail
    reconstructed += tail
    if min(split.values()) < 0 or sum(split.values()) != reconstructed:
        raise ValueError('causal interval partition is not closed')
    out = dict(legacy)
    out.update({k + '_legacy': v for k, v in legacy.items()})
    out.update(corrected_status='available', task_dag_source=str(source),
               task_dag_source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
               window=window, cp_corrected_ns=cp, cp_corrected_ms=cp / 1e6,
               cp_lb_nosync_ns=cp, cp_lb_nosync_ms=cp / 1e6,
               cp_lb_sync_ns=sync_cp, cp_lb_sync_ms=sync_cp / 1e6,
               cp_corrected_nodes=len(path), cp_corrected_task_ns=sum(run(s) for s in path),
               cp_plan_nosync_ns=plan_cp,
               cp_corrected_path=','.join(f'{rows[s]["stage"]}:{rows[s]["logical_task"]}' for s in path),
               cp_realized_path=','.join(f'{rows[s]["stage"]}:{rows[s]["logical_task"]}' for s in realized_path),
               dag_same_worker_edges=sum(rows[p]['worker'] == rows[s]['worker'] for s in incoming for p in incoming[s]),
               dag_cross_worker_edges=sum(rows[p]['worker'] != rows[s]['worker'] for s in incoming for p in incoming[s]),
               cp_reconstructed_ns=reconstructed, cp_reconstructed_ms=reconstructed / 1e6,
               cp_nodes=len(realized_path), cp_chain_span_ns=reconstructed)
    out.update({'cp_split_' + k + '_ns': v for k, v in split.items()})
    out['cp_error_vs_l2_ms'] = abs(reconstructed / 1e6 - float(meta['l2_ms'])) / float(meta['l2_ms'])
    out['cp_with_publish_ns'] = reconstructed
    out['cp_with_publish_ms'] = reconstructed / 1e6
    out['cp_with_publish_error'] = out['cp_error_vs_l2_ms']
    edges = out['dag_same_worker_edges'] + out['dag_cross_worker_edges']
    out['dag_cross_worker_fraction'] = out['dag_cross_worker_edges'] / edges if edges else 0
    # DAG-ready instants include dependencies whose polls were lifted away.
    # Count actual head stalls once per worker, only when an eligible later
    # slot is dependency-ready; for W>1 this is not a FIFO-only estimate.
    start = min(r['wait_begin'] for r in slots)
    dag_ready_at = {s: max((rows[p]['run_end'] for p in incoming[s]), default=start)
                    for s in rows}
    hol = 0
    hol_workers = []
    for queue in queues.values():
        worker_hol = 0
        actual = sorted(queue, key=lambda s: (rows[s]['run_begin'], rows[s]['run_end'], s))
        done, previous = set(), start
        for s in actual:
            lo, hi = max(previous, rows[s]['wait_begin']), rows[s]['ready']
            head = next((j for j, t in enumerate(queue) if t not in done), len(queue))
            # Reclaimability asks whether opening the queue could expose work;
            # report all remaining slots, as the legacy HOL definition does.
            alternatives = [t for t in queue[head:] if t not in done and t != s]
            ready_at = [dag_ready_at[t] for t in alternatives]
            if ready_at:
                worker_hol += max(0, hi - max(lo, min(ready_at)))
            done.add(s)
            previous = rows[s]['publish_end']
        hol_workers.append(worker_hol)
        hol += worker_hol
    out['hol_reclaimable_ns'] = hol
    out['hol_fraction_of_kernel'] = hol / (out['kernel_span_ns'] * len(queues))
    out['hol_worker_p50_ns'] = percentile(hol_workers, .5)
    out['hol_worker_p90_ns'] = percentile(hol_workers, .9)
    out['hol_worker_max_ns'] = max(hol_workers)
    out['hol_workers_nonzero'] = sum(v > 0 for v in hol_workers)
    return out


def analyze_phases(dump, source, window=1, variant=0):
    """Phase totals on the corrected semantic DAG path; no legacy columns change.

    Absolute times use the original globaltimer stamps. clock64 supplies a
    separate within-CTA share estimate, never an invented inter-SM clock.
    Epilogue includes the separately reported return-to-run_end harness tail.
    """
    result = analyze(dump, source, window, variant)
    _, slots, _, _ = load(dump)
    phases = read_tsv(Path(dump) / 'phases.tsv')
    by_slot = {r['slot']: r for r in slots}
    path = {tuple(map(int, x.split(':'))) for x in result['cp_corrected_path'].split(',')}
    names = ('run_begin', 'setup_end', 'first_operand_ready', 'mainloop_end', 'epilogue_end', 'run_end')
    rows = []
    for item in phases:
        r = {k: int(v) for k, v in item.items()}
        slot = by_slot[r['slot']]
        out = dict(slot=r['slot'], stage=slot['stage'], logical_task=slot['logical_task'],
                   kind=r['kind'], on_cp=int((slot['stage'], slot['logical_task']) in path),
                   tile_m=r['tile_m'], tile_n=r['tile_n'], tile_k=r['tile_k'],
                   split_k=r['split_k'], operand_bytes=r['operand_bytes'])
        for unit in ('ns', 'cycles'):
            stamps = [r[n+'_'+unit] for n in names]
            if not stamps[0] or any(b<a for a,b in zip(stamps,stamps[1:])):
                raise ValueError(f"nonmonotonic/missing phase: slot {r['slot']} {unit} {stamps}")
            for name, lo, hi in [('setup',0,1),('load_wait',1,2),('mainloop',2,3),('epilogue',3,5),
                                 ('body_epilogue',3,4),('harness_tail',4,5),('run',0,5)]:
                out[name+'_'+unit]=stamps[hi]-stamps[lo]
        if out['run_ns'] != slot['run_end']-slot['run_begin']:
            raise ValueError(f"phase and slot clocks disagree: {r['slot']}")
        rows.append(out)
    if len(rows)!=len(slots): raise ValueError('phase slot coverage')
    for label, group in [('all',rows),('cp',[r for r in rows if r['on_cp']])]:
        for unit in ('ns','cycles'):
            total=sum(r['run_'+unit] for r in group)
            result[f'phase_{label}_run_{unit}']=total
            for name in ('setup','load_wait','mainloop','epilogue','body_epilogue','harness_tail'):
                v=sum(r[name+'_'+unit] for r in group)
                result[f'phase_{label}_{name}_{unit}']=v
                result[f'phase_{label}_{name}_share_{unit}']=v/total if total else 0
    return result, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="trace v2 dump directories")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--source", help="generated source containing the unsimplified runtime DAG")
    ap.add_argument("--window", type=int, choices=(1, 2, 4))
    ap.add_argument("--variant", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [analyze_phases(d, args.source, args.window, args.variant)[0]
            if args.source and (Path(d) / 'phases.tsv').exists()
            else analyze(d, args.source, args.window, args.variant) for d in args.dumps]
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
        if args.source:
            f.write("R4 mode: runtime-DAG dependencies retain same-worker edges. "
                    "The reconstruction partitions non-overlapping causal intervals; "
                    "all R1 metrics remain in the TSV with `_legacy` suffixes.\n\n")
            f.write("| dump | legacy zero-sync bound (ms) | corrected (ms) | DAG nodes |\n")
            f.write("|---|---:|---:|---:|\n")
            for d, r in zip(args.dumps, rows):
                f.write(f"| {d} | {r['cp_lb_nosync_ms_legacy']:.6f} | "
                        f"{r['cp_corrected_ms']:.6f} | {r['cp_corrected_nodes']} |\n")
        else:
            f.write("Legacy-only mode: no unsimplified runtime DAG was supplied. "
                    "These results do not establish the R4 corrected gates.\n\n")
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
