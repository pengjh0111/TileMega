#!/usr/bin/env python3
"""Where a decode step's time goes, per stage (MX3-1 item 2).

Reads the runtime's own stage_trace.tsv (and task_trace.tsv when present) and
splits each stage's envelope on the critical CTA -- the last one to arrive --
into the terms MX3-4 is ordered by:

  bound      the stage's bandwidth lower bound, bytes / 1.61 TB/s
  inefficiency  mean task body time above that bound
  imbalance  critical CTA's task time minus the mean across CTAs with tasks
  startup    stage open to the first task's first byte landing (first_ready)
  gaps       time between tasks on the critical CTA
  epilogue   epilogue_ns summed on the critical CTA
  barrier    last task end to the stage's release

Every term is measured, not modelled; the residual is reported rather than
distributed, so a term that does not add up is visible.
"""
import argparse, csv, collections
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('trace', type=Path, help='directory holding stage_trace.tsv')
ap.add_argument('--label', required=True, help='cell label, e.g. qwen3_0_6b/B1')
ap.add_argument('--campaign', default='tm-mx2')
ap.add_argument('--peak-bytes-per-s', type=float, default=1.61e12)
ap.add_argument('--out', type=Path)
a = ap.parse_args()


def read(path):
    if not path.exists():
        return []
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    return [{k: int(v) if v and v.lstrip('-').isdigit() else v for k, v in r.items()}
            for r in rows]


stages = read(a.trace / 'stage_trace.tsv')
tasks = read(a.trace / 'task_trace.tsv')
if not stages:
    raise SystemExit('no stage_trace.tsv in %s' % a.trace)

# One iteration only: the middle past, so KV length is representative and the
# pipeline is warm. Pick the modal iteration with the most complete rows.
by_iteration = collections.Counter(r['iteration'] for r in stages if r.get('t_release'))
iteration = by_iteration.most_common(1)[0][0]
rows = [r for r in stages if r['iteration'] == iteration and r.get('t_release')]

tasks_by_stage = collections.defaultdict(list)
for t in tasks:
    if t.get('iteration') == iteration:
        tasks_by_stage[t.get('stage')].append(t)

ledger = []
for stage in sorted({r['stage'] for r in rows}):
    group = [r for r in rows if r['stage'] == stage]
    active = [r for r in group if r.get('tasks')]
    if not active:
        continue
    begin = min(r['t_begin'] for r in group)
    release = max(r['t_release'] for r in group)
    envelope = release - begin
    # The critical CTA is the last to finish its tasks.
    critical = max(active, key=lambda r: r['t_tasks_end'])
    body = [r['t_tasks_end'] - r['t_begin'] for r in active]
    mean_body = sum(body) / len(body)
    critical_body = critical['t_tasks_end'] - critical['t_begin']
    mine = tasks_by_stage.get(stage, [])
    on_critical = [t for t in mine if t.get('worker') == critical.get('worker')]
    first_ready = min((t['first_ready'] - t['run_begin'] for t in on_critical
                       if t.get('first_ready') and t.get('run_begin')), default=0)
    epilogue = sum(t.get('epilogue_ns', 0) for t in on_critical)
    run = sorted((t['run_begin'], t['run_end']) for t in on_critical
                 if t.get('run_begin') and t.get('run_end'))
    gaps = sum(max(0, b - a_end) for (_, a_end), (b, _) in zip(run, run[1:]))
    byte_total = sum(t.get('bytes', 0) for t in mine)
    bound = byte_total / a.peak_bytes_per_s * 1e9 if byte_total else 0.0
    barrier = release - critical['t_tasks_end']
    accounted = bound + max(0.0, mean_body - bound) + (critical_body - mean_body) \
        + first_ready + gaps + epilogue + barrier
    ledger.append(dict(
        campaign=a.campaign, cell=a.label, stage=stage,
        ctas_with_tasks=len(active), tasks=sum(r['tasks'] for r in active),
        envelope_ns=envelope, bytes=byte_total,
        bound_ns=round(bound, 1),
        inefficiency_ns=round(max(0.0, mean_body - bound), 1),
        imbalance_ns=round(critical_body - mean_body, 1),
        startup_ns=first_ready, gaps_ns=gaps, epilogue_ns=epilogue,
        barrier_ns=barrier,
        residual_ns=round(envelope - accounted, 1)))

total = {k: sum(r[k] for r in ledger) for k in
         ('envelope_ns', 'bound_ns', 'inefficiency_ns', 'imbalance_ns',
          'startup_ns', 'gaps_ns', 'epilogue_ns', 'barrier_ns', 'residual_ns')}
print('%s %s: %d stages, step envelope %.3f us' %
      (a.campaign, a.label, len(ledger), total['envelope_ns'] / 1e3))
over = total['envelope_ns'] - total['bound_ns']
print('  bandwidth bound      %9.3f us' % (total['bound_ns'] / 1e3))
print('  over the bound       %9.3f us  = %.1fx the bound' %
      (over / 1e3, total['envelope_ns'] / total['bound_ns'] if total['bound_ns'] else float('nan')))
for term in ('inefficiency_ns', 'imbalance_ns', 'startup_ns', 'gaps_ns',
             'epilogue_ns', 'barrier_ns', 'residual_ns'):
    share = 100 * total[term] / over if over else 0
    print('    %-16s %9.3f us  %5.1f%% of the excess' % (term[:-3], total[term] / 1e3, share))

if a.out:
    new = not a.out.exists()
    with a.out.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        if new:
            writer.writeheader()
        writer.writerows(ledger)
    print('-> %s (%d rows)' % (a.out, len(ledger)))
