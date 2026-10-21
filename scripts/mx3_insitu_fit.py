#!/usr/bin/env python3
"""MX3-5: in-situ costs the isolated calibrators cannot see, fitted from
diagnostic plans of the megakernel itself (TILEMEGA_DIAG_EMPTY_KINDS, MX3-1).

For each reference cell, three plans differ only in a diagnostic mask:
  base      the plan as built;
  empty_all every stage keeps its barrier and runs no task;
  empty_c   split-K combine stages keep their barrier and run no task.
Then, per cell,
  stage skeleton (barrier, stage loop, descriptor prefetch) = empty_all / stages
  combine task                                            = (base - empty_c) / combine stages
and the target gets their mean as
  serving_hop[0]           so that hop(1, 1) is the skeleton, and
  task_body.serving.gemm_combine.fixed_ns (TaskModel prices serving combines with it).
Every input TPOT is the median of the plan's measured rounds.

Usage: mx3_insitu_fit.py --base TARGET --out TARGET --cell model:B:stages:combines:diag_root ...
"""
import argparse
import glob
import json
import statistics
from pathlib import Path


def tpot(path):
    values = [json.load(open(f))['tpot_p50_seconds'] * 1e9
              for f in glob.glob(str(path / 'measure/round*/measurements.json'))]
    if not values:
        raise SystemExit('no measurement under %s' % path)
    return statistics.median(values)


ap = argparse.ArgumentParser()
ap.add_argument('--base', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--cell', action='append', required=True,
                help='model:batch:stages:combine_stages:diag_root (diag_root/<plan>/<model>/B<batch>)')
ap.add_argument('--plans', default='base3,emptyAll3,emptyC3')
a = ap.parse_args()
base_name, all_name, combine_name = a.plans.split(',')

skeleton, combine, evidence = [], [], []
for spec in a.cell:
    model, batch, stages, combines, root = spec.split(':', 4)
    cell = Path(model) / ('B' + batch)
    base = tpot(Path(root) / base_name / cell)
    empty_all = tpot(Path(root) / all_name / cell)
    empty_c = tpot(Path(root) / combine_name / cell)
    skeleton.append(empty_all / int(stages))
    combine.append((base - empty_c) / int(combines))
    evidence.append(dict(cell='%s/B%s' % (model, batch), base_ns=round(base), empty_all_ns=round(empty_all),
                         empty_combine_ns=round(empty_c), stages=int(stages), combine_stages=int(combines),
                         skeleton_ns=round(skeleton[-1], 1), combine_task_ns=round(combine[-1], 1)))
    print('%s/B%s: skeleton %.0f ns/stage, combine task %.0f ns' % (model, batch, skeleton[-1], combine[-1]))

target = json.load(open(a.base))
hop = target['serving_hop']
s = statistics.mean(skeleton)
c = statistics.mean(combine)
spread = lambda v: max(abs(x / statistics.mean(v) - 1) for x in v)
hop_old = list(hop)
hop[0] = s - hop[1] - hop[2]
serving = target['calibration_by_dtype']['bf16']['task_body']['serving']
serving['gemm_combine'] = dict(fixed_ns=c, byte_ns=0.0, flop_ns=0.0,
                               median_relative_error=spread(combine), samples=len(combine))
target['insitu_mx3'] = dict(
    method='TILEMEGA_DIAG_EMPTY_KINDS plans of the measured executor (L1), scripts/mx3_insitu_fit.py',
    serving_hop_before=hop_old, skeleton_ns=s, combine_task_ns=c, cells=evidence)
a.out.write_text(json.dumps(target, indent=2) + '\n')
print('serving_hop %s -> %s; gemm_combine fixed %.0f ns -> %s' % (hop_old, hop, c, a.out))
