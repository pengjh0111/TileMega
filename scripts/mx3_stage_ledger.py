#!/usr/bin/env python3
"""Where a decode step's time goes, per stage (MX3-1 items 2 and 3).

Reads one `tilemega.serving.trace --stage --task` directory: the runtime's
own stage_trace.tsv (one row per worker, stage and launch), task_profile.tsv
(one row per task) and runtime_stages.tsv (what each stage is). Timestamps are
the 100 MHz device realtime counter in ns; every number below is measured.

For each stage and launch, with `open` the earliest worker to enter it and the
critical CTA the last to finish its tasks, the envelope splits exactly as

  envelope = bound + inefficiency + imbalance + barrier

  bound        bytes the stage must move / 1.61 TB/s
  inefficiency mean task span over active CTAs, minus the bound
  imbalance    critical CTA's span minus that mean
  barrier      critical CTA's last task end to the stage's release

and, separately, the critical CTA's span splits as

  span = lead + dispatch + first_byte + body + epilogue + gaps + tail

  lead         stage open to this CTA entering it (it arrived late)
  dispatch     entering to its first task starting
  first_byte   first task start to its first operand tile landing
  body         task time after the first byte, epilogue excluded
  epilogue     epilogue_ns as the task body measured it
  gaps         time between consecutive tasks
  tail         last task end to the CTA marking its tasks done

Each stage's terms are medians over launches; the per-class and per-step
totals sum those medians. Inter-stage time (one stage's release to the next
stage's open) is reported on its own, so nothing is folded into a term it
does not belong to.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('trace', type=Path, help='a past<N> directory from tilemega.serving.trace')
ap.add_argument('--cell', required=True, help='label, e.g. qwen3_0_6b/B1')
ap.add_argument('--campaign', required=True, help='tm-mx2, mx3-pipe, mx3-final ...')
ap.add_argument('--kv-bytes-per-layer', type=float, default=0.0,
                help='KV bytes one attention stage reads at this past, all batches')
ap.add_argument('--peak-bytes-per-s', type=float, default=1.61e12)
ap.add_argument('--out', type=Path, help='append per-stage rows to this CSV')
ap.add_argument('--summary', type=Path, help='append per-class rows to this CSV')
a = ap.parse_args()

meta = pd.read_csv(a.trace / 'runtime_stages.tsv', sep='\t')
st = pd.read_csv(a.trace / 'stage_trace.tsv', sep='\t')
tk = pd.read_csv(a.trace / 'task_profile.tsv', sep='\t')
st = st[st.t_release > 0]


def stage_class(row):
    kind = row.kind_name
    name = row['name'] if isinstance(row['name'], str) else ''
    for tag, label in (('qkv', 'qkv'), ('o_proj', 'o'), ('gate_up', 'gate_up'),
                       ('down_proj', 'down'), ('lm_head', 'lm_head')):
        if tag in name:
            return label + ('_combine' if kind == 'kGemmCombine' else '')
    return {'kFusedAttention': 'attention', 'kAttentionMerge': 'merge',
            'kEmbedding': 'embedding', 'kArgmaxReduce': 'argmax'}.get(kind, kind)


meta['cls'] = meta.apply(stage_class, axis=1)
meta['bytes'] = meta.weight_bytes.astype(float)
meta.loc[meta.cls == 'attention', 'bytes'] += a.kv_bytes_per_layer

key = ['iteration', 'stage']
opens = st.groupby(key).t_begin.min().rename('open')
release = st.groupby(key).t_release.max().rename('release')
frame = pd.concat([opens, release], axis=1).reset_index()

active = st[st.tasks > 0].merge(frame[key + ['open']], on=key)
active['span'] = active.t_tasks_end - active.open
mean_span = active.groupby(key).span.mean().rename('mean_span')
ctas = active.groupby(key).worker.nunique().rename('ctas')
ntasks = active.groupby(key).tasks.sum().rename('ntasks')
crit = active.loc[active.groupby(key).span.idxmax(),
                  key + ['worker', 't_begin', 't_tasks_end', 'span']]
crit = crit.rename(columns={'worker': 'crit_worker', 't_begin': 'crit_begin',
                            't_tasks_end': 'crit_end', 'span': 'crit_span'})

tk = tk.merge(crit[key + ['crit_worker']], on=key)
on_crit = tk[tk.worker == tk.crit_worker].sort_values(key + ['run_begin'])
on_crit = on_crit.assign(busy=on_crit.run_end - on_crit.run_begin)
on_crit['prev_end'] = on_crit.groupby(key).run_end.shift(1)
on_crit['gap'] = (on_crit.run_begin - on_crit.prev_end).clip(lower=0).fillna(0)
g = on_crit.groupby(key)
first = g.first()
crit_terms = pd.DataFrame({
    'first_run_begin': first.run_begin,
    'first_byte': (first.first_ready - first.run_begin).clip(lower=0),
    'last_run_end': g.run_end.max(),
    'busy': g.busy.sum(),
    'epilogue': g.epilogue_ns.sum(),
    'gaps': g.gap.sum(),
}).reset_index()

df = (frame.merge(mean_span.reset_index(), on=key, how='left')
      .merge(ctas.reset_index(), on=key, how='left')
      .merge(ntasks.reset_index(), on=key, how='left')
      .merge(crit, on=key, how='left')
      .merge(crit_terms, on=key, how='left')
      .merge(meta[['stage', 'cls', 'bytes']], on='stage'))
df['envelope'] = df.release - df.open
df['bound'] = df.bytes / a.peak_bytes_per_s * 1e9
df['inefficiency'] = df.mean_span - df.bound
df['imbalance'] = df.crit_span - df.mean_span
df['barrier'] = df.release - df.crit_end
df['lead'] = df.crit_begin - df.open
df['dispatch'] = df.first_run_begin - df.crit_begin
df['body'] = df.busy - df.epilogue - df.first_byte
df['tail'] = df.crit_end - df.last_run_end
# A stage where no CTA ran a task carries its whole envelope as barrier.
idle = df.crit_span.isna()
for t in ('inefficiency', 'imbalance', 'lead', 'dispatch', 'first_byte', 'body',
          'epilogue', 'gaps', 'tail', 'ctas', 'ntasks'):
    df.loc[idle, t] = 0.0
df.loc[idle, 'barrier'] = df.loc[idle, 'envelope'] - df.loc[idle, 'bound']

df = df.sort_values(key)
df['to_next'] = (df.groupby('iteration').open.shift(-1) - df.release).clip(lower=0).fillna(0)

terms = ['envelope', 'bound', 'inefficiency', 'imbalance', 'barrier', 'lead', 'dispatch',
         'first_byte', 'body', 'epilogue', 'gaps', 'tail', 'to_next', 'ctas', 'ntasks']
per_stage = df.groupby(['stage', 'cls'])[terms].median().reset_index()
step = df.groupby('iteration').release.max() - df.groupby('iteration').open.min()

cls = per_stage.groupby('cls')[terms[:-2]].sum()
cls['stages'] = per_stage.groupby('cls').stage.count()
cls['ctas'] = per_stage.groupby('cls').ctas.median()
cls = cls.sort_values('envelope', ascending=False)
tot = cls.sum()
over = tot.envelope - tot.bound

print('%s %s: %d stages, %d launches, traced step %.3f ms (median)' % (
    a.campaign, a.cell, per_stage.stage.nunique(), df.iteration.nunique(), step.median() / 1e6))
print('  sum of envelopes %.3f ms, bound %.3f ms, excess %.3f ms (%.1fx); inter-stage %.3f ms'
      % (tot.envelope / 1e6, tot.bound / 1e6, over / 1e6, tot.envelope / tot.bound,
         tot.to_next / 1e6))
print('  excess = inefficiency %.1f%% + imbalance %.1f%% + barrier %.1f%%' % tuple(
    100 * tot[t] / over for t in ('inefficiency', 'imbalance', 'barrier')))
print('  critical span = lead %.3f + dispatch %.3f + first_byte %.3f + body %.3f + '
      'epilogue %.3f + gaps %.3f + tail %.3f ms' % tuple(
          tot[t] / 1e6 for t in ('lead', 'dispatch', 'first_byte', 'body', 'epilogue', 'gaps', 'tail')))
print('  %-14s %4s %5s %8s %7s %7s %7s %7s %7s %7s %7s' % (
    'class', 'n', 'ctas', 'envelope', 'bound', 'ineff', 'imbal', 'barrier', 'firstB',
    'epilog', 'gaps'))
for name, r in cls.iterrows():
    print('  %-14s %4d %5.0f %8.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f' % (
        name, r.stages, r.ctas, r.envelope / 1e6, r.bound / 1e6, r.inefficiency / 1e6,
        r.imbalance / 1e6, r.barrier / 1e6, r.first_byte / 1e6, r.epilogue / 1e6,
        r.gaps / 1e6))

if a.out:
    out = per_stage.assign(campaign=a.campaign, cell=a.cell)
    out.to_csv(a.out, mode='a', index=False, header=not a.out.exists(), float_format='%.1f')
if a.summary:
    s = cls.reset_index().assign(campaign=a.campaign, cell=a.cell,
                                 traced_step_ms=round(step.median() / 1e6, 4))
    s.to_csv(a.summary, mode='a', index=False, header=not a.summary.exists(), float_format='%.1f')
