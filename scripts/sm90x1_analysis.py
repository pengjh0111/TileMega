"""SM90X1 trace diagnostics and conservative comparisons (no GPU imports)."""
import csv
import itertools
import math
from pathlib import Path
import statistics as st

from sm90d1_common import read
from sm90r4_ledger import kind, tsv


def span(values):
    return 100*(max(values)-min(values))/st.median(values)


def threshold(a, b, field):
    return max(0.5, 2*max(a[field+'_range_pct'], b[field+'_range_pct']))


def compare(a, b, field):
    change = 100*(a[field]/b[field]-1)
    bound = threshold(a, b, field)
    return change, 'tie' if abs(change) <= bound else 'faster' if change < 0 else 'slower'


def criterion_order(rows, field):
    ranked = sorted(rows, key=lambda r: (r[field], r['variant']))
    reference = next(r for r in rows if r['reference'])
    tied = [r for r in ranked if compare(r, ranked[0], field)[1]=='tie']
    # The prompt resolves an uncertain choice to W, including a top tie between two alternatives.
    if len(tied)>1 or compare(ranked[0], reference, field)[1]=='tie':
        ranked = [reference] + [r for r in ranked if not r['reference']]
    return ranked


def trace_process(path, moe):
    stages = tsv(Path(path) / 'runtime_stages.tsv')
    descriptors = {int(s['stage']): s for s in stages}
    rows = tsv(Path(path) / 'stage_trace.tsv')
    groups = {}
    for r in rows:
        if 567<=int(r['past'])<=582:
            groups.setdefault((int(r['past']), int(r['stage'])), []).append(r)
    own, ledger, steps = {}, {}, []
    for past in range(567,583):
        present = sorted(g for p,g in groups if p==past)
        expected = sorted(descriptors)
        if present != expected and not (descriptors[expected[0]]['kind_name']=='kEmbedding' and present==expected[1:]):
            raise ValueError(f'incomplete trace window at past {past}: {path}')
        release = {g:min(int(r['t_release']) for r in groups[past,g]) for g in present}
        begin = min(int(r['t_begin']) for r in groups[past,present[0]])
        for index,g in enumerate(present):
            latest = max(groups[past,g],key=lambda r:(int(r['t_tasks_end']),-int(r['worker'])))
            duration = (int(latest['t_tasks_end'])-int(latest['t_begin']))/1000
            interval = (release[g]-(release[present[index-1]] if index else begin))/1000
            if duration<0 or interval<0:
                raise ValueError('nonmonotone trace stamps')
            own.setdefault(g,[]).append(duration)
            ledger.setdefault(g,[]).append(interval)
        steps.append((release[present[-1]]-begin)/1000)
    by_family = {}
    stage_rows = []
    for g in sorted(own):
        s = descriptors[g]
        family = kind(s,moe)
        if moe and family in ('expert_gate_up','expert_down'):
            family = 'experts'
        by_family[family] = by_family.get(family,0)+st.median(own[g])
        stage_rows.append(dict(stage=g, stage_kind=family, stage_name=s['name'],
                               own_us=st.median(own[g]), ledger_us=st.median(ledger[g])))
    return dict(own=by_family, step_us=st.median(steps), stages=stage_rows, steps=16)


def aggregate(unit, family, moe):
    record = read(Path(unit) / 'unit.json')
    processes = [trace_process(p,moe) for p in record['runs']]
    own = [p['own'][family] for p in processes]
    step = [p['step_us'] for p in processes]
    stages = []
    for index,s in enumerate(processes[0]['stages']):
        if not all(p['stages'][index]['stage']==s['stage'] for p in processes):
            raise ValueError('process trace schemas differ')
        stages.append(dict(s, ledger_us=st.median(p['stages'][index]['ledger_us'] for p in processes)))
    return dict(own_us=st.median(own), step_us=st.median(step),
                own_us_range_pct=span(own), step_us_range_pct=span(step),
                own_process_us=own, step_process_us=step, processes=len(processes), stages=stages)


def kendall(rows):
    c=d=tx=ty=0
    for a,b in itertools.combinations(rows,2):
        x=(a['own_us']>b['own_us'])-(a['own_us']<b['own_us'])
        y=(a['step_us']>b['step_us'])-(a['step_us']<b['step_us'])
        if x and y:
            if x==y:c+=1
            else:d+=1
        elif y:tx+=1
        elif x:ty+=1
    denominator=math.sqrt((c+d+tx)*(c+d+ty))
    return (c-d)/denominator if denominator else None


def coupled_share(rows):
    own=sum(abs(r['delta_own_us']) for r in rows)
    coupled=sum(abs(r['delta_coupled_us']) for r in rows)
    return coupled/(own+coupled) if own+coupled else None


def reversals(rows):
    out=[]
    for a,b in itertools.combinations(rows,2):
        own,oc=compare(a,b,'own_us');step,sc=compare(a,b,'step_us')
        if {oc,sc}=={'faster','slower'}:
            out.append(dict(a=a['variant'],b=b['variant'],own_a_us=a['own_us'],own_b_us=b['own_us'],
                            step_a_us=a['step_us'],step_b_us=b['step_us'],strength_pct=min(abs(own),abs(step))))
    return sorted(out,key=lambda r:-r['strength_pct'])


def write_csv(path, rows, fields):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
