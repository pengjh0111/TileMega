"""R14 selection primitives; execution is delegated to the guarded caller."""
import math
import statistics

PASTS=(64,575,1000)

class SelectionBudgetExhausted(Exception):
    """Stop admitting pilot work; confirm previously measured finalists."""

def admit_pilot(candidate, label, now, deadline):
    """Already built winners still need a measured execution baseline.

    Only new structural variants consume admission budget. Once those stop,
    halving confirms the execution alternatives of the existing plans.
    """
    if label.startswith('pilot') and now>=deadline and not candidate.get('base_variant'):
        raise SelectionBudgetExhausted('selection budget exhausted before new pilot')

def first_stage_budget_ms(seconds, pg_count, decode):
    """Reserve two thirds of decode time for compilation and integral pilots."""
    return max(1,int(1000*seconds/(3*pg_count))) if decode else max(1,int(1000*seconds)-200000)


def integrated_ms(by_past):
    """Uniform discrete request trajectory; linear interior, held endpoints."""
    samples=[float(by_past[str(p)]['mean_ms']) for p in PASTS]
    if any(not math.isfinite(x) or x<=0 for x in samples):
        raise ValueError('selection needs positive finite measurements at all three pasts')
    total=0.
    for past in range(64,1088):
        if past>=1000:value=samples[2]
        else:
            i=0 if past<575 else 1
            weight=(past-PASTS[i])/(PASTS[i+1]-PASTS[i])
            value=samples[i]*(1-weight)+samples[i+1]*weight
        total+=value
    return total/1024

def successive_halving(candidates,measure):
    """Keep all evidence; finalists get three fresh alternating paired rounds.

    ``measure(candidate, label)`` must run through the caller's GPU guard and
    return by_past plus execution_identity. Exit 75 propagates unchanged.
    """
    rows=[dict(c,measurements=[],eliminated_round=None) for c in candidates]
    if not rows:raise ValueError('empty candidate set')
    active=list(range(len(rows)));round_index=0
    def observe(i,label):
        try:r=measure(rows[i],label)
        except RuntimeError as error:
            rows[i]['error']=str(error);rows[i]['eliminated_round']=label
            return math.inf
        if not r.get('execution_identity'):raise ValueError('unidentified measurement')
        if r['execution_identity'].get('trace'):raise ValueError('trace timing cannot enter selection')
        old=rows[i]['measurements']
        if old and old[0]['execution_identity']!=r['execution_identity']:
            raise ValueError('candidate execution identity changed across rounds')
        value=integrated_ms(r['by_past']);rows[i]['measurements'].append(dict(round=label,score_ms=value,**r))
        return value
    while len(active)>3:
        scores={};exhausted=False
        order=active[round_index%len(active):]+active[:round_index%len(active)]
        for pos,i in enumerate(order):
            try:scores[i]=observe(i,f'pilot{round_index}')
            except SelectionBudgetExhausted:
                exhausted=True
                for pending in order[pos:]:
                    old=rows[pending]['measurements']
                    if old:scores[pending]=old[-1]['score_ms']
                    else:rows[pending]['eliminated_round']='budget_unmeasured'
                break
        ranked=sorted((i for i in active if i in scores and not rows[i].get('error')),key=lambda i:(scores[i],i))
        if not ranked:raise RuntimeError('all serving execution candidates rejected')
        keep=3 if exhausted else max(3,(len(ranked)+1)//2)
        for i in ranked[keep:]:rows[i]['eliminated_round']=f'pilot{round_index}'
        active=ranked[:keep];round_index+=1
    for r in range(3):
        active=[i for i in active if not rows[i].get('error')]
        if not active:raise RuntimeError('all serving execution candidates rejected')
        shift=r%len(active)
        for i in active[shift:]+active[:shift]:observe(i,f'final{r}')
    active=[i for i in active if not rows[i].get('error')]
    if not active:raise RuntimeError('all serving execution candidates rejected')
    for i in active:
        rows[i]['samples_ms']=[r['score_ms'] for r in rows[i]['measurements'] if r['round'].startswith('final')]
        rows[i]['median_ms']=statistics.median(rows[i]['samples_ms'])
    winner=min(active,key=lambda i:(rows[i]['median_ms'],i))
    return rows[winner],rows
