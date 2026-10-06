"""R14 selection primitives; execution is delegated to the guarded caller."""
import math
import statistics

PASTS=(64,575,1000)

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
        r=measure(rows[i],label)
        if not r.get('execution_identity'):raise ValueError('unidentified measurement')
        value=integrated_ms(r['by_past']);rows[i]['measurements'].append(dict(round=label,score_ms=value,**r))
        return value
    while len(active)>3:
        scores={i:observe(i,f'pilot{round_index}') for i in active[round_index%len(active):]+active[:round_index%len(active)]}
        ranked=sorted(active,key=lambda i:(scores[i],i));keep=(len(ranked)+1)//2
        for i in ranked[keep:]:rows[i]['eliminated_round']=f'pilot{round_index}'
        active=ranked[:keep];round_index+=1
    for r in range(3):
        shift=r%len(active)
        for i in active[shift:]+active[:shift]:observe(i,f'final{r}')
    for i in active:
        rows[i]['samples_ms']=[r['score_ms'] for r in rows[i]['measurements'] if r['round'].startswith('final')]
        rows[i]['median_ms']=statistics.median(rows[i]['samples_ms'])
    winner=min(active,key=lambda i:(rows[i]['median_ms'],i))
    return rows[winner],rows
