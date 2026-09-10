#!/usr/bin/env python3
"""Verify frozen receipts and compute paired state comparisons."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys


def interval(values):
    rng=random.Random(20260910)
    draws=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(20000))
    return [draws[500],draws[19499]]


def signed_rank(values):
    ordered=sorted((abs(v),v>0) for v in values if v)
    n=len(ordered)
    if not n: return 1.0
    positive=ties=0
    i=0
    while i<n:
        j=i+1
        while j<n and ordered[j][0]==ordered[i][0]: j+=1
        positive+=(i+1+j)/2*sum(sign for _,sign in ordered[i:j])
        ties+=(j-i)**3-(j-i)
        i=j
    variance=n*(n+1)*(2*n+1)/24-ties/48
    z=max(0,abs(positive-n*(n+1)/4)-.5)/math.sqrt(variance)
    return math.erfc(z/math.sqrt(2))


root=Path(sys.argv[1])
rows=list(csv.DictReader((root/'paired.tsv').open(),delimiter='\t'))
lookup={}
for row in rows:
    key=(row['model'],int(row['seq']),int(row['state']),int(row['round']))
    if key in lookup or row['status']!='PASS': raise ValueError('duplicate or failing receipt')
    log=root/'logs'/f'{key[0]}_p{key[2]}_s{key[1]}_r{key[3]}.txt'
    if hashlib.sha256(log.read_bytes()).hexdigest()!=row['log_sha256']:
        raise ValueError('changed GPU log')
    lookup[key]=row
expected={(m,s,a,r) for m in ('gqa2','mha4') for s in (4,128) for a in (0,4) for r in range(50)}
if lookup.keys()!=expected: raise ValueError('incomplete 400-process matrix')
results=[]
for model in ('gqa2','mha4'):
    for seq in (4,128):
        predicted=(root.parent/'projected_runtime'/f'{model}_price_s{seq}.txt').read_text()
        price=dict(re.findall(r'(\w+)=([^\s]+)',next(l for l in predicted.splitlines()
                                                       if l.startswith('PLACEMENT_EVENT_PRICE '))))
        for r in range(50):
            observed=json.loads(lookup[model,seq,4,r]['schedule_json'])
            if int(observed['waits'])!=int(price['balanced_waits']):
                raise ValueError('CG/GPU balanced wait mismatch')
        for rounds in (25,50):
            for metric in ('l05_ms','l1_ms','l2_ms'):
                base=[float(lookup[model,seq,0,r][metric]) for r in range(rounds)]
                new=[float(lookup[model,seq,4,r][metric]) for r in range(rounds)]
                differences=[b-a for a,b in zip(base,new)]
                ratios=[b/a for a,b in zip(base,new)]
                results.append(dict(model=model,seq=seq,rounds=rounds,metric=metric,
                    baseline_median_ms=statistics.median(base),balanced_median_ms=statistics.median(new),
                    paired_delta_ms=statistics.median(differences),delta_ci95=interval(differences),
                    paired_ratio=statistics.median(ratios),ratio_ci95=interval(ratios),
                    wilcoxon_normal_tie_corrected_p=signed_rank(differences),
                    predicted_event_delta_ns=float(price['delta_ns'])))
print(json.dumps(dict(correctness='400/400',balanced_wait_checks='200/200',
    primary_rounds=25,all_rounds_sensitivity=50,bootstrap_draws=20000,results=results),indent=2))
