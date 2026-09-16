#!/usr/bin/env python3
"""Recompute protocol attribution and paired intervals from process logs."""
import csv
import json
from pathlib import Path
import random
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ('full', 'nofence', 'nowait', 'neither', 'l1nosync')
CONFIGS = ('baseline', 'c1', 'c2', 'window2', 'local2', 'window4', 'local4')


def timing(path):
    lines = re.findall(r'^E2E_TIME (.+)$', path.read_text(), re.M)
    if len(lines) != 1:
        raise ValueError(f'{path}: expected one timing')
    return {k: float(v) for k, v in re.findall(r'(\w+)=([0-9.eE+-]+)', lines[0])}


def interval(values):
    rng = random.Random(167)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return statistics.median(values), boot[249], boot[9749]


def samples(folder):
    rows = []
    for i in range(25):
        arms = {a: timing(folder/a/f'r{i}.log') for a in ARMS}
        f, nf, nw, ne = [arms[a]['l2_ms'] for a in ARMS[:4]]
        l1 = arms['full']['l1_ms']
        bar = l1-arms['l1nosync']['l1_ms']
        if bar == 0:
            raise ValueError(f'{folder}/r{i}: zero barrier estimate makes the paired ratio undefined')
        rows.append(dict(l2_ms=f, l1_ms=l1, nofence_ms=nf, nowait_ms=nw,
            neither_ms=ne, l1nosync_ms=arms['l1nosync']['l1_ms'], fence_ms=f-nf,
            wait_ms=f-nw, notify_ms=nw-ne, barrier_ms=bar, protocol_ms=f-ne,
            protocol_over_barrier=(f-ne)/bar, l2_over_l1=f/l1))
    return rows


def summarize(folder):
    raw = samples(folder)
    out = {}
    for k in raw[0]:
        mid, lo, hi = interval([r[k] for r in raw])
        out.update({k: mid, k+'_ci_low': lo, k+'_ci_high': hi})
    return out


def matrix_rows():
    out = []
    for name in CONFIGS:
        for model in ('gqa2', 'mha4'):
            for seq in (4, 128):
                for placement in (0, 5):
                    folder = HERE/'ablation'/name/'paired'/f'{model}_s{seq}_p{placement}'
                    out.append(dict(configuration=name, model=model, seq=seq,
                        placement=placement, evidence=str(folder.relative_to(REPO)), **summarize(folder)))
    for name in CONFIGS[:5]:
        folder = HERE/'realwidth'/name/'paired/real_s4_p0'
        out.append(dict(configuration=name, model='real', seq=4, placement=0,
                        evidence=str(folder.relative_to(REPO)), **summarize(folder)))
    return out


def write_table(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    rows = matrix_rows()
    write_table(HERE/'ablation.tsv', rows)
    scores = {}
    protocol_scores = {}
    achieved = {}
    for c in CONFIGS:
        group = [r for r in rows if r['configuration']==c and r['placement']==0 and r['model']!='real']
        scores[c] = statistics.geometric_mean(r['l2_ms'] for r in group)
        protocol_scores[c] = statistics.geometric_mean(r['protocol_over_barrier'] for r in group)
        achieved[c] = sum(r['protocol_over_barrier']<=1 for r in group)
    best = min(CONFIGS, key=lambda c: (-achieved[c], protocol_scores[c], scores[c]))
    fastest = min(scores, key=scores.get)
    (HERE/'best_configuration.json').write_text(json.dumps(dict(configuration=best,
        criterion='all four cells for every configuration; most cells at the registered threshold, then lowest protocol/barrier geometric mean',
        fastest_end_to_end=fastest, l2_scores=scores, protocol_scores=protocol_scores,
        achieved_cells=achieved), indent=2)+'\n')
    historic = {('gqa2',4): 2.23, ('gqa2',128): 2.49, ('mha4',4): 2.10, ('mha4',128): 2.91}
    research = []
    all_research = []
    for r in rows:
        if r['placement']==0 and r['model']!='real':
            initial = historic[r['model'],r['seq']]
            ratio = r['protocol_over_barrier']
            row=dict(model=r['model'],seq=r['seq'],configuration=r['configuration'],
                ratio=ratio, ci_low=r['protocol_over_barrier_ci_low'], ci_high=r['protocol_over_barrier_ci_high'],
                historical_ratio=initial, historical_gap_closed=(initial-ratio)/(initial-1),
                median_pass=int(ratio<=1), evidence=r['evidence'])
            all_research.append(row)
            if r['configuration']==best:research.append(row)
    write_table(HERE/'research.tsv', research)
    write_table(HERE/'research_all.tsv', all_research)
    print(f'RESEARCH configuration={best}; {sum(r["median_pass"] for r in research)}/4 medians <= 1')
    for r in research:
        print(r)


if __name__=='__main__':
    main()
