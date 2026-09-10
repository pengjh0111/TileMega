#!/usr/bin/env python3
"""Audit fresh-process receipts and summarize paired fusion measurements."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

def interval(values):
    rng=random.Random(20260910)
    draws=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(20000))
    return [draws[500],draws[19499]]

def signed_rank(values):
    ordered=sorted((abs(v),v>0) for v in values if v)
    n=len(ordered)
    if not n:
        return 1.0
    positive=ties=0
    i=0
    while i<n:
        j=i+1
        while j<n and ordered[j][0]==ordered[i][0]:
            j+=1
        positive+=(i+1+j)/2*sum(sign for _,sign in ordered[i:j])
        ties+=(j-i)**3-(j-i)
        i=j
    variance=n*(n+1)*(2*n+1)/24-ties/48
    z=max(0,abs(positive-n*(n+1)/4)-.5)/math.sqrt(variance)
    return math.erfc(z/math.sqrt(2))

parser=argparse.ArgumentParser()
parser.add_argument("root",type=Path)
parser.add_argument("--out",type=Path)
args=parser.parse_args()
root=args.root
rows=list(csv.DictReader((root/"paired.tsv").open(),delimiter="\t"))
if not rows:
    raise RuntimeError("empty evidence matrix")
chain="consumer" in rows[0]
axis="consumer" if chain else "model"
states=("separate","fused") if chain else ("0","1")
groups=("add","norm") if chain else ("gqa2","mha4")
lookup={}
for row in rows:
    key=(row[axis],int(row["seq"]),row["state"],int(row["round"]))
    if key in lookup:
        raise RuntimeError("duplicate process receipt")
    tag=key[0]+("_"+key[2] if chain else "_f"+key[2])
    log=root/"logs"/f"{tag}_s{key[1]}_r{key[3]:03d}.txt"
    text=log.read_text()
    if "RESULT status=PASS" not in text:
        raise RuntimeError("receipt lacks passing process output")
    for field,label in (("resource_json","E2E_RESOURCE"),("schedule_json","E2E_SCHEDULE"),("hash_json","E2E_HASH")):
        lines=[line for line in text.splitlines() if line.startswith(label+" ")]
        if len(lines)!=1 or dict(re.findall(r"(\w+)=([^\s]+)",lines[0]))!=json.loads(row[field]):
            raise RuntimeError("receipt differs from source log")
    lookup[key]=row
expected={(group,seq,state,r) for group in groups for seq in (4,128) for state in states for r in range(50)}
if set(lookup)!=expected:
    raise RuntimeError("incomplete 400-process matrix")
results=[]
for group in groups:
    for seq in (4,128):
        for r in range(50):
            a=json.loads(lookup[group,seq,states[0],r]["hash_json"])
            b=json.loads(lookup[group,seq,states[1],r]["hash_json"])
            if a!=b or len(set(a.values()))!=1:
                raise RuntimeError("fusion/control or level output bits differ")
        for rounds in (25,50):
            for metric in ("l05_ms","l1_ms","l2_ms"):
                base=[float(lookup[group,seq,states[0],r][metric]) for r in range(rounds)]
                fused=[float(lookup[group,seq,states[1],r][metric]) for r in range(rounds)]
                delta=[b-a for a,b in zip(base,fused)]
                ratios=[b/a for a,b in zip(base,fused)]
                results.append(dict(group=group,seq=seq,rounds=rounds,metric=metric,
                    baseline_median_ms=statistics.median(base),fused_median_ms=statistics.median(fused),
                    paired_delta_ms=statistics.median(delta),delta_ci95=interval(delta),
                    paired_ratio=statistics.median(ratios),ratio_ci95=interval(ratios),
                    wilcoxon_normal_tie_corrected_p=signed_rank(delta)))
text=json.dumps(dict(correctness="400/400",primary_rounds=25,sensitivity_rounds=50,
    receipts_sha256=hashlib.sha256((root/"paired.tsv").read_bytes()).hexdigest(),results=results),indent=2)
if args.out:
    with args.out.open("x") as stream:
        stream.write(text+"\n")
else:
    print(text)
