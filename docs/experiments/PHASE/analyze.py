#!/usr/bin/env python3
"""Analyze every phase dump using the retained R4 formulas plus R5 columns."""
import csv
import os
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('trace_analyze',REPO/'docs/experiments/TRACE_V2/analyze.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
from run import source
from fork import decide

def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    raw=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'raw'
    rows=[]
    for folder in sorted((raw/'trace').glob('*')):
        m,s,p=folder.name.split('_');seq=int(s[1:]);placement=int(p[1:])
        values,nodes=trace.analyze_phases(folder/'dump',source(m))
        rows.append(dict(values, model=m,seq=seq,placement=placement,dump=os.path.relpath(folder/'dump',REPO)))
        write(folder/'node_phases.tsv',nodes)
    write(raw/'analysis.tsv',rows)
    print(decide(rows))

if __name__=='__main__':main()
