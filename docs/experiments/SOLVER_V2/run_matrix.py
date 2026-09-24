#!/usr/bin/env python3
"""Run one R9 skeleton arm from a freshly solved legacy seed.

The default searches CandidateGenerator's entire legal GEMM domain, P=3,
all resident levels, top-5 real occupancy checks, top-3 simulation-ranked
GPU builds, then ten fresh processes per candidate. --domain/--passes are
explicit diagnostic/degradation controls, recorded in solve.command.json.
Outputs: selected.cu[.topN.cu/.mlir], search/phase/resource/class TSVs,
finalN task and edge dumps, ptxas logs, raw process logs and winner metadata.
"""
import argparse,json,pathlib,shutil,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent));from measure import ROOT,run
import os,subprocess

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['bridge','fixture','seed','out']:ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--seq',type=int,required=True);ap.add_argument('--past',type=int,default=3)
    ap.add_argument('--k-base',choices=['4','8','16','W'],default='8');ap.add_argument('--passes',type=int,default=3)
    ap.add_argument('--search-jobs',type=int,default=1)
    ap.add_argument('--domain',type=pathlib.Path);ap.add_argument('--compiler',type=pathlib.Path,default=ROOT/'build-portable/tools/tilemega-compile')
    ap.add_argument('--variant-cache',type=pathlib.Path,default=pathlib.Path('/root/r9_work/variant_resources'))
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);source=a.out/'selected.cu'
    if not a.seed.exists():raise RuntimeError('fresh legacy seed is not ready: '+str(a.seed))
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    cmd=[a.compiler,a.bridge,source,'--solver=skeleton','--legacy-seed',a.seed,'--solve',ROOT/'docs/experiments/COSTMODEL/event_fit/target.json','--seq',a.seq,'--past',a.past,'--k-base',a.k_base,'--search-passes',a.passes,'--search-jobs',a.search_jobs,'--variant-cache',a.variant_cache,'--dump-cg',a.out/'selected.mlir','--hop-curve',ROOT/'docs/experiments/SIMULATOR/hop_ns.tsv']
    if a.domain:cmd+=['--search-domain',a.domain]
    if run(cmd,a.out/'solve.log',env):raise RuntimeError('skeleton solve failed')
    if run(['python3',ROOT/'docs/experiments/SOLVER_V2/measure.py','--source',source,'--fixture',a.fixture,'--top3'],a.out/'measure.log',env):raise RuntimeError('top-3 build/internal gate failed')
    if run([ROOT/'build-portable/tools/tilemega-skeleton-audit',a.out/'selected.mlir',ROOT/'docs/experiments/COSTMODEL/event_fit/target.json',a.seq,a.past],a.out/'oracle_audit.log',env):raise RuntimeError('anchored Oracle set equality failed')
if __name__=='__main__':main()
