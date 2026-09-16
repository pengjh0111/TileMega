#!/usr/bin/env python3
"""Recompute all R5 gates from raw stamps, logs and CPU evaluations; never fail fast."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];PHASE=REPO/'docs/experiments/PHASE'
sys.path.insert(0,str(PHASE))
from fork import decide
spec=importlib.util.spec_from_file_location('trace_r5',REPO/'docs/experiments/TRACE_V2/analyze.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
from measure import timing,source
def table(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def ratio_stats(values):
    rng=random.Random(5005);boot=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(5000))
    return statistics.median(values),boot[125],boot[4874]
def paired(folder,a,b,n=25):
    values=[];sessions=set();starts=set()
    for i in range(n):
        logs={arm:folder/arm/f'r{i}.log' for arm in (a,b)}
        metadata={arm:json.loads(p.with_suffix('.json').read_text()) for arm,p in logs.items()}
        for arm,p in logs.items():
            assert 'RESULT status=PASS' in p.read_text() and metadata[arm]['exit_code']==0,str(p)
            sessions.add(metadata[arm]['session']);starts.add(metadata[arm].get('started_ns',metadata[arm].get('time_ns')))
        arms=('off','phase') if {a,b}=={'off','phase'} else ('control','top1','top2','top3')
        for arm in (a,b):assert metadata[arm]['order']==(arms.index(arm)-i)%len(arms),str(logs[arm])
        assert metadata[a]['order']!=metadata[b]['order']
        values.append(timing(logs[a])['l2_ms']/timing(logs[b])['l2_ms'])
    assert len(sessions)==1 and len(starts)==2*n
    return ratio_stats(values)
def correct(folder,n=50):
    files=list(folder.glob('r*.log'));assert len(files)==n,f'{folder}: {len(files)}/{n}'
    starts=set();binaries=set()
    for i in range(n):
        p=folder/f'r{i}.log';m=json.loads(p.with_suffix('.json').read_text())
        assert m['exit_code']==0 and 'RESULT status=PASS' in p.read_text(),str(p)
        starts.add(m.get('started_ns',m.get('time_ns')));binaries.add(m['binary_sha256'])
    assert len(starts)==n and len(binaries)==1,str(folder)
    return n
def spearman(x,y):
    def ranks(v):
        s=sorted(v);return [(s.index(z)+len(s)-s[::-1].index(z)+1)/2 for z in v]
    a,b=ranks(x),ranks(y);am,bm=statistics.mean(a),statistics.mean(b)
    return sum((u-am)*(v-bm) for u,v in zip(a,b))/(sum((u-am)**2 for u in a)*sum((v-bm)**2 for v in b))**.5
def chosen(cell):
    medians={arm:statistics.median(timing(cell/'pilot'/arm/f'r{i}.log')['l2_ms'] for i in range(5)) for arm in ('top1','top2','top3')}
    arm=min(medians,key=medians.get);frozen=json.loads((cell/'choice.json').read_text());assert frozen['arm']==arm
    for p in (cell/'measure'/arm).glob('r*.json'):assert json.loads(p.read_text())['started_ns']>frozen['frozen_ns']
    return arm
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args();gates=[];phase_rows=[]
    def gate(name,hard,fn):
        try:passed,detail,path=fn();status='PASS' if passed else 'FAIL'
        except Exception as e:status='FAIL';detail=f'{type(e).__name__}: {e}';path='see error path'
        line=f'{name} {status} {detail} evidence={path}';print(line,flush=True);gates.append(dict(gate=name,status=status,hard=hard,detail=detail,evidence=path))
    def identity():
        d=PHASE/'sass_identity';m=json.loads((d/'manifest.json').read_text());assert not m['provisional']
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        parent=subprocess.check_output(['git','rev-parse','HEAD^'],cwd=REPO,text=True).strip()
        assert m['tested_head'] in (head,parent),'stamp is not the final source commit or its artifact child'
        if m['tested_head']==parent:
            changes=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r','HEAD'],cwd=REPO,text=True).splitlines()
            assert all('/sass_identity/' in p for p in changes),'artifact child contains code/docs changes'
        for model in ('gqa2','mha4'):
            base=(d/f'{model}_base.sass').read_bytes();current=(d/f'{model}_head.sass').read_bytes()
            assert base==current and len(current)>1000
            assert hashlib.sha256(current).hexdigest()==m['models'][model]['head_sha256']
        for p,digest in m['inputs'].items():assert hashlib.sha256((REPO/p).read_bytes()).hexdigest()==digest,p
        return True,'models=2/2 bytes_identical final_source_stamp_valid',d
    gate('D3-a/H2',True,identity)
    def phase_correct():
        detail=[]
        for m in ('gqa2','mha4'):
            for s in (4,128):detail.append(f'{m}:s{s}={correct(PHASE/f"raw/correctness/{m}_s{s}")}/50')
        return True,' '.join(detail),PHASE/'raw/correctness'
    gate('D3-b',True,phase_correct)
    def perturb():
        details=[];ok=True
        for m in ('gqa2','mha4','real'):
            for s in (4,128):
                for p in (0,5):
                    med,lo,hi=paired(PHASE/f'raw/measure/{m}_s{s}_p{p}','phase','off')
                    details.append(f'{m}:s{s}:p{p}={med:.6f}[{lo:.6f},{hi:.6f}]')
                    if m!='real':ok &= med<=1.05
        return ok,' '.join(details),PHASE/'raw/measure'
    gate('D3-c',True,perturb)
    def closure():
        count=0;worst=0.
        for m in ('gqa2','mha4','real'):
            for s in (4,128):
                for p in (0,5):
                    d=PHASE/f'raw/trace/{m}_s{s}_p{p}/dump';values,nodes=trace.analyze_phases(d,source(m));phase_rows.append(dict(values,model=m,seq=s,placement=p))
                    for n in nodes:
                        terms=[n[x+'_ns'] for x in ('setup','load_wait','mainloop','epilogue')]
                        assert min(terms)>=0;worst=max(worst,abs(sum(terms)-n['run_ns'])/max(1,n['run_ns']));count+=1
        baseline=count
        supplemental=[]
        for c in (HERE/'raw').glob('*_s*'):
            if (c/'choice.json').exists():
                arm=chosen(c);top=table(c/'top3.tsv')[int(arm[-1])-1]
                supplemental.append((c/'phase'/arm,c/'plans'/top['key']/(top['placement']+'.cu')))
            for folder in (c/'operand_probe').glob('*'):
                supplemental.append((folder,c/'plans'/folder.name/'eft.cu'))
        for folder,src in supplemental:
            assert 'RESULT status=PASS' in (folder/'r0.log').read_text(),str(folder)
            assert json.loads((folder/'r0.json').read_text())['exit_code']==0
            _,nodes=trace.analyze_phases(folder/'dump',src)
            for n in nodes:
                terms=[n[x+'_ns'] for x in ('setup','load_wait','mainloop','epilogue')]
                assert min(terms)>=0;worst=max(worst,abs(sum(terms)-n['run_ns'])/max(1,n['run_ns']));count+=1
        return worst<=.02,f'baseline_nodes={baseline} supplemental_nodes={count-baseline} max_relative_closure_error={worst:.9f}','PHASE/raw/trace + JOINT/raw/*/{phase,operand_probe}'
    gate('D3-d',True,closure)
    def fork():
        line=decide(phase_rows);assert (PHASE/'fork.txt').read_text().strip()==line
        assert line in (REPO/'docs/FINDINGS.md').read_text()
        order=subprocess.run(['git','merge-base','--is-ancestor','495e7c10','60f6e4bc'],cwd=REPO).returncode
        return order==0,line,PHASE/'raw/trace'
    gate('D3-e/H4',True,fork)
    simdir=REPO/'docs/experiments/SIMULATOR/r5'
    def budget():
        rows=table(simdir/'evaluations.tsv');ref=max(float(r['full_us']) for r in rows if r['model']!='real');real=max(float(r['full_us']) for r in rows if r['model']=='real')
        return ref<1000 and real<10000,f'reference_full_us={ref:.3f} real_full_us={real:.3f}',simdir/'evaluations.tsv'
    gate('S1c-a',True,budget)
    def calibration():
        modes={'legacy_grid_stride':'0','balanced':'4','rotate':'5'};rows=[r for r in table(simdir/'evaluations.tsv') if r['model']!='real' and r['candidate'] in modes]
        raw=table(REPO/'docs/experiments/SIMULATOR/raw/time/l2.tsv')
        actual=[statistics.median(float(x['l2_ms']) for x in raw if (x['model'],x['seq'],x['place'])==(r['model'],r['seq'],modes[r['candidate']])) for r in rows]
        pred=[float(r['coarse_ns']) for r in rows];rho=spearman(pred,actual);best=min(range(len(rows)),key=lambda i:actual[i]);rank=sorted(range(len(rows)),key=lambda i:pred[i]).index(best)+1
        full=spearman([float(r['full_ns']) for r in rows],actual)
        return rho>=.85 and rank<=3,f'points={len(rows)} coarse_rho={rho:.9f} full_rho={full:.9f} actual_top1_coarse_rank={rank}',simdir/'evaluations.tsv'
    gate('S1c-b',True,calibration)
    def errors():
        rows=table(simdir/'replay.tsv');values=[]
        for r in rows:
            meta=trace.read_meta(REPO/r['dump']/'meta.tsv');measured=float(meta['l2_ms'])*1e6
            assert abs(measured-float(r['measured_ns']))<1e-6;values.append(abs(float(r['predicted_ns'])/measured-1))
        return len(rows)==68,f'dumps={len(rows)} abs_relative_p50={statistics.median(values):.9f} p90={trace.percentile(values,.9):.9f} max={max(values):.9f}',simdir/'replay.tsv'
    gate('S1c-c',False,errors)
    def selected_correct():
        details=[]
        for m in ('gqa2','mha4'):
            for s in (4,128):
                c=HERE/f'raw/{m}_s{s}';chosen(c);details.append(f'{m}:s{s}={correct(c/"correctness")}/50')
                for scan_seq,past in ((1,0),(128,512),(2048,0)):
                    correct(c/'seqscan'/f's{scan_seq}_p{past}')
        return True,' '.join(details)+' SEQSCAN=PASS',HERE/'raw'
    gate('S3-a',True,selected_correct)
    def research():
        passed=0;details=[]
        for m in ('gqa2','mha4'):
            for s in (4,128):
                c=HERE/f'raw/{m}_s{s}';arm=chosen(c);med,lo,hi=paired(c/'measure',arm,'control');passed+=med<1 and hi<1
                details.append(f'{m}:s{s}:{arm}={med:.6f}[{lo:.6f},{hi:.6f}]')
        return passed>=3,f'achieved={passed}/4 '+' '.join(details),HERE/'raw/*/measure'
    gate('S3-b',False,research)
    def ranking():
        details=[]
        for m in ('gqa2','mha4','real'):
            for s in (4,128):
                c=HERE/f'raw/{m}_s{s}';med={arm:statistics.median(timing(c/'measure'/arm/f'r{i}.log')['l2_ms'] for i in range(25)) for arm in ('top1','top2','top3')}
                details.append(f'{m}:s{s}:predicted_top1_measured_rank={sorted(med,key=med.get).index("top1")+1}/3')
        return True,' '.join(details),HERE/'raw/*/measure'
    gate('S1c-d',False,ranking)
    def top_percent():
        ok,detail,path=ranking()
        return False,detail+' catalog_top_3_percent=unresolved unmeasured_catalog_candidates_have_no_empirical_rank',path
    gate('S3-c',True,top_percent)
    def paths_and_ratios():
        details=[]
        for m in ('gqa2','mha4','real'):
            for s in (4,128):
                c=HERE/f'raw/{m}_s{s}';arm=chosen(c)
                top=table(c/'top3.tsv')[int(arm[-1])-1]
                for tag,src in [('control',source(m)),(arm,c/'plans'/top['key']/(top['placement']+'.cu'))]:
                    v=trace.analyze(c/'trace'/tag/'dump',src,1);floor=max(v['cp_corrected_ns'],v['queue_lb_ns']);assert v['measured_l2_ms']*1e6>=floor
                    logs=[timing(c/'measure'/tag/f'r{i}.log') for i in range(25)]
                    l2=statistics.median(r['l2_ms'] for r in logs)
                    details.append(f'{m}:s{s}:{tag}:cp_ns={v["cp_corrected_ns"]},nodes={v["cp_corrected_nodes"]},mean_ns={v["cp_corrected_ns"]/v["cp_corrected_nodes"]:.3f},queue_cp={v["queue_lb_ns"]/v["cp_corrected_ns"]:.6f},l2_l1={statistics.median(r["l2_ms"]/r["l1_ms"] for r in logs):.6f},l2_floor={l2*1e6/floor:.6f}')
        return True,' '.join(details),HERE/'raw/*/trace'
    gate('S3-d/f',False,paths_and_ratios)
    def real():
        details=[]
        for s in (4,128):
            c=HERE/f'raw/real_s{s}';arm=chosen(c);med,lo,hi=paired(c/'measure',arm,'control');details.append(f'real:s{s}={med:.6f}[{lo:.6f},{hi:.6f}] correctness={correct(c/"correctness")}/50')
        return True,' '.join(details),HERE/'raw/real*'
    gate('S3-e',False,real)
    def prefetch():
        line=decide(phase_rows);assert 'rule=3' in line or 'rule=2' in line
        return True,'not_applicable '+line+' EX-E4 excluded by frozen rule',PHASE/'raw/trace'
    gate('E4-conditional',False,prefetch)
    hard_failed=sum(r['hard'] and r['status']=='FAIL' for r in gates)
    print(f'VERIFY5 gates={len(gates)} hard_failed={hard_failed}',flush=True)
    if a.output:a.output.write_text(json.dumps(gates,indent=2)+'\n')
    return int(hard_failed>0)
if __name__=='__main__':sys.exit(main())
