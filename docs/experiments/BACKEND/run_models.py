#!/usr/bin/env python3
"""R8 A-b / A-e / A-g: the reference models on the reworked backend.

One runner for three gates, because they want the same binaries:

  A-b  correctness, 50 fresh processes per cell, every switch off
  A-e  the occupancy closed form against the driver, read off each run's own
       `E2E_RESOURCE` line rather than a separate probe
  A-g  the three levels per decode seq, same caliber as R7 D-c/D-d

The Plans are solved at this HEAD, so they carry `tilemega.solved_arch` and
the binaries assert their architecture at run time (BE-1).
"""
import argparse,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure
TARGET=REPO/'docs/experiments/COSTMODEL/event_fit/target.json'
DOMAIN=REPO/'docs/experiments/COSTMODEL/event_fit/search_domain.json'
HOP=REPO/'docs/experiments/SIMULATOR/hop_ns.tsv'
CELLS=[('gqa2',4),('gqa2',128),('mha4',4),('mha4',128)]

def export(model):
    return REPO/f'docs/experiments/SEQSCAN/raw/export/{model}.json'

def solve(cell,model,seq,capacity):
    out=cell/'auto.cu'
    if out.exists():return
    cell.mkdir(parents=True,exist_ok=True)
    cmd=[str(REPO/'build-portable/tools/tilemega-compile'),str(export(model)),str(out),
         '--solve',str(TARGET),'--seq',str(seq),'--past','3',
         '--search-capacity',str(capacity),'--search-domain',str(DOMAIN),
         '--hop-curve',str(HOP)]
    start=time.time_ns()
    with (cell/'solve.log').open('w') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
    (cell/'solve.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,
        elapsed_ns=time.time_ns()-start,head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('solve failed: '+str(cell/'solve.log'))

def macro(source,name,default):
    m=re.search(r'^#define '+name+r' (\d+)$',Path(source).read_text(),re.M)
    return m[1] if m else default

def run(binary,fixture,log,warmup,repeat,session,round_):
    if log.exists():return log.read_text()
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    env.update(TILEMEGA_WARMUP=str(warmup),TILEMEGA_REPEAT=str(repeat))
    cmd=[str(binary),str(fixture)]
    with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
        r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=1800)
    log.write_text(r.stdout+r.stderr)
    log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,round=round_,
        session=session,started_ns=start,elapsed_ns=time.time_ns()-start,
        exit_code=r.returncode))+'\n')
    return r.stdout+r.stderr

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['solve','correctness','timing','report'])
    ap.add_argument('--out',type=Path,default=HERE/'models')
    ap.add_argument('--rounds',type=int,default=50)
    ap.add_argument('--timing-rounds',type=int,default=5)
    ap.add_argument('--capacity',type=int,default=12)
    ap.add_argument('--arch',default='sm_89')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    session=str(time.time_ns())
    if a.action=='solve':
        for model,seq in CELLS:
            solve(a.out/f'{model}_s{seq}',model,seq,a.capacity)
            print('SOLVED',model,seq,flush=True)
        return
    if a.action=='correctness':
        for model,seq in CELLS:
            cell=a.out/f'{model}_s{seq}'
            spec=dict(source=str(cell/'auto.cu'),
                      kappa=macro(cell/'auto.cu','TILEMEGA_EVENT_KAPPA','1'),
                      residency=macro(cell/'auto.cu','TILEMEGA_RESIDENCY_CAP','0'),
                      placement_macro='0',extra=['MIDPOINT_REFINE=1'])
            if measure.build(cell,model,'default',spec,a.arch):
                raise RuntimeError('compile failed '+str(cell))
            folder=cell/'correctness';folder.mkdir(exist_ok=True)
            passing=0
            for i in range(a.rounds):
                text=run(cell/'bin'/'default',measure.fixture(model,seq),
                         folder/f'r{i}.log',0,1,session,i)
                passing+='RESULT status=PASS' in text
            print('A-b',model,seq,'rounds',a.rounds,'passing',passing,flush=True)
        return
    if a.action=='timing':
        for model,seq in CELLS:
            cell=a.out/f'{model}_s{seq}';folder=cell/'timing';folder.mkdir(exist_ok=True)
            for i in range(a.timing_rounds):
                run(cell/'bin'/'default',measure.fixture(model,seq),
                    folder/f'r{i}.log',5,11,session,i)
            print('A-g',model,seq,'rounds',a.timing_rounds,flush=True)
        return
    # report: correctness, occupancy against the driver, and the three levels
    rows=[]
    for model,seq in CELLS:
        cell=a.out/f'{model}_s{seq}'
        logs=sorted((cell/'correctness').glob('r*.log'),key=lambda p:int(p.stem[1:]))
        passing=sum('RESULT status=PASS' in p.read_text() for p in logs)
        timing=sorted((cell/'timing').glob('r*.log'),key=lambda p:int(p.stem[1:]))
        text=(timing or logs)[0].read_text() if (timing or logs) else ''
        res=dict(kv.split('=',1) for kv in
                 (re.search(r'^E2E_RESOURCE (.*)$',text,re.M)[1].split()
                  if re.search(r'^E2E_RESOURCE ',text,re.M) else []))
        def median(name):
            got=sorted(float(m) for p in timing
                       for m in re.findall(name+r'=([0-9.]+)',p.read_text()))
            return got[len(got)//2] if got else float('nan')
        rows.append(dict(cell=f'{model}_s{seq}',rounds=len(logs),passing=passing,
            block=res.get('block'),reg=res.get('reg'),task_smem=res.get('task_smem'),
            driver_ctas=res.get('ctas_per_sm'),regs_per_sm=res.get('regs_per_sm'),
            smem_per_sm=res.get('smem_per_sm'),threads_per_sm=res.get('threads_per_sm'),
            l05_ms=median('l05_ms'),l1_ms=median('l1_ms'),l2_ms=median('l2_ms')))
    columns=list(rows[0])
    with (a.out/'models.tsv').open('w') as f:
        f.write('\t'.join(columns)+'\n')
        for r in rows:f.write('\t'.join(str(r[c]) for c in columns)+'\n')
    for r in rows:print('MODEL',r['cell'],'passing',r['passing'],'/',r['rounds'],
        'l05',r['l05_ms'],'l1',r['l1_ms'],'l2',r['l2_ms'],flush=True)
if __name__=='__main__':main()
