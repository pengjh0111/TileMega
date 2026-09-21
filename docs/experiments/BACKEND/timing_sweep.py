#!/usr/bin/env python3
"""R8 A-g, full: the three levels over the decode sweep, with and without
`MIDPOINT_REFINE`.

No solve happens here. Every Plan is one R7 already produced and left on disk,
so the only thing that changes between R7's number and this one is the backend
the same Plan is compiled against -- which is what A-g is asking. The refine
arm is the switch F-239 priced: R7's reference-model timings were built
without it and its Llama timings with it, so comparing across the two families
without separating the switch compares two different things.

Cells and their Plans:
  gqa2/mha4 seq 1, 16   /root/r7_work/ref/<model>_s<seq>/auto.cu      (R7 D-d)
  gqa2/mha4 seq 4, 128  E2E_REAL/topk/<model>_s<seq>/auto.cu          (R7 D-b/D-d)
  llama seq 1, 16, 64   /root/r7_work/llama_s<seq>/auto.cu            (R7 D-c)
  llama seq 4           /root/r7_work/llama_hoist/auto.cu             (R7 D-a/D-c)
"""
import argparse,fcntl,json,os,re,statistics,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure
R7=Path('/root/r7_work')

def cells():
    for model in ('gqa2','mha4'):
        for seq in (1,4,16,128):
            source=(R7/'ref'/f'{model}_s{seq}'/'auto.cu' if seq in (1,16)
                    else REPO/f'docs/experiments/E2E_REAL/topk/{model}_s{seq}/auto.cu')
            yield f'{model}_s{seq}',source,model,seq,None
    for seq in (1,4,16,64):
        root=R7/('llama_hoist' if seq==4 else f'llama_s{seq}')
        yield f'llama_s{seq}',root/'auto.cu',None,seq,root/'fixture'

def macro(source,name,default):
    m=re.search(r'^#define '+name+r' (\d+)$',Path(source).read_text(),re.M)
    return m[1] if m else default

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=HERE/'timing_sweep')
    ap.add_argument('--rounds',type=int,default=25)
    ap.add_argument('--llama-rounds',type=int,default=10)
    ap.add_argument('--arch',default='sm_89')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    session=str(time.time_ns());rows=[]
    for name,source,model,seq,fixture in cells():
        if not Path(source).is_file():
            print('MISSING',name,source,flush=True);continue
        cell=a.out/name;cell.mkdir(parents=True,exist_ok=True)
        for arm,extra in (('refine_off',[]),('refine_on',['MIDPOINT_REFINE=1'])):
            spec=dict(source=str(source),kappa=macro(source,'TILEMEGA_EVENT_KAPPA','1'),
                      residency=macro(source,'TILEMEGA_RESIDENCY_CAP','0'),
                      placement_macro='0',extra=extra)
            if not (cell/'bin'/arm).exists():
                if measure.build(cell,name,arm,spec,a.arch):
                    print('BUILD_FAILED',name,arm,flush=True);continue
            folder=cell/arm;folder.mkdir(exist_ok=True)
            target=fixture if fixture is not None else measure.fixture(model,seq)
            rounds=a.llama_rounds if model is None else a.rounds
            samples={'l05_ms':[],'l1_ms':[],'l2_ms':[]}
            for i in range(rounds):
                log=folder/f'r{i}.log'
                if log.exists():
                    text=log.read_text()
                else:
                    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
                    env.update(TILEMEGA_WARMUP='5',TILEMEGA_REPEAT='11')
                    cmd=[str(cell/'bin'/arm),str(target)]
                    with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
                        r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=3600)
                    text=r.stdout+r.stderr;log.write_text(text)
                    log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,
                        round=i,session=session,started_ns=start,
                        elapsed_ns=time.time_ns()-start,exit_code=r.returncode))+'\n')
                for key in samples:
                    m=re.search(key+r'=([0-9.]+)',text)
                    if m:samples[key].append(float(m[1]))
            if not samples['l2_ms']:
                print('NO_SAMPLES',name,arm,flush=True);continue
            row=dict(cell=name,arm=arm,rounds=rounds,
                     **{k:statistics.median(v) for k,v in samples.items()})
            row['l2_over_l1']=row['l2_ms']/row['l1_ms']
            rows.append(row)
            print('SWEEP',name,arm,'rounds',rounds,
                  'l05',round(row['l05_ms'],4),'l1',round(row['l1_ms'],4),
                  'l2',round(row['l2_ms'],4),flush=True)
    if rows:
        columns=list(rows[0])
        with (a.out/'timing_sweep.tsv').open('w') as f:
            f.write('\t'.join(columns)+'\n')
            for r in rows:f.write('\t'.join(str(r[c]) for c in columns)+'\n')
    # The switch's cost, cell by cell.
    by={}
    for r in rows:by.setdefault(r['cell'],{})[r['arm']]=r
    for cell,arms in by.items():
        if len(arms)==2:
            print('REFINE_COST',cell,'l2 off',round(arms['refine_off']['l2_ms'],4),
                  'on',round(arms['refine_on']['l2_ms'],4),
                  'ratio',round(arms['refine_on']['l2_ms']/arms['refine_off']['l2_ms'],3),
                  flush=True)
if __name__=='__main__':main()
