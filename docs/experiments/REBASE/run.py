#!/usr/bin/env python3
"""R6 solved-geometry placement/protocol ablation, W=1 throughout.

Actions: generate, build, measure, trace_build, trace.
Inputs are frozen JOINT2 cell directories, not a manually chosen geometry.
Each cell has six placement controls under R3 B and four protocol variants
at the selected placement. Five arms are rotated across all configurations
within each of 25 fresh-process paired rounds. CUDA architecture is explicit.
"""
import argparse,concurrent.futures,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure as r5
sys.path.insert(0,str(REPO/'docs/experiments/JOINT2'));from process import run as run_process
ARMS={'full':[],'nofence':['UNSAFE_NO_NOTIFY_FENCE=1'],'nowait':['UNSAFE_NO_EVENT_WAIT=1'],'neither':['UNSAFE_NO_EVENT_WAIT=1','UNSAFE_NO_EVENT_NOTIFY=1'],'l1nosync':['UNSAFE_NO_GRID_SYNC=1']}
def cells(a):return [(m,s,HERE/a.out/f'{m}_s{s}',a.joint/f'{m}_s{s}') for m in a.models for s in a.seqs]
def specs(cell):
 rows=list(csv.DictReader((cell/'placements.tsv').open(),delimiter='\t'))
 chosen=json.loads((cell/'selection.json').read_text())['placement']
 out={r['placement']:dict(source=r['source'],kappa=r['kappa'],residency=r['residency'],placement_macro='0') for r in rows}
 selected=out[chosen]
 for name,extra in {'protocol_off':['BARRIER_V2=0','EVENT_SOLO=0','EVENT_RED_PUBLISH=0','WAIT_POLICY=0'],
  'c1':['RELEASE_AFTER_BARRIER=1'], 'c2':['RELEASE_AFTER_BARRIER=1','ASYNC_PUBLISH=1'],
  'c3':['RELEASE_AFTER_BARRIER=1','ASYNC_PUBLISH=1','LOCAL_DEP_SMEM=1']}.items():out[name]=dict(selected,extra=extra)
 return {config+'__'+arm:dict(spec,extra=spec.get('extra',[])+flags) for config,spec in out.items() for arm,flags in ARMS.items()}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['generate','build','measure','trace_build','trace']);ap.add_argument('--joint',type=Path,default=REPO/'docs/experiments/JOINT2/reduced');ap.add_argument('--out',type=Path,default=Path('raw'));ap.add_argument('--models',nargs='+',default=['gqa2','mha4']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--arch',default='sm_89');a=ap.parse_args();session=str(time.time_ns())
 if a.action=='generate':
  for m,s,cell,joint in cells(a):
   cell.mkdir(parents=True,exist_ok=True)
   choice=json.loads((joint/'choice.json').read_text());rank=choice['arm'][3:]
   row=next(r for r in csv.DictReader((joint/'auto.cu.top3.tsv').open(),delimiter='\t') if r['rank']==rank)
   (cell/'selection.json').write_text(json.dumps(dict(joint=str(joint),choice=choice,placement=row['placement'],frozen_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()),indent=2)+'\n')
   cmd=[os.getenv('R6_REBASE_DRIVER','/tmp/r6_rebase_generate'),row['cg'],os.getenv('R6_REBASE_TARGET',str(REPO/'docs/experiments/COSTMODEL/event_fit/target.json')),str(cell)]
   (cell/'generate.json').write_text(json.dumps(dict(command=cmd,session=session),indent=2)+'\n')
   with (cell/'generate.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  return
 if a.action in ('build','trace_build'):
  import shutil
  free=shutil.disk_usage(HERE).free//2**20;print(f'DISK NEED_MIB=16384 FREE_MIB={free}',flush=True)
  if free<16384:raise RuntimeError('insufficient build disk')
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
   futures=[]
   for m,s,cell,j in cells(a):
    sp=specs(cell);(cell/'specs.json').write_text(json.dumps(sp,indent=2)+'\n')
    for name,spec in sp.items():
     if a.action=='trace_build' and not name.endswith('__full'):continue
     binary=cell/'bin'/(name+('_trace' if a.action=='trace_build' else ''))
     if binary.exists():continue
     futures.append(pool.submit(r5.build,cell,m,name,spec,a.arch,a.action=='trace_build'))
   if any(f.result()!=0 for f in futures):raise RuntimeError('ablation compilation failed')
  return
 for m,s,cell,j in cells(a):
  names=list(specs(cell))
  for i in range(25 if a.action=='measure' else 1):
   for k in range(len(names)):
    name=names[(i+k)%len(names)]
    if a.action=='trace' and not name.endswith('__full'):continue
    arm=name+('_trace' if a.action=='trace' else '')
    try:run_process(cell,m,s,arm,cell/a.action/name,i,k,session,a.action=='trace')
    except RuntimeError:
     # Unsafe timing probes deliberately permit numerical mismatches. A
     # complete timing record is still required, just as in the R4 harness.
     if name.endswith('__full') or a.action=='trace':raise
     r5.timing(cell/a.action/name/f'r{i}.log')
   print('REBASE',m,s,a.action,'round',i+1,flush=True)
if __name__=='__main__':main()
