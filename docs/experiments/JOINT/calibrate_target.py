#!/usr/bin/env python3
"""Fresh target-local hop curve and five-arm publication calibration."""
import argparse,json,os,subprocess,time
from pathlib import Path
from measure import build,run,source,REPO
ARMS={'full':[], 'nofence':['UNSAFE_NO_NOTIFY_FENCE=1'], 'nowait':['UNSAFE_NO_EVENT_WAIT=1'],
      'neither':['UNSAFE_NO_EVENT_WAIT=1','UNSAFE_NO_EVENT_NOTIFY=1'], 'l1nosync':['UNSAFE_NO_GRID_SYNC=1']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',type=Path,required=True);ap.add_argument('--arch',default='sm_120');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 sim=REPO/'docs/experiments/SIMULATOR';session=str(time.time_ns())
 for name,extra in [('rmw',[]),('load',['-DTILEMEGA_EVENT_LOAD_POLL=1'])]:
  binary=a.out/('contention_'+name);cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch='+a.arch,*extra,'-I'+str(REPO/'include'),str(sim/'contention.cu'),'-o',str(binary)]
  with (a.out/(name+'.build.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run([str(binary),'4096','20000',str(a.out/(name+'.tsv'))],check=True)
 subprocess.run(['python3',str(sim/'hop_fit.py'),str(a.out/'rmw.tsv'),str(a.out/'load.tsv'),str(a.out/'hop_ns.tsv')],check=True)
 # Unsafe probes retain their pre-existing contract: do not accept their
 # numerical output as correctness. Preserve process failures and timings.
 for m in ('gqa2','mha4'):
  for s in (4,128):
   for p in (0,5):
    cell=a.out/'fence'/f'{m}_s{s}_p{p}';cell.mkdir(parents=True)
    for arm,extra in ARMS.items():
     if build(cell,m,arm,dict(source=str(source(m)),kappa='1',residency='0',placement_macro=str(p),extra=extra),a.arch):raise RuntimeError('probe compile')
    # Publication residuals require event timestamps, not phase-only slots.
    spec=dict(source=str(source(m)),kappa='1',residency='0',placement_macro=str(p))
    if build(cell,m,'calibration',spec,a.arch,trace=True):raise RuntimeError('trace calibration compile')
    run(cell,m,s,'calibration_trace',a.out/'trace'/f'{m}_s{s}_p{p}',0,0,session,dump=True)
    for i in range(25):
     for j in range(5):
      arm=list(ARMS)[(i+j)%5]
      try:run(cell,m,s,arm,cell/arm,i,j,session)
      except RuntimeError:
       meta=json.loads((cell/arm/f'r{i}.json').read_text())
       text=(cell/arm/f'r{i}.log').read_text()
       if arm=='full' or 'E2E_TIME ' not in text or meta['exit_code'] not in (0,1):raise
 subprocess.run(['python3',str(sim/'r5/calibrate.py'),'--fence-root',str(a.out/'fence'),'--trace-root',str(a.out/'trace'),'--hop',str(a.out/'hop_ns.tsv'),'--out',str(a.out)],check=True)
if __name__=='__main__':main()
