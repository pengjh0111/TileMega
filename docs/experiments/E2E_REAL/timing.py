#!/usr/bin/env python3
"""§7.2 D-c/D-d: the three harness levels on the anchored model, per decode seq.

Each round is one fresh process, and one process reports all three levels from
the same allocation and the same inputs, so L0.5, L1 and L2 are paired by
construction; the pairing is what a ratio may be taken over. The harness's own
defaults do the averaging inside a round (five warmups, median of eleven), so a
round is a median and the twenty-five rounds are the spread across processes.

No threshold is applied here (D-c sets none): this prints the numbers and the
ratios the report gate asks for.
"""
import argparse,fcntl,hashlib,json,os,re,statistics,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

def macro(text,name,default=None):
 m=re.search(r'^#define '+name+r' (\d+)$',text,re.M)
 return m[1] if m else default

def floor_ns(root):
 """The winner's lower bound, as the solver reported it in the top-3 table."""
 table=root/'auto.cu.top3.tsv'
 if not table.is_file():return None
 rows=[l.split('\t') for l in table.read_text().strip().splitlines()]
 head={k:i for i,k in enumerate(rows[0])}
 return float(rows[1][head['floor_ns']])

def cell(root,rounds,out,arch,model=None):
 root=root.resolve();text=(root/'auto.cu').read_text()
 seq=int(macro(text,'TILEMEGA_SOLVED_SEQ'));past=int(macro(text,'TILEMEGA_SOLVED_PAST'))
 # A2's refinement is part of the anchored model's admission, not of the two
 # reference models, which D-d compares against in their own default build.
 refine=model is None
 spec=dict(source=str(root/'auto.cu'),kappa=macro(text,'TILEMEGA_EVENT_KAPPA','1'),
           residency=macro(text,'TILEMEGA_RESIDENCY_CAP','0'),placement_macro='0',
           extra=['MIDPOINT_REFINE=1'] if refine else [])
 binary=root/'bin/selected'
 if not binary.exists() and measure.build(root,'real','selected',spec,arch):
  raise RuntimeError('compile failed')
 folder=out/f'seq{seq}';folder.mkdir(parents=True,exist_ok=True)
 digest=hashlib.sha256(binary.read_bytes()).hexdigest();session=str(time.time_ns())
 samples=[]
 for i in range(rounds):
  log=folder/f'r{i}.log'
  if log.exists():raise RuntimeError('refusing overwrite '+str(log))
  env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
  fixture=measure.fixture(model,seq,past) if model else root/'fixture'
  command=[str(binary),str(fixture)]
  with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
   r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=3600)
  log.write_text(r.stdout+r.stderr)
  log.with_suffix('.json').write_text(json.dumps(dict(command=command,round=i,session=session,
   started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=r.returncode,
   binary_sha256=digest,seq=seq,past=past))+'\n')
  samples.append(measure.timing(log));print('TIMING',root.name,seq,i+1,'/',rounds,flush=True)
 keys=('l05_ms','l1_ms','l2_ms')
 row=dict(cell=root.name,model=model or 'llama',seq=seq,past=past,rounds=rounds,
          midpoint_refine=int(refine),floor_ns=floor_ns(root),binary_sha256=digest)
 for k in keys:
  v=[s[k] for s in samples]
  row[k+'_median']=statistics.median(v);row[k+'_min']=min(v);row[k+'_max']=max(v)
 # Ratios are taken per round and then summarized: the levels are paired inside
 # a process, so a ratio of medians would discard that pairing.
 row['l2_over_l1_median']=statistics.median(s['l2_ms']/s['l1_ms'] for s in samples)
 row['l1_over_l05_median']=statistics.median(s['l1_ms']/s['l05_ms'] for s in samples)
 row['l2_over_l05_median']=statistics.median(s['l2_ms']/s['l05_ms'] for s in samples)
 if row['floor_ns']:
  row['l2_over_floor_median']=statistics.median(
   s['l2_ms']*1e6/row['floor_ns'] for s in samples)
 (folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
 return row

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--roots',nargs='*',type=Path,default=[])
 ap.add_argument('--collect',action='store_true',
                 help='rebuild timing.tsv from the summary.json each seq already wrote')
 ap.add_argument('--rounds',type=int,default=25);ap.add_argument('--arch',default='sm_89')
 ap.add_argument('--out',type=Path,default=HERE/'timing')
 ap.add_argument('--model',default=None,help='reference-model fixture to feed instead of <root>/fixture')
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
 for root in a.roots:rows.append(cell(root,a.rounds,a.out,a.arch,a.model))
 # A cell's rounds are run once and never overwritten, so the table is assembled
 # from every summary.json under --out rather than from this invocation alone;
 # otherwise each run would truncate the table to its own roots.
 done={r['cell']:r for r in rows}
 for s in sorted(a.out.glob('seq*/summary.json')):
  r=json.loads(s.read_text())
  # Rows written before D-d generalized this script carry no model field; the
  # anchored model is the only one that ran into --out's default directory.
  r.setdefault('model',a.model or 'llama');r.setdefault('midpoint_refine',1)
  done.setdefault(r['cell'],r)
 rows=sorted(done.values(),key=lambda r:r['seq'])
 columns=list(max(rows,key=len))
 with (a.out/'timing.tsv').open('w') as f:
  f.write('\t'.join(columns)+'\n')
  for r in rows:f.write('\t'.join(str(r.get(c,'')) for c in columns)+'\n')
 for r in rows:
  # the anchored model is D-c's subject; the two reference models are D-d's
  print('D-c' if r['model']=='llama' else 'D-d',r['model'],'seq',r['seq'],'l05',round(r['l05_ms_median'],3),'l1',round(r['l1_ms_median'],3),
        'l2',round(r['l2_ms_median'],3),'l2/l1',round(r['l2_over_l1_median'],4),
        'l2/floor',round(r.get('l2_over_floor_median',0),4),flush=True)
if __name__=='__main__':main()
