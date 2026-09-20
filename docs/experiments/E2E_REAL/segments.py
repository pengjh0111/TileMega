#!/usr/bin/env python3
"""§5.4 B3: segmented geometry inside a finite theta interval.

One `tilemega-compile` per model produces both arms from the same search, so
the comparison is not two invocations:

  fixed       auto.cu.fixed.cu, one variant -- the geometry W2 picks at the
              interval's upper endpoint, priced across the whole interval
  segmented   auto.cu, two variants -- the best cut of the interval into two
              geometries, each variant carrying its own sub-range of the plan
              table while both advertise the whole interval in the macros

then, on the segmented build,

  segment_proof  the S5 ISL certificate at every integer point of every segment
  segment_check  every point's table re-solved on its own segment's graph,
                 endpoints and interior alike

and both arms are built and run in fresh processes at the seqs the fixtures
cover inside the interval.  The pass rate is the claim; the benefit is the
predicted sum over the interval and the measured end-to-end time.
"""
import argparse,json,re,statistics,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

TARGET=REPO/'docs/experiments/COSTMODEL/event_fit/target.json'
HOP=REPO/'docs/experiments/SIMULATOR/hop_ns.tsv'

def record(path,command,start,extra=None):
 path.write_text(json.dumps(dict(command=[str(x) for x in command],start_ns=start,
  elapsed_ns=time.time_ns()-start,head=subprocess.check_output(
   ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),**(extra or {})),indent=2)+'\n')

def shell(name,cell,command,log):
 start=time.time_ns()
 with (cell/log).open('w') as f:
  r=subprocess.run([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
 record(cell/(Path(log).stem+'.json'),command,start,dict(exit_code=r.returncode))
 if r.returncode:raise RuntimeError(name+' failed; see '+str(cell/log))

def solve(cell,m,end,past,capacity,candidates):
 if (cell/'auto.cu').exists():return
 cell.mkdir(parents=True,exist_ok=True)
 shell('solve',cell,[REPO/'build-portable/tools/tilemega-compile',
  REPO/f'docs/experiments/SEQSCAN/raw/export/{m}.json',cell/'auto.cu',
  '--solve',TARGET,'--seq',end,'--past',past,'--seq-begin',1,
  '--segments',2,'--segment-candidates',candidates,'--search-capacity',capacity,
  '--search-domain',REPO/'docs/experiments/COSTMODEL/event_fit/search_domain.json',
  '--hop-curve',HOP,'--dump-cg',cell/'auto.mlir'],'solve.log')

def proof(cell,tool,pieces,end,jobs):
 """One certificate per process: the points are independent and each costs
 minutes, so the interval is proved `jobs` points at a time and the per-point
 tables are concatenated into one proofs.tsv."""
 if (cell/'proof'/'proofs.tsv').exists():return
 pending=list(range(1,end+1));running={};start=time.time_ns()
 while pending or running:
  while pending and len(running)<jobs:
   seq=pending.pop(0);out=cell/'proof'/f'p{seq}';out.mkdir(parents=True,exist_ok=True)
   command=[str(tool),str(out),'--only',str(seq),*map(str,pieces)]
   running[seq]=subprocess.Popen(command,stdout=(out/'proof.log').open('w'),
    stderr=subprocess.STDOUT,cwd=REPO)
  for seq,p in list(running.items()):
   if p.poll() is None:continue
   del running[seq]
   if p.returncode:raise RuntimeError(f'segment_proof failed at seq {seq}; see {cell}/proof/p{seq}/proof.log')
   print('B3 proof',cell.name,'seq',seq,'done',flush=True)
  time.sleep(5)
 rows=[];header=None
 for seq in range(1,end+1):
  lines=(cell/'proof'/f'p{seq}'/'proofs.tsv').read_text().splitlines()
  header=lines[0];rows+=lines[1:]
 if len(rows)!=end:raise RuntimeError(f'{len(rows)} proved points for {end}')
 (cell/'proof'/'proofs.tsv').write_text(header+'\n'+'\n'.join(rows)+'\n')
 (cell/'proof'/'coverage.tsv').write_text((cell/'proof'/'p1'/'coverage.tsv').read_text())
 record(cell/'proof.json',[str(tool),'--only','<seq>',*map(str,pieces)],start,
  dict(jobs=jobs,points=end,failed=sum(1 for r in rows if not r.endswith('PASS'))))
 if any(not r.endswith('PASS') for r in rows):raise RuntimeError('a segment point failed its proof: '+str(cell))

def summary(cell):
 line=re.search(r'^SEGMENT_SUMMARY (.*)$',(cell/'solve.log').read_text(),re.M)
 if not line:raise RuntimeError('no SEGMENT_SUMMARY in '+str(cell/'solve.log'))
 return dict(re.findall(r'(\w+)=(\S+)',line[1]))

def fixtures(root,m,seqs,past):
 """Fixtures for interior seqs are generated beside the round, never into the
 shared SEQSCAN tree, so the campaign adds no file to another round's evidence."""
 made=[]
 for s in seqs:
  dest=root/'fixture'/f'{m}_s{s}_p{past}'
  if dest.exists():continue
  shared=REPO/f'docs/experiments/SEQSCAN/raw/fixture/{m}_s{s}_p{past}'
  export=REPO/f'docs/experiments/SEQSCAN/raw/export/{m}'
  dest.parent.mkdir(parents=True,exist_ok=True)
  if shared.exists():
   subprocess.run(['cp','-r',str(shared),str(dest)],check=True)
  elif m=='gqa2':
   subprocess.run([sys.executable,str(REPO/'docs/experiments/E2E/prepare_e2e.py'),
    '--vh-raw',str(export),'--out',str(dest),'--seq',str(s),'--past',str(past)],
    cwd=REPO,check=True,stdout=subprocess.DEVNULL)
  else:
   subprocess.run([sys.executable,str(REPO/'docs/experiments/P3_GENERALIZATION/prepare_fixture.py'),
    '--repo',str(REPO),'--program',str(export/'exported_program.pt2'),'--out',str(dest),
    '--seq',str(s),'--past',str(past)],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
  made.append(s)
 return made

def macro(text,name):
 m=re.search(r'^#define '+name+r' (.+)$',text,re.M)
 return re.sub(r'\s+','',m[1]) if m else ''

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--models',default='gqa2,mha4');ap.add_argument('--end',type=int,default=16)
 ap.add_argument('--past',type=int,default=3);ap.add_argument('--rounds',type=int,default=50)
 ap.add_argument('--timing-rounds',type=int,default=20)
 ap.add_argument('--jobs',type=int,default=16)
 ap.add_argument('--capacity',type=int,default=12);ap.add_argument('--candidates',type=int,default=4)
 ap.add_argument('--arch',default='sm_89');ap.add_argument('--tools',type=Path,required=True)
 ap.add_argument('--inputs',type=Path,default=Path('/root/r7_work/b3/inputs'))
 ap.add_argument('--out',type=Path,default=HERE/'segments')
 a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 rows=[];timings=[]
 for m in a.models.split(','):
  cell=out/f'{m}_i1_{a.end}'
  solve(cell,m,a.end,a.past,a.capacity,a.candidates)
  fields=summary(cell);cut=int(fields['cut'])
  pieces=[cell/'auto.mlir']+sorted(cell.glob('auto.segment*.mlir'))
  proof(cell,a.tools/'segment_proof',pieces,a.end,a.jobs)
  shell('segment_check',cell,[a.tools/'segment_check',TARGET,cell/'check',*pieces],'check.log')
  # The fixtures decide which points can be run: the interval is dense but the
  # exported programs are not, so the campaign runs the covered seqs and says so.
  covered=sorted({s for s in (1,cut-1,cut,a.end,4) if s and 1<=s<=a.end})
  made=fixtures(a.inputs.resolve(),m,covered,a.past)
  import os;os.environ['R5_INPUT_ROOT']=str(a.inputs.resolve())
  arms={'segmented':cell/'auto.cu','fixed':cell/'auto.cu.fixed.cu'}
  for arm,src in arms.items():
   text=src.read_text()
   spec=dict(source=str(src),kappa=macro(text,'TILEMEGA_EVENT_KAPPA') or '1',
    residency=macro(text,'TILEMEGA_RESIDENCY_CAP') or '0',placement_macro='0',
    extra=['MIDPOINT_REFINE=1'])
   if measure.build(cell,m,arm,spec,a.arch):raise RuntimeError('compile failed: '+str(cell))
  session=str(time.time_ns())
  for seq in covered:
   for arm in arms:
    folder=cell/'correctness'/f's{seq}'/arm;passing=0
    for i in range(a.rounds):
     measure.run(cell,m,seq,arm,folder,i,i,session,past=a.past);passing+=1
     print('B3',m,arm,'seq',seq,i+1,'/',a.rounds,flush=True)
    rows.append(dict(model=m,interval=f'1..{a.end}',seq=seq,arm=arm,
     segment=0 if (not cut or seq<cut) else 1,rounds=a.rounds,passing=passing,
     binary_sha256=json.loads((cell/'build'/(arm+'.json')).read_text())['binary_sha256']))
   # Paired timing, arms alternating inside one session so drift hits both.
   samples={arm:[] for arm in arms}
   for i in range(a.timing_rounds):
    for arm in (arms if i%2==0 else list(arms)[::-1]):
     folder=cell/'timing'/f's{seq}'/arm
     measure.run(cell,m,seq,arm,folder,i,i,session,past=a.past)
     samples[arm].append(measure.timing(folder/f'r{i}.log')['l2_ms'])
   timings.append(dict(model=m,seq=seq,rounds=a.timing_rounds,
    segmented_l2_ms=statistics.median(samples['segmented']),
    fixed_l2_ms=statistics.median(samples['fixed']),
    ratio=statistics.median(samples['segmented'])/statistics.median(samples['fixed'])))
  (cell/'summary.json').write_text(json.dumps(dict(fields,covered=covered,
   generated_fixtures=made),indent=2)+'\n')
  print('B3_SEGMENT',m,' '.join(f'{k}={v}' for k,v in fields.items()),flush=True)
 def write(path,data):
  if not data:return
  keys=list(data[0]);mine={r['model'] for r in data}
  # Models run in separate invocations; a rerun replaces its own rows only.
  kept=[r for r in (dict(zip(keys,l.split('\t'))) for l in path.read_text().splitlines()[1:])
        if r['model'] not in mine] if path.exists() else []
  data=kept+data
  path.write_text('\t'.join(keys)+'\n'+''.join('\t'.join(str(r[k]) for k in keys)+'\n' for r in data))
 write(out/'correctness.tsv',rows);write(out/'timing.tsv',timings)
 for r in rows+timings:print('B3_RESULT',' '.join(f'{k}={v}' for k,v in r.items()),flush=True)
if __name__=='__main__':main()
