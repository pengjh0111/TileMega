#!/usr/bin/env python3
"""§5.3 B2: per-producer-stage kappa on the two reference models.

Two arms per model, each solved by tilemega-compile with the JOINT2 command:
  searched  --per-stage-kappa 1   the solver's coordinate descent over stages;
                                  the table it delivers, if it moves off uniform
  forced    --stage-kappa CSV     a mixed table pinned from the command line, so
                                  the per-stage runtime path executes even when
                                  the search keeps uniform kappa
Every arm is built and run in fresh processes; the pass rate is the claim.
"""
import argparse,json,re,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

def solve(cell,m,seq,past,capacity,extra):
 source=cell/'auto.cu'
 if source.exists():return
 cell.mkdir(parents=True,exist_ok=True)
 command=[str(REPO/'build-portable/tools/tilemega-compile'),
  str(REPO/f'docs/experiments/SEQSCAN/raw/export/{m}.json'),str(source),
  '--solve',str(REPO/'docs/experiments/COSTMODEL/event_fit/target.json'),
  '--seq',str(seq),'--past',str(past),'--search-capacity',str(capacity),
  '--search-domain',str(REPO/'docs/experiments/COSTMODEL/event_fit/search_domain.json'),
  '--dump-cg',str(cell/'auto.mlir'),
  '--hop-curve',str(REPO/'docs/experiments/SIMULATOR/hop_ns.tsv'),*extra]
 start=time.time_ns()
 with (cell/'solve.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
 (cell/'solve.json').write_text(json.dumps(dict(command=command,exit_code=r.returncode,
  elapsed_ns=time.time_ns()-start,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()),indent=2)+'\n')
 if r.returncode:raise RuntimeError('solve failed; see '+str(cell/'solve.log'))

def macro(text,name):
 # The kappa table is a comma list with spaces, so the value is the rest of the
 # line and not the first token.
 m=re.search(r'^#define '+name+r' (.+)$',text,re.M)
 return re.sub(r'\s+','',m[1]) if m else ''

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--models',default='gqa2,mha4');ap.add_argument('--seq',type=int,default=4)
 ap.add_argument('--past',type=int,default=3);ap.add_argument('--rounds',type=int,default=50)
 ap.add_argument('--capacity',type=int,default=12);ap.add_argument('--arch',default='sm_89')
 ap.add_argument('--out',type=Path,default=HERE/'stage_kappa')
 ap.add_argument('--arms',default='searched,forced')
 ap.add_argument('--report',action='store_true',
  help='rewrite results.tsv from the recorded runs instead of running new ones')
 a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for m in a.models.split(','):
  base=out/f'{m}_s{a.seq}'
  searched=base/'searched';solve(searched,m,a.seq,a.past,a.capacity,['--per-stage-kappa','1'])
  # The table is indexed by projected stage, not by model stage: split-K adds
  # combine stages, so only the solver knows the length, and it reports it.
  stages=int(re.search(r'^SOLVE_STAGE_KAPPA stages=(\d+)',
                       (searched/'solve.log').read_text(),re.M)[1])
  table=','.join(str((1,2,4)[i%3]) for i in range(stages))
  forced=base/'forced';solve(forced,m,a.seq,a.past,a.capacity,['--stage-kappa',table])
  for arm in a.arms.split(','):
   cell=base/arm;text=(cell/'auto.cu').read_text();log=(cell/'solve.log').read_text()
   sk=re.search(r'^SOLVE_STAGE_KAPPA (.*)$',log,re.M)
   fields=dict(re.findall(r'(\w+)=(\S+)',sk[1])) if sk else {}
   spec=dict(source=str(cell/'auto.cu'),kappa=macro(text,'TILEMEGA_EVENT_KAPPA') or '1',
             residency=macro(text,'TILEMEGA_RESIDENCY_CAP') or '0',placement_macro='0',
             extra=['MIDPOINT_REFINE=1'])
   folder=cell/'correctness'
   if a.report:
    # Every recorded round is one fresh process; a failed one raised before
    # the next started, so the count of passing logs is the pass count.
    runs=[json.loads(p.read_text()) for p in sorted(folder.glob('r*.json'))]
    passing=sum(1 for r in runs if r['exit_code']==0 and
                'RESULT status=PASS' in (folder/f"r{r['round']}.log").read_text())
    if len(runs)!=a.rounds:raise RuntimeError(f'{folder}: {len(runs)} recorded rounds, expected {a.rounds}')
   else:
    if measure.build(cell,m,arm,spec,a.arch):raise RuntimeError('compile failed: '+str(cell))
    session=str(time.time_ns());passing=0
    for i in range(a.rounds):
     measure.run(cell,m,a.seq,arm,folder,i,i,session,past=a.past);passing+=1
     print('B2',m,arm,i+1,'/',a.rounds,flush=True)
   rows.append(dict(model=m,seq=a.seq,arm=arm,capacity=a.capacity,stages=stages,
    kappa=spec['kappa'],residency=spec['residency'],grid=macro(text,'TILEMEGA_SOLVED_GRID'),
    per_stage=macro(text,'TILEMEGA_EVENT_KAPPA_PER_STAGE') or '0',
    table=macro(text,'TILEMEGA_EVENT_KAPPA_TABLE') or 'uniform',
    moves=fields.get('moves','-'),uniform_ns=fields.get('uniform_ns','-'),
    per_stage_ns=fields.get('per_stage_ns','-'),rounds=a.rounds,passing=passing,
    binary_sha256=json.loads((cell/'build'/(arm+'.json')).read_text())['binary_sha256']))
 keys=list(rows[0]);results=out/'results.tsv'
 # Models may be run in separate invocations; a rerun replaces its own rows.
 kept=[r for r in (dict(zip(keys,l.split('\t'))) for l in results.read_text().splitlines()[1:])
       if not any((r['model'],r['seq'],r['arm'])==(n['model'],str(n['seq']),n['arm']) for n in rows)] if results.exists() else []
 rows=kept+rows
 results.write_text('\t'.join(keys)+'\n'+''.join('\t'.join(str(r[k]) for k in keys)+'\n' for r in rows))
 for r in rows:print('B2_RESULT',' '.join(f'{k}={r[k]}' for k in keys),flush=True)
if __name__=='__main__':main()
