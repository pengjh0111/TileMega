#!/usr/bin/env python3
"""§7.2 D-b: top-k quality under §6 C1-c's caliber, on the six cells.

C1-c asks how good the search's own shortlist is, not how well the model ranks:

  ratio = (fastest measured among the top three)
          / (fastest measured among every candidate the search evaluated)

so the denominator needs a buildable source per evaluated candidate, which is
what `--dump-evaluated` writes. Every candidate's entry is its own best plan --
the residency and placement the search would have taken had that candidate won
-- so the comparison is between choices, not between arbitrary plans.

Arms are rotated within each round, and each round is a fresh process, so an
ordering or a drift affects every arm alike.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,re,statistics,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure,search as jsearch
TARGET=REPO/'docs/experiments/COSTMODEL/event_fit/target.json'
DOMAIN=REPO/'docs/experiments/COSTMODEL/event_fit/search_domain.json'
HOP=REPO/'docs/experiments/SIMULATOR/hop_ns.tsv'
CELLS=[('gqa2',4),('gqa2',128),('mha4',4),('mha4',128),('real',4),('real',128)]

def exported(model,seq):
 if model!='real':return jsearch.exported(model)
 return REPO/f'docs/experiments/REALMODEL/raw/work/r2sim_s{seq}/model.json'

def table(path):
 rows=[l.split('\t') for l in path.read_text().strip().splitlines()]
 head={k:i for i,k in enumerate(rows[0])}
 return [{k:r[i] for k,i in head.items()} for r in rows[1:]]

def solve(cell,model,seq,capacity):
 out=cell/'auto.cu'
 if out.exists():return
 cell.mkdir(parents=True,exist_ok=True)
 command=[str(REPO/'build-portable/tools/tilemega-compile'),str(exported(model,seq)),str(out),
  '--solve',str(TARGET),'--seq',str(seq),'--past','3','--search-capacity',str(capacity),
  '--search-domain',str(DOMAIN),'--dump-cg',str(cell/'auto.mlir'),'--hop-curve',str(HOP),
  '--dump-evaluated','1']
 start=time.time_ns()
 with (cell/'search.log').open('w') as f:
  r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
 (cell/'search.json').write_text(json.dumps(dict(command=command,exit_code=r.returncode,
  elapsed_ns=time.time_ns()-start,head=subprocess.check_output(['git','rev-parse','HEAD'],
  cwd=REPO,text=True).strip()),indent=2)+'\n')
 if r.returncode:raise RuntimeError('search failed; see '+str(cell/'search.log'))

def arms(cell):
 """arm name -> (key, source) for every evaluated candidate."""
 return {'cand'+r['index']:r for r in table(cell/'auto.cu.evaluated.tsv')}

def spec(row):
 text=Path(row['source']).read_text()
 macro=lambda n,d:(lambda m:m[1] if m else d)(__import__('re').search(r'^#define '+n+r' (\d+)$',text,__import__('re').M))
 return dict(source=row['source'],kappa=macro('TILEMEGA_EVENT_KAPPA','1'),
             residency=macro('TILEMEGA_RESIDENCY_CAP','0'),placement_macro='0')

def run(cell,model,seq,arm,folder,round_,order,session):
 """Run one round; return the harness verdict instead of raising on a mismatch.

 A candidate the device disagrees with is a measurement result, not a campaign
 failure: C1-c compares the shortlist against the evaluated candidates that were
 actually measured, so the caller drops such a candidate and reports coverage.
 An existing log is read back rather than re-run, so a resumed campaign keeps the
 round it already recorded."""
 folder.mkdir(parents=True,exist_ok=True);log=folder/f'r{round_}.log'
 if log.exists():return 'RESULT status=PASS' in log.read_text()
 binary=cell/'bin'/arm
 env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
 command=[str(binary),str(measure.fixture(model,seq))]
 with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
  r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=900)
 log.write_text(r.stdout+r.stderr)
 log.with_suffix('.json').write_text(json.dumps(dict(command=command,round=round_,order=order,
  session=session,started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=r.returncode,
  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))+'\n')
 return not r.returncode and 'RESULT status=PASS' in r.stdout

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('action',choices=['search','build','measure','report'])
 ap.add_argument('--out',type=Path,default=HERE/'topk');ap.add_argument('--rounds',type=int,default=25)
 ap.add_argument('--capacity',type=int,default=12);ap.add_argument('--arch',default='sm_89')
 ap.add_argument('--jobs',type=int,default=8)
 ap.add_argument('--cells',nargs='*',default=[f'{m}_s{s}' for m,s in CELLS])
 a=ap.parse_args();session=str(time.time_ns())
 cells=[(m,s) for m,s in CELLS if f'{m}_s{s}' in a.cells]
 if a.action=='search':
  for m,s in cells:
   solve(a.out/f'{m}_s{s}',m,s,a.capacity);print('SEARCHED',m,s,flush=True)
  return
 if a.action=='build':
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
   jobs=[]
   for m,s in cells:
    cell=a.out/f'{m}_s{s}'
    for arm,row in arms(cell).items():
     if (cell/'bin'/arm).exists():continue
     jobs.append(pool.submit(measure.build,cell,m,arm,spec(row),a.arch))
   failures=sum(j.result()!=0 for j in jobs)
  if failures:raise RuntimeError(f'{failures} builds failed')
  return
 if a.action=='measure':
  for m,s in cells:
   cell=a.out/f'{m}_s{s}';names=sorted(arms(cell),key=lambda n:int(n[4:]))
   dropped={}
   for i in range(a.rounds):
    for j in range(len(names)):
     arm=names[(i+j)%len(names)]
     if arm in dropped:continue
     if not run(cell,m,s,arm,cell/'measure'/arm,i,j,session):
      # The candidate stops being timed the moment it disagrees with the golden
      # output, and the exclusion is written where the report can read it.
      dropped[arm]=str(cell/'measure'/arm/f'r{i}.log')
      print('EXCLUDED',m,s,arm,dropped[arm],flush=True)
   (cell/'excluded.tsv').write_text('arm\tfirst_mismatch_log\n'+
    ''.join(f'{k}\t{v}\n' for k,v in sorted(dropped.items())))
   print('MEASURED',m,s,len(names)-len(dropped),'of',len(names),'arms',a.rounds,'rounds',flush=True)
  return
 rows=[]
 for m,s in cells:
  cell=a.out/f'{m}_s{s}';evaluated=arms(cell)
  top3={r['key'] for r in table(cell/'auto.cu.top3.tsv')}
  median={}
  # A candidate that disagreed with the golden output was dropped mid-campaign,
  # so its one recorded round is not a timing sample; C1-c's ratio is taken over
  # the candidates that were actually measured and the coverage says how many.
  excluded={l.split('\t')[0] for l in (cell/'excluded.tsv').read_text().splitlines()[1:]
            if l.strip()} if (cell/'excluded.tsv').is_file() else set()
  for arm,row in evaluated.items():
   if arm in excluded:continue
   logs=sorted((cell/'measure'/arm).glob('r*.log'),key=lambda p:int(p.stem[1:]))
   if not logs:continue
   median[arm]=statistics.median(measure.timing(p)['l2_ms'] for p in logs)
  shortlisted=[n for n in median if evaluated[n]['key'] in top3]
  # A cell whose arms have not been measured yet contributes no row; the table
  # says which cells it covers rather than silently standing in for all six.
  if not shortlisted:
   print('D-b-skip',f'{m}_s{s}','no measured shortlisted arm',flush=True);continue
  best_short=min(shortlisted,key=median.get);best_all=min(median,key=median.get)
  # Two counterfactuals over the very same measurements, so that the next step is
  # quantified rather than asserted. Both re-spend the three slots under a pure
  # ordering rule -- no new pricing, no re-ranking of anything the search did:
  #   spread  : three distinct geometries, kappa variants collapsed;
  #   spread4 : the same, after breaking exact (floor, predicted) ties by the
  #             larger kappa, which the cost model prices at zero and the device
  #             does not.
  bare=lambda k:re.sub(r'kappa\d+','',re.sub(r'r\d+$','',k))
  kap=lambda k:int(re.search(r'kappa(\d+)',k)[1])
  base=table(cell/'auto.cu.evaluated.tsv')
  def first3(rows):
   picked=[];seen=set()
   for r in rows:
    g=bare(r['key'])
    if g in seen:continue
    seen.add(g);picked.append(r['key'])
    if len(picked)==3:break
   return picked
  by_rank=sorted(base,key=lambda r:(float(r['floor_ns']),float(r['predicted_ns']),int(r['index'])))
  by_kappa=sorted(base,key=lambda r:(float(r['floor_ns']),float(r['predicted_ns']),-kap(r['key']),int(r['index'])))
  distinct=first3(by_rank);distinct4=first3(by_kappa)
  pick=lambda keys:min([n for n in median if evaluated[n]['key'] in set(keys)] or [best_short],key=median.get)
  best_spread=pick(distinct);best_spread4=pick(distinct4)
  rows.append(dict(cell=f'{m}_s{s}',evaluated=len(evaluated),measured=len(median),
   excluded_mismatch=','.join(sorted(excluded)) or '-',
   shortlisted_measured=len(shortlisted),rounds=a.rounds,
   best_shortlisted=evaluated[best_short]['key'],best_shortlisted_ms=median[best_short],
   best_evaluated=evaluated[best_all]['key'],best_evaluated_ms=median[best_all],
   ratio=median[best_short]/median[best_all],
   geometry_spread_keys=','.join(distinct),
   geometry_spread_ratio=median[best_spread]/median[best_all],
   kappa_first_spread_keys=','.join(distinct4),
   kappa_first_spread_ratio=median[best_spread4]/median[best_all]))
 # What the excluded candidates disagreed on, read back off their own logs: the
 # same buffer and element in every cell, which is what makes the exclusion a
 # property of the model's output rather than of one candidate.
 with (a.out/'mismatch.tsv').open('w') as f:
  f.write('cell\tarm\tkey\tdiff_line\tlog\n')
  for m,s in cells:
   cell=a.out/f'{m}_s{s}';record=cell/'excluded.tsv'
   if not record.is_file():continue
   evaluated=arms(cell)
   for line in record.read_text().splitlines()[1:]:
    if not line.strip():continue
    arm,log=line.split('\t')
    diff=next((l for l in Path(log).read_text().splitlines()
               if l.startswith('E2E_OUTPUT_DIFF') and 'mismatch=0' not in l),'')
    f.write(f"{m}_s{s}\t{arm}\t{evaluated.get(arm,{}).get('key','?')}\t{diff}\t{log}\n")
 columns=list(rows[0])
 with (a.out/'topk.tsv').open('w') as f:
  f.write('\t'.join(columns)+'\n')
  for r in rows:f.write('\t'.join(str(r[c]) for c in columns)+'\n')
 for r in rows:
  print('D-b',r['cell'],'ratio',round(r['ratio'],4),'coverage',
        f"{r['measured']}/{r['evaluated']}",'excluded',r['excluded_mismatch'],
        'best',r['best_evaluated'],flush=True)
  print('D-b-counterfactual',r['cell'],'spread',round(r['geometry_spread_ratio'],4),
        'spread+kappa',round(r['kappa_first_spread_ratio'],4),
        'over',r['kappa_first_spread_keys'],flush=True)
 print('D-b RESULT cells',len(rows),'passing',sum(r['ratio']<=1.05 for r in rows),flush=True)
if __name__=='__main__':main()
