"""Finish incomplete cost chunks on available cards without changing samples."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--cards',default='3,4,5');a=p.parse_args()
original=a.original.resolve();a.out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();ref=lambda p:dict(path=str(pathlib.Path(p).resolve()),sha256=sha(p))
plan=json.loads((original/'PLAN.json').read_text());driver=plan['driver']['path'];assert sha(driver)==plan['driver']['sha256']
cards=list(map(int,a.cards.split(',')));assert len(set(cards))==len(cards)and set(cards)<=set(range(8))
completed=[];pending=[]
for number,entry in enumerate(plan['jobs']):
 folder=original/f'job{number:04d}';key=folder/'keys.json';assert ref(key)==entry['keys'];records=[]
 for round_ in [1,2,3]:
  run=folder/f'round{round_}';proc=run/'PROCESS.json';raw=run/'raw.jsonl';guard=run/'guard/guard_result.json'
  if not all(p.exists()for p in[proc,raw,guard]):break
  if json.loads(proc.read_text())['returncode']or json.loads(guard.read_text())['code']:break
  records.append(dict(round=round_,raw=ref(raw),process=ref(proc),guard=ref(guard)))
 if len(records)==3:completed.append((number,dict(keys=ref(key),records=records)))
 else:pending.append((number,key))
(a.out/'DECISION.json').write_text(json.dumps(dict(original=ref(original/'PLAN.json'),accepted_complete_chunks=[n for n,_ in completed],replaced_incomplete_chunks=[n for n,_ in pending],policy='Choose solely by three-process completion before reading performance values; retain all old raw samples.',runner=ref(__file__)),indent=2)+'\n');shutil.copy2(__file__,a.out/'runner.py')
print('retained',len(completed),'replaced',len(pending),flush=True)
def work(card,jobs):
 rows=[]
 for number,key in jobs:
  folder=a.out/f'job{number:04d}';folder.mkdir();records=[]
  for round_ in [1,2,3]:
   run=folder/f'round{round_}';run.mkdir();raw=run/'raw.jsonl'
   cmd=['scripts/mx_gpu_job.sh',str(card),'900',str(run/'guard'),sys.executable,driver,'--keys',str(key),'--out',str(raw),'--round',str(round_)]
   start=time.time()
   with(run/'stdout.log').open('w')as log:
    child=subprocess.Popen(cmd,env=dict(os.environ,PYTHONPATH=str(pathlib.Path.cwd()/'python'),NEEDS_MIB='12288'),stdout=log,stderr=subprocess.STDOUT);code=child.wait()
   proc=run/'PROCESS.json';proc.write_text(json.dumps(dict(argv=cmd,card=card,PID=child.pid,start=start,end=time.time(),returncode=code),indent=2)+'\n')
   if code:raise RuntimeError('rebalanced cost job failed; preserve '+str(run))
   records.append(dict(round=round_,raw=ref(raw),process=ref(proc),guard=ref(run/'guard/guard_result.json')))
  rows.append((number,dict(keys=ref(key),records=records)));print('rebalanced chunk',number,'card',card,flush=True)
 return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards))as pool:
 futures=[pool.submit(work,card,pending[i::len(cards)])for i,card in enumerate(cards)]
 for f in futures:completed.extend(f.result())
request=dict(plan,jobs=[row for _,row in sorted(completed)],rebalance=ref(a.out/'DECISION.json'))
with(original/'REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
print('complete source-bound cost request',len(completed),flush=True)
