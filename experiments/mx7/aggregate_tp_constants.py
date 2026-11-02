"""Qualify measured TP constants, retaining noisy samples and their provenance."""
import argparse,collections,hashlib,json,os,pathlib,statistics,subprocess,sys
p=argparse.ArgumentParser(description=__doc__)
for name in ['measurements','builds','out','work']:p.add_argument('--'+name,type=pathlib.Path,required=True)
p.add_argument('--tp',type=int,choices=(2,4),action='append')
p.add_argument('--group',help='physical group for a single requested TP degree')
a=p.parse_args();a.work.mkdir(parents=True,exist_ok=False)
if a.group and (not a.tp or len(a.tp)!=1 or len(set(a.group.split(',')))!=a.tp[0]):p.error('--group requires exactly one matching --tp')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(pathlib.Path(p).resolve()),sha256=sha(p))
def read(folder,rounds):
 records=[]
 for number in rounds:
  path=folder/f'process{number}'/'result.json';x=json.loads(path.read_text())
  if x['returncode']or len(x['records'])!=9 or not all(r['pass']for r in x['records']):raise ValueError('constant process is not qualified')
  if sha(x['build']['binary'])!=x['build']['binary_sha256']:raise ValueError('constant binary changed')
  if sha(folder/'measure_driver.py')!=x['driver_sha256']:raise ValueError('constant harness changed')
  records.append((x,ref(path)))
 return records
result={'t_ar':[],'t_head_x':[]};history=[]
for tp in a.tp or [2,4]:
 for b in [1,2,4,8,16]:
  name=f'tp{tp}_B{b}';folder=a.measurements/name
  guard=folder/'guard/guard_result.json'
  if json.loads(guard.read_text())['code']:raise ValueError('constant guard failed')
  records=read(folder/'cohort',range(3));history.append(ref(guard))
  physical_group=a.group or ('1,3'if tp==2 else'1,3,4,5')
  if any(x['gpu_indices']!=physical_group for x,_ in records):raise ValueError('constant records used a different physical GPU group')
  if len({x['PID']for x,_ in records})!=3:raise ValueError('three fresh constant processes required')
  rows=[]
  for i in range(9):
   first=records[0][0]['records'][i]
   if (first['tp'],first['b'])!=(tp,b):raise ValueError('constant scope changed')
   vals=[x['records'][i]['median_ns']for x,_ in records];m=statistics.median(vals)
   rows.append(dict(key=dict(tp=tp,b=b,stage=first['stage'],tn=first['tn']),median_ns=m,spread=(max(vals)-min(vals))/max(abs(m),1e-9),samples=[dict(PID=x['PID'],input=r,median_ns=x['records'][i]['median_ns'],p90_ns=x['records'][i]['p90_ns'])for x,r in records]))
  groups=collections.defaultdict(list)
  for row in rows:groups[row['key']['stage']].append(row)
  eligible=[]
  for group in groups.values():
   best=min(row['median_ns']for row in group)
   eligible += [row for row in group if row['spread']>.02 and row['median_ns']<=best+abs(best)*.02]
  extras=[]
  if eligible:
   out=a.work/name
   cmd=['scripts/mx_gpu_group_job.sh',physical_group,'900',str(out/'guard'),sys.executable,'experiments/mx7/measure_tp_constants.py','--build',str(pathlib.Path(records[0][0]['build']['binary']).parent),'--out',str(out/'cohort'),'--processes','2','--round-start','3']
   out.mkdir();
   with(out/'run.log').open('w')as log:code=subprocess.run(cmd,env=dict(os.environ,NEEDS_MIB='2048'),stdout=log,stderr=subprocess.STDOUT).returncode
   if code:raise RuntimeError('constant noise supplement failed: '+str(out))
   extras=read(out/'cohort',range(3,5));history.append(ref(out/'guard/guard_result.json'))
  for i,row in enumerate(rows):
   if row in eligible:
    row['samples'] += [dict(PID=x['PID'],input=r,median_ns=x['records'][i]['median_ns'],p90_ns=x['records'][i]['p90_ns'])for x,r in extras]
   samples=row.pop('samples');vals=[s['median_ns']for s in samples];m=statistics.median(vals)
   if len({s['PID']for s in samples})!=len(samples):raise ValueError('constant processes are not independent')
   row.update(source_digest=records[0][0]['source_digest'],binary=records[0][0]['build']['binary_sha256'],median_ns=m,p90_ns=statistics.median(s['p90_ns']for s in samples),spread=(max(vals)-min(vals))/max(abs(m),1e-9),processes=len(samples),rounds=samples)
   row['noisy']=row['spread']>.02
   result['t_head_x'if row['key']['stage']=='head'else't_ar'].append(row)
  print('constant scope qualified',name,'supplements',len(eligible),flush=True)
a.out.write_text(json.dumps(result,indent=2)+'\n');(a.work/'PROVENANCE.json').write_text(json.dumps(dict(guards=history,output=ref(a.out)),indent=2)+'\n')
