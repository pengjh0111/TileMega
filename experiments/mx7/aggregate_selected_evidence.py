"""Append per-plan resource queries and candidate/final correctness evidence."""
import argparse,csv,hashlib,json,pathlib,re
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=lambda p:dict(path=str(p),sha256=sha(p))
def append(path,rows,keys):
 path=pathlib.Path(path);old=list(csv.DictReader(path.open()));fields=list(old[0]);seen={tuple(r[k]for k in keys)for r in old}
 fresh=[r for r in rows if tuple(str(r[k])for k in keys)not in seen]
 if fresh:
  with path.open('a',newline='')as f:csv.DictWriter(f,fieldnames=fields,lineterminator='\n').writerows(fresh)
 return len(fresh)
resources=[];correct=[]
for cell in sorted(a.root.glob('tp*/B*')):
 t=int(cell.parent.name[2:]);b=int(cell.name[1:])
 builds=list(cell.glob('rank*/build'))+([cell/'final']if(cell/'final').exists()else[])
 for folder in builds:
  so=folder/'plan.so';report=pathlib.Path(str(so)+'.resources.json');source=pathlib.Path(str(so)+'.source.json')
  if not report.exists()or not source.exists():continue
  r=json.loads(report.read_text());k=r['kernel'];s=json.loads(source.read_text());q=folder/'occupancy/result.json';query=json.loads(q.read_text())if q.exists()else None
  good=bool(query and query['qualified']);kind='selected'if folder.name=='final'else'candidate'
  if good:
   assert all(x['resident_ctas']==1 and x['attributes']['localSizeBytes']==x['attributes']['sharedSizeBytes']==0 for x in query['ranks'])
  cu=pathlib.Path(str(so)+'.cu');core=int(re.search(r'kBodySmemBytes\s*=\s*(\d+)',cu.read_text())[1]);smem=(core+b*4+15)//16*16+16
  resources.append(dict(record_type=kind+'_actual_residency'if good else kind+'_resource_rejected',evidence='verified',model='qwen3_1_7b',batch=b,phase='decode',tp=t,threads=512,MTreg=k['registers'],STreg=k['scalar_registers'],stack_bytes=k['stack_bytes'],static_smem_bytes=k['static_smem_bytes'],dynamic_smem_bytes=smem,resident_ctas=1 if good else'',query_status='all_ranks_pass:'+str(q)if good else'CPU_rejected_or_query_failed',code_bytes=r['code_bytes'],source_head=s['head'],artifact=str(so),artifact_sha256=sha(so),resources_json=json.dumps(ref(report)),source_snapshot_json=json.dumps(ref(source))))
  root=folder/'correctness';hf=root/'hf_check.json';native=root/'c1_tokens/check.json'
  if not hf.exists()or not native.exists():continue
  h=json.loads(hf.read_text());n=json.loads(native.read_text());assert h['pass']and n['rank_tokens_identical']
  correct.append(dict(record_type=kind+'_C1',evidence='verified',test='hf_teacher_forcing',cases=h['positions'],status='pass',fresh_processes=1,scope=f'TP{t} B{b} {kind};ratio={h["gap_le_0_5_ratio"]};max_gap={h["max_gap"]}',gpu_indices=','.join(map(str,n['gpu_indices'])),source_head=n['host_source_identity']['head'],test_sha256=n['host_source_identity']['source_files']['python/tilemega/serving/hf_check.py'],raw_log=str(hf),raw_log_sha256=sha(hf)))
  files=sorted(root.glob('c2_process*/check.json'));expected=5 if kind=='selected'else 2
  if len(files)!=expected:continue
  xs=[json.loads(f.read_text())for f in files]
  assert len({x['PID']for x in xs})==expected and len({x['token_sha256']for x in xs})==1
  assert all(x['pass']and x['rank_tokens_identical']and x['steps']==1024 for x in xs)
  assert all(sha(f.parent/'tokens.json')==x['token_sha256']for f,x in zip(files,xs))
  proof=root/'C2_qualification.json'
  data=dict(pass_=True,tp=t,batch=b,fresh_processes=expected,token_sha256=xs[0]['token_sha256'],checks=[ref(f)for f in files])
  if proof.exists():assert json.loads(proof.read_text())==data
  else:proof.write_text(json.dumps(data,indent=2)+'\n')
  correct.append(dict(record_type=kind+'_C2',evidence='verified',test='tp_tokens_1024',cases=b*1024,status='pass',fresh_processes=expected,scope=f'TP{t} B{b} {kind};all-rank exact tokens',gpu_indices=','.join(map(str,xs[0]['gpu_indices'])),source_head=xs[0]['host_source_identity']['head'],test_sha256=xs[0]['host_source_identity']['source_files']['python/tilemega/serving/check_tp.py'],raw_log=str(proof),raw_log_sha256=sha(proof)))
print('resources appended',append('docs/results/MX7_resources_c550.csv',resources,['record_type','artifact']))
print('correctness appended',append('docs/results/MX7_tp_correctness_c550.csv',correct,['record_type','raw_log']))
