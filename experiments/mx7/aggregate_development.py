"""Append complete MX-7 development correctness cohorts and residency queries."""
import argparse,csv,hashlib,json,pathlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
correct=pathlib.Path('docs/results/MX7_tp_correctness_c550.csv');old=list(csv.DictReader(correct.open()));fields=list(old[0]);seen={x['raw_log']for x in old};rows=[]
for cell in a.root.glob('dev_c2_litmus/tp*/B*'):
 t=int(cell.parent.name[2:]);b=int(cell.name[1:]);reference=None
 for kind in ['baseline','litmus']:
  files=sorted((cell/kind).glob('*/check.json'))
  if len(files)!=5:continue
  records=[json.loads(f.read_text())for f in files]
  if len({x['PID']for x in records})!=5 or not all(x['pass']and x['rank_tokens_identical']and x['steps']==1024 for x in records):raise ValueError('development C2/litmus failed')
  if len({x['token_sha256']for x in records})!=1:raise ValueError('development token hashes differ')
  if kind=='baseline':reference=records[0]['token_sha256']
  elif records[0]['token_sha256']!=reference:raise ValueError('litmus differs from baseline')
  summary=cell/kind/'qualification.json'
  if not summary.exists():summary.write_text(json.dumps(dict(tp=t,batch=b,kind=kind,pass_=True,fresh_processes=5,token_sha256=records[0]['token_sha256'],inputs=[dict(path=str(f),sha256=sha(f))for f in files]),indent=2)+'\n')
  if str(summary)in seen:continue
  rows.append(dict(record_type='development_C2'if kind=='baseline'else'development_litmus',evidence='verified',test='tp_tokens_1024'if kind=='baseline'else'tp_injected_delay',cases=b*1024,status='pass',fresh_processes=5,scope=f'TP{t} B{b}; exact tokens; each process one cohort',gpu_indices=','.join(map(str,records[0]['gpu_indices'])),source_head=records[0]['host_source_identity']['head'],test_sha256=records[0]['host_source_identity']['source_files']['python/tilemega/serving/check_tp.py'],raw_log=str(summary),raw_log_sha256=sha(summary)))
with correct.open('a',newline='')as f:csv.DictWriter(f,fieldnames=fields,lineterminator='\n').writerows(rows)
print('development cohorts appended',len(rows))
resource=pathlib.Path('docs/results/MX7_resources_c550.csv');old=list(csv.DictReader(resource.open()));fields=list(old[0]);seen={(x['record_type'],x['artifact'])for x in old};rows=[]
for file in a.root.glob('dev_no_dynamic_array/tp*/B*/occupancy/result.json'):
 item=json.loads(file.read_text());assert item['qualified'];folder=file.parent.parent;so=folder/'plan.so';kind='development_actual_residency'
 if(kind,str(so))in seen:continue
 r=json.loads(pathlib.Path(str(so)+'.resources.json').read_text());k=r['kernel'];source=pathlib.Path(str(so)+'.source.json');receipt=json.loads(source.read_text())
 for rank in item['ranks']:assert rank['resident_ctas']==1 and rank['attributes']['localSizeBytes']==rank['attributes']['sharedSizeBytes']==0
 rows.append(dict(record_type=kind,evidence='verified',model='qwen3_1_7b',batch=folder.name[1:],phase='decode',tp=folder.parent.name[2:],threads=512,MTreg=k['registers'],STreg=k['scalar_registers'],stack_bytes=0,static_smem_bytes=0,dynamic_smem_bytes=item['ranks'][0]['dynamic_smem_bytes']if'dynamic_smem_bytes'in item['ranks'][0]else json.loads((folder/'plan.so.identity.json').read_text())['shared_memory_bytes'],resident_ctas=1,query_status='all_ranks_pass:'+str(file),code_bytes=r['code_bytes'],source_head=receipt['head'],artifact=str(so),artifact_sha256=sha(so),resources_json=json.dumps(dict(path=str(folder/'plan.so.resources.json'),sha256=sha(folder/'plan.so.resources.json'))),source_snapshot_json=json.dumps(dict(path=str(source),sha256=sha(source)))))
if rows:
 with resource.open('a',newline='')as f:csv.DictWriter(f,fieldnames=fields,lineterminator='\n').writerows(rows)
print('development residency rows appended',len(rows))
