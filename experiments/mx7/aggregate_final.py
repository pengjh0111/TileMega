"""Append source-bound MX-7 final/selection rows and publish selected pins."""
import argparse,csv,hashlib,json,pathlib,statistics,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
ROOT=pathlib.Path.cwd();head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def checked(ref):
 q=pathlib.Path(ref['path']);assert hashlib.sha256(q.read_bytes()).hexdigest()==ref['sha256'],str(q)
 return q
sha=lambda q:hashlib.sha256(pathlib.Path(q).read_bytes()).hexdigest()
def append(path,fields,rows,key):
 path=pathlib.Path(path);old=list(csv.DictReader(path.open()))if path.exists()else[]
 seen={tuple(r[k]for k in key)for r in old};fresh=[r for r in rows if tuple(str(r[k])for k in key)not in seen]
 if fresh:
  with path.open('a',newline='')as f:
   w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
   if not old:w.writeheader()
   w.writerows(fresh)
 return len(fresh)
frozen={int(r['batch']):r for r in csv.DictReader(open('docs/results/MX5_llm_performance_snapshot_c550.csv'))if r['model']=='qwen3_1_7b'}
baselines={}
for r in csv.DictReader(open('docs/results/MX7_tp_baselines_c550.csv')):
 if r['record_type']=='final_two_fresh_processes'and r['status']=='pass':baselines[int(r['tp']),int(r['batch']),r['system']]=r
final=[];selection=[]
for file in sorted(a.root.glob('tp*/B*/results.json')):
 result=json.loads(file.read_text());t,b=result['tp'],result['batch'];timing=result['final_timing'];metrics=[json.loads(checked(ref).read_text())for ref in timing['measurements']]
 assert len(metrics)==2 and all(x['tp']==t and x['batch']==b and x['rank_tokens_identical']and x['timed_tokens_identical']for x in metrics)
 pin=checked(result['selected_pin']);pinvalue=json.loads(pin.read_text());assert pinvalue['mx_decode_v2']['purpose']=='manual'
 published=pathlib.Path(f'configs/pins/mx7/qwen3_1_7b_tp{t}_B{b}_selected.json');published.parent.mkdir(parents=True,exist_ok=True)
 if published.exists():assert published.read_bytes()==pin.read_bytes()
 else:published.write_bytes(pin.read_bytes())
 refs=result['resources'];binary=checked(refs['binary']);identity=json.loads(pathlib.Path(str(binary)+'.identity.json').read_text());receipt=json.loads(pathlib.Path(str(binary)+'.source.json').read_text())
 manifest=json.loads(pathlib.Path(str(binary)+'.plan.json').read_text())
 assert manifest['tp']==t and manifest['threads']==512 and manifest['watchdog']==0
 assert not any(manifest.get(k,False)for k in ['trace_stage','diag','tp_litmus'])
 assert all(x['execution_identity']['artifact_id']==identity['artifact_id']and not x['execution_identity']['trace']and not x['execution_identity']['spill']for x in metrics)
 occupancy=json.loads(checked(refs['occupancy']).read_text());assert occupancy['qualified']and len(occupancy['ranks'])==t
 for rank in occupancy['ranks']:assert rank['resident_ctas']==1 and rank['attributes']['localSizeBytes']==rank['attributes']['sharedSizeBytes']==0
 median=lambda k:statistics.median(x[k]for x in metrics)
 tp1=float(frozen[b]['tpot_ms']);tpot=median('tpot_seconds')*1000
 row=dict(record_type='final_two_fresh_processes',evidence='verified',model='qwen3_1_7b',tp=t,batch=b,gpu_indices=result['group'],contract='bf16_greedy',isl=64,osl=1024,prompt_mode=True,executor='L1_persistent',warmup=1,repeats=2,fresh_processes=2,tpot_ms=tpot,ttft_ms=median('ttft_seconds')*1000,e2e_ms=median('e2e_seconds')*1000,tokens_per_s=median('output_tokens_per_second'),tp1_frozen_tpot_ms=tp1,scaling_efficiency=tp1/(t*tpot),round1_tpot_ms=metrics[0]['tpot_seconds']*1000,round2_tpot_ms=metrics[1]['tpot_seconds']*1000)
 for engine in ['vllm','sglang','torchcompile']:
  ref=baselines.get((t,b,engine));value=float(ref['tpot_ms'])if ref else None
  row[engine+'_tpot_ms']=value if value is not None else'N/A';row['speedup_vs_'+engine]=value/tpot if value is not None else'N/A'
 available=[float(row[x+'_tpot_ms'])for x in ['vllm','sglang','torchcompile']if row[x+'_tpot_ms']!='N/A'];best=min(available)if available else None
 row.update(fastest_baseline_ms=best if best else'N/A',speedup_vs_fastest=best/tpot if best else'N/A',pin=str(published),pin_sha256=sha(published),cost_table_sha256=result['cost_table_sha256'],candidate_rank=pinvalue['evidence']['candidate_rank'],predicted_ns=pinvalue['evidence']['predicted_ns'],pred_over_meas=pinvalue['evidence']['predicted_ns']/(tpot*1e6),source_head=receipt['head'],artifact=str(binary),artifact_sha256=sha(binary),measurements_json=json.dumps(timing['measurements']),selection_results=str(file))
 if any((t,b,engine)not in baselines for engine in ['vllm','sglang','torchcompile']):raise ValueError('formal baseline matrix is incomplete for final reporting')
 final.append(row)
 for i,c in enumerate(result['candidates']):
  xs=[json.loads(checked(r).read_text())for r in c['timing']['measurements']];m=statistics.median(x['tpot_seconds']for x in xs)*1e9
  selection.append(dict(record_type='top4_candidate',evidence='verified',tp=t,batch=b,qualified_rank=i,candidate_rank=c['choice']['candidate_rank'],selected=i==result['selection']['selected'],predicted_ns=c['choice']['predicted_ns'],measured_ns=m,pred_over_meas=c['choice']['predicted_ns']/m,round1_ns=xs[0]['tpot_seconds']*1e9,round2_ns=xs[1]['tpot_seconds']*1e9,cost_table_sha256=result['cost_table_sha256'],pin_json=json.dumps(json.loads(checked(c['pin']).read_text())['mx_decode_v2'],sort_keys=True),evidence_json=json.dumps(dict(build=c['build'],correctness=c['correctness'],timing=c['timing'])),selection_results=str(file)))
if final:print('final rows appended',append('docs/results/MX7_tp_final_c550.csv',list(final[0]),final,['tp','batch','artifact_sha256']))
if selection:print('candidate rows appended',append('docs/results/MX7_tp_selection_c550.csv',list(selection[0]),selection,['tp','batch','candidate_rank','cost_table_sha256']))
print('qualified final scopes',len(final))
