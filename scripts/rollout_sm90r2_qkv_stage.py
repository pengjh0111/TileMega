#!/usr/bin/env python3
"""Source-bounded QKV response rollout with a dynamic shared fluid resource."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):return isinstance(x,(int,float))and not isinstance(x,bool)and math.isfinite(x)
def positive(x):return finite(x)and x>0

def fluid_completions(arrivals,work):
 """Equal-weight work sharing; times are relative ns, work is server ns."""
 require(positive(work)and all(finite(t)for t in arrivals),'invalid fluid requests')
 indexed=sorted(enumerate(arrivals),key=lambda p:(p[1],p[0]));active={};done={};i=0;now=indexed[0][1]if indexed else 0
 while i<len(indexed)or active:
  arrival=indexed[i][1]if i<len(indexed)else math.inf
  completion=now+min(active.values())*len(active)if active else math.inf
  later=min(arrival,completion);elapsed=later-now
  if active:
   decrement=elapsed/len(active)
   active={j:max(0,left-decrement)for j,left in active.items()}
  now=later
  if completion<=arrival:
   finished=[j for j,left in active.items()if left<=max(1e-9,work*1e-12)]
   require(bool(finished),'fluid completion made no progress')
   for j in finished:done[j]=now;del active[j]
  while i<len(indexed)and indexed[i][1]==now:
   j,_=indexed[i];active[j]=work;i+=1
 return[done[j]for j in range(len(arrivals))]

def source_domain(source):
 work=source['work'];require(len(work['stages'])==1,'one exact source stage required')
 stage=work['stages'][0];d=stage['service_descriptor'];domain=d['domain'];coords=domain['coordinates'];g=coords['gemms']['qkv']
 require((work['grid'],work['num_sms'],work['requested_residency'],stage['kind'],stage['layer'],stage['tasks'],stage['ring_slots'])==(132,132,1,'qkv',0,128,2),'unsupported QKV stage geometry')
 require((domain['batch'],domain['shapes'][0],g['tile_n'],g['slot_bytes'],g['slots'],g['warps_n'],g['warps_k'],g['register_stages'],coords['swiglu_u'])==(16,[4096,1024],32,32768,2,1,4,1,8),'unsupported body/layout')
 require(d['plan_binding']==dict(epilogue='store',input_norm=True,output_sums=False),'actual plan presence/epilogue mismatch')
 binding=d['source_gemm'];require(binding['family']=='qkv'and binding['stage']==1 and binding['gemm']==0 and binding['norm_ss']is not None and binding['ss_out']is None,'actual Program binding IDs missing or mismatched')
 require(stage['incoming_order']=='none'and stage['prefix_bytes']==0 and stage['l2_hint_bytes']==0 and stage['inactive_zero_bytes_per_cta']==0,'unsupported incoming/tail/initialization work')
 for c,owner in enumerate(stage['ctas']):
  require(owner['cta']==c and len(owner['tasks'])==(1 if c<128 else 0),'CTA ownership differs')
  for task in owner['tasks']:
   require(task['task']==c and len(task['blocks'])==2,'unsupported local task/block sequence')
   for q,b in enumerate(task['blocks']):
    require((b['sequence'],b['slot'],b['generation'],b['previous_reader'],b['weight_bytes'],b['activation_bytes'],b['prefix_bytes'],b['remainder_requests'],b['activation_arrivals'],b['empty_arrivals'])==(q,q,0,-1,32768,16384,0,1,32,128),'native full/empty/request contract differs')
 return stage

def load_prefed(collected,fresh=0,trial=1):
 """One paired trial vector, never separate marginal summary statistics."""
 collected=Path(collected);receipt=json.loads((collected/'receipt.json').read_text())
 require(receipt['complete_cohort']and receipt['numerical_passes']==5 and receipt['pending']==[],'five accepted prefed processes required')
 review=receipt['function_comparison'];require(review['production_function_byte_equal']and review['production_resource_metadata_equal'],'Consume equivalence missing')
 require('ILi16ELi4096ELi1024ELi32ELi512ELi2ELi1ELi4E'in review['comparison']['key']and'ConsumeILb1ELb0E'in review['comparison']['key'],'wrong actual Consume instantiation')
 evidence=next(p for p in receipt['processes']if p['fresh']==fresh)
 raw=collected/'raw'/f'f{fresh}-raw.csv';ctas=collected/'raw'/f'f{fresh}-ctas.csv'
 for filename,path in [('raw.csv',raw),('ctas.csv',ctas)]:
  matches=[h for p,h in evidence['receipts'].items()if p.endswith('/probe/'+filename)]
  require(matches==[sha(path)],'retained paired raw no longer matches accepted receipt')
 groups=defaultdict(list);places={}
 for row in csv.DictReader(raw.open()):
  if(int(row['active_ctas']),row['feed_mode'],int(row['observe_empty']),int(row['trial']))==(128,'prefed',1,trial):groups[int(row['cta'])].append({k:int(v)if k!='feed_mode'else v for k,v in row.items()})
 for r in csv.DictReader(ctas.open()):
  if(int(r['active_ctas']),r['feed_mode'],int(r['observe_empty']),int(r['trial']))==(128,'prefed',1,trial):places[int(r['cta'])]=int(r['smid'])
 require(set(groups)==set(range(128))and set(places)==set(range(132))and len(set(places.values()))==132,'incomplete paired role/placement vector')
 base=min(r['entry']for rows in groups.values()for r in rows);profiles={}
 for c,rows in groups.items():
  require(len(rows)==4 and{r['warp']for r in rows}==set(range(4)),'missing/duplicate warp')
  begin=min(r['entry']for r in rows);last_entry=max(r['entry']for r in rows);first_return=min(r['returned']for r in rows);end=max(r['returned']for r in rows)
  require(all(r['full_ready_mask']==3 and max(r['full0_confirmed'],r['full1_confirmed'])<=r['setup_acquired']<=r['entry']for r in rows),'prefeed not causally confirmed')
  require(len({tuple(r[k]for k in('empty0_observed_ready','empty1_observed_ready'))for r in rows})==1,'mixed observer vector')
  upper=[min(first_return,rows[0][f'empty{q}_observed_ready'])-begin for q in(0,1)]
  lower=[last_entry-begin]*2
  require(0<=lower[0]<=upper[0]<=upper[1]<=end-begin,'inconsistent paired progress/return')
  profiles[c]=dict(progress_lower_ns=lower,progress_upper_ns=upper,return_ns=end-begin,entry_skew_ns=begin-base,smid=places[c])
 return dict(profiles=profiles,smids=places,receipt_sha256=sha(collected/'receipt.json'),raw_sha256=sha(raw),fresh=fresh,trial=trial,reference_binary_sha256=review['reference_binary_sha256'],ownership_verified=receipt['ownership_verified'])

def resolve_policy(policy):
 if policy.get('cache_preparation')!='warm_touch':return None,['cache preparation outside exact warm A+B support is unpriced']
 require(policy['traffic']=='A+B'and policy['request_work_bytes']==49152,'exact QKV requires one coupled logical A+B request')
 if policy['status']in('failed_validation','rejected'):return None,['transfer model rejected by its retained validation']
 if policy['kind']not in('equal_weight_processor_sharing','effective_FIFO'):return None,['shared transfer architecture unsupported; coefficients cannot be relabeled']
 require(positive(policy['request_work_ns'])and finite(policy['packet_response_ns'])and policy['packet_response_ns']>=0 and finite(policy['propagation_observation_lag_ns'])and policy['propagation_observation_lag_ns']>=0,'invalid transfer model')
 require(policy.get('packet_queue_fraction',0)==0,'nonzero source packet queue feedback unsupported')
 pending=[]if policy['status']=='validated_native_stage'else['transfer policy is diagnostic/inferred, not a validated native stage service']
 return policy,pending

def rollout(source,initial,paired,policy,boundary=None,landmark='upper'):
 stage=source_domain(source);model,pending=resolve_policy(policy)
 if model is None:return dict(prediction_available=False,pending=pending)
 require(initial['initialized_before_entry']is True,'source ring initialization must be supplied, not silently free')
 states={int(c):v for c,v in initial['ctas'].items()};profiles={int(c):v for c,v in paired['profiles'].items()}
 require(set(states)==set(range(132))and set(profiles)==set(range(128)),'exact resident and active state vectors required')
 require(initial['shared_transfer_inflight']==[],'carried dynamic transfer work is unimplemented, not assumed empty')
 require(len({v['smid']for v in states.values()})==132,'residency1 requires one supplied resident CTA per SM')
 if initial['status']!='verified_native_stage':pending.append('initial role-entry/empty-credit/placement state is supplied diagnostic state')
 if initial['program_binary_sha256']!=paired['reference_binary_sha256']:pending.append('Consume response lacks exact equivalence to requested program binary')
 if policy.get('consume_stall_extension_verified')is not True:pending.append('prefed-to-stalled max-plus response extension is an inferred model')
 if policy.get('native_packet_applicability_verified')is not True:pending.append('literal packet model applicability to actual producer issue/SM coupling is unverified')
 requests=[];packets={};entries={}
 for c,v in sorted(states.items()):
  require(set(v)=={'smid','initialization_ready_ns','producer_entry_ns','consumer_entry_ns','empty_ready_ns'},'only initial role/credit state allowed, no future issue gates')
  require(all(finite(v[k])for k in('initialization_ready_ns','producer_entry_ns','consumer_entry_ns'))and len(v['empty_ready_ns'])==2 and all(finite(t)for t in v['empty_ready_ns']),'invalid initial role/credit state')
  require(max(v['empty_ready_ns'])<=v['initialization_ready_ns'],'first-use ring credits must complete by supplied initialization completion')
  if c>=128:continue
  clock=max(v['initialization_ready_ns'],v['producer_entry_ns']);entries[c]=max(v['initialization_ready_ns'],v['consumer_entry_ns'])
  for q in(0,1):
   start=max(clock,v['empty_ready_ns'][q]);end=start+model['packet_response_ns'];packets[c,q]=dict(start_ns=start,end_ns=end);requests.append((start,c,q));clock=end
 requests.sort()
 if model['kind']=='equal_weight_processor_sharing':
  done=fluid_completions([r[0]for r in requests],model['request_work_ns']);resource_final=max(done)
 else:
  free=initial['shared_resource_available_ns'];require(finite(free),'missing/invalid carried FIFO resource state');done=[]
  for start,_,_ in requests:
   admitted=max(start,free);free=admitted+model['request_work_ns'];done.append(admitted)
  resource_final=free
 ready={};source_gates=0
 for(_,c,q),time in zip(requests,done):
  response=time+model['propagation_observation_lag_ns'];end=packets[c,q]['end_ns'];source_gates+=response<end
  ready[c,q]=max(response,end);packets[c,q]['full_ready_ns']=ready[c,q]
 rows=[];milestones={};kinds=source['kinds']
 for c in range(128):
  p=profiles[c];progress=p[f'progress_{landmark}_ns'];total=p['return_ns']
  require(len(progress)==2 and all(finite(x)for x in progress)and 0<=progress[0]<=progress[1]<=total,'invalid paired connected response')
  clock=entries[c];offset=0;empty=[]
  for q in(0,1):
   clock=max(clock,ready[c,q])+progress[q]-offset;offset=progress[q];empty.append(clock)
  returned=clock+total-progress[-1]
  rows.append(dict(cta=c,smid=states[c]['smid'],entry_ns=entries[c],empty_ns=empty,returned_ns=returned,packets=[packets[c,q]for q in(0,1)]))
  milestones[c]=dict(empty=empty,returned=returned)
 # Only observable composite endpoints are attached to raw source IDs.
 # Finalize/epilogue internal milestones stay covered but unresolved individually.
 mapped={};upper_bounds={}
 for event_id,e in enumerate(source['flow']['events']):
  if e['stage']!=0:continue
  c=e['cta'];kind=e['kind'];q=e['sequence']
  if c in milestones and kind==kinds['empty']and q in(0,1):mapped[event_id]=milestones[c]['empty'][q]
  elif c in milestones and kind==kinds['full']and q in(0,1):mapped[event_id]=ready[c,q]
  elif c in milestones and kind==kinds['consumer_return']:upper_bounds[event_id]=milestones[c]['returned']
 body_end=max(r['returned_ns']for r in rows);stage_end=None
 if boundary is None:pending.append('producer generation publication, CTA join and V3 boundary services are unpriced')
 else:
  require(boundary['status']=='supplied_calibration'and all(finite(boundary[k])and boundary[k]>=0 for k in('producer_publication_ns','cta_join_ns','grid_arrival_ns','grid_arrival_occupancy_ns','grid_release_ns')),'invalid boundary services')
  free=boundary['initial_grid_resource_available_ns'];require(finite(free),'invalid carried counter state');arrivals=[]
  for c,v in states.items():
   producer=packets[c,1]['end_ns']+boundary['producer_publication_ns']if c<128 else v['initialization_ready_ns']
   consumer=milestones[c]['returned']if c<128 else v['initialization_ready_ns']
   arrivals.append((max(producer,consumer)+boundary['cta_join_ns'],c))
  complete=[]
  for t,c in sorted(arrivals):
   start=max(t,free);free=start+boundary['grid_arrival_occupancy_ns'];complete.append(start+boundary['grid_arrival_ns'])
  stage_end=max(complete)+boundary['grid_release_ns']
 return dict(prediction_available=False,conditional_numeric_available=True,adapter_validation='not_performed',pending=pending,conditional_body_end_ns=body_end,conditional_stage_end_ns=stage_end,
  ctas=rows,source_event_times_ns=mapped,source_event_upper_bounds_ns=upper_bounds,initial_state=initial,per_cta_packet_issue_next_available_ns={c:packets[c,1]['end_ns']for c in range(128)},per_sm_consumer_available_ns={states[c]['smid']:milestones[c]['returned']for c in range(128)},shared_transfer_final_inflight=[],shared_resource_available_ns=resource_final,last_transfer_response_ns=max(ready.values()),
  composite_ownership='paired connected Consume includes full waits/empty arrivals/MMA/finalize/epilogue/generation once; segment differences are response landmarks, not purecompute service fits',
  unresolved_internal_phase_times=['finalize','epilogue','generation-store instruction'],landmark_selection=landmark,
  response_stall_extension='inferred source-ordered max-plus response, not a proved prediction bound',source_packet_completion_gates=source_gates,
  observed_future_issue_gates_used=False,physical_HBM_bandwidth_claim=False,solver_eligible=False,E3_validated=False)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in('source','initial','paired','policy','out'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--boundary',type=Path);p.add_argument('--landmark',choices=['lower','upper'],default='upper');a=p.parse_args()
 require(not a.out.exists(),'retain prior prediction')
 inputs={k:json.loads(getattr(a,k).read_text())for k in('source','initial','paired','policy')}
 result=rollout(**inputs,boundary=json.loads(a.boundary.read_text())if a.boundary else None,landmark=a.landmark)
 result['input_hashes']={str(getattr(a,k)):sha(getattr(a,k))for k in inputs}
 if a.boundary:result['input_hashes'][str(a.boundary)]=sha(a.boundary)
 a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
