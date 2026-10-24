#!/usr/bin/env python3
"""Declared fluid processor-sharing inference; no population-specific constants."""
import argparse,csv,heapq,json,math,time
from collections import defaultdict,deque
from pathlib import Path
from statistics import median
from fit_sm90r2_flow_transfers import load,save,summary

def fluid_completions(arrivals,work_ns):
 """One shared work-conserving byte server, equal-weight active requests.

 Equal request work makes virtual completion order monotone. Delay after this
 service does not occupy bandwidth. work_ns=0 is an explicit no-byte baseline.
 """
 if not math.isfinite(work_ns)or work_ns<0 or any(not math.isfinite(t)for t in arrivals):raise ValueError('finite arrivals and nonnegative work required')
 if not work_ns:return list(map(float,arrivals))
 if any(b<a for a,b in zip(arrivals,arrivals[1:])):raise ValueError('ordered arrivals required')
 output=[0.]*len(arrivals);queue=deque();index=0;clock=0.;virtual=0.
 while index<len(arrivals)or queue:
  arrival=arrivals[index]if index<len(arrivals)else math.inf
  completion=clock+max(0,queue[0][0]-virtual)*len(queue)*work_ns if queue else math.inf
  if arrival<=completion:
   if queue:virtual+=(arrival-clock)/(len(queue)*work_ns)
   else:virtual=0.
   clock=arrival
   while index<len(arrivals)and arrivals[index]==clock:queue.append((virtual+1,index));index+=1
  else:
   clock=completion;virtual=queue[0][0]
   while queue and queue[0][0]<=virtual+1e-12:
    _,i=queue.popleft();output[i]=clock
 return output

def profile(trials,work_ns,retire):
 weight=target=target2=pweight=ptarget=ptarget2=0.
 for trial in trials:
  done=fluid_completions([r[2]for r in trial.rows],work_ns);clock={};maximum=0.
  for row,service_done in zip(trial.rows,done):
   c,s,start,packet,response,_=row
   observed_ready=max(service_done,clock.get(c,-math.inf)+retire);clock[c]=observed_ready
   y=response-(observed_ready-start);weight+=trial.wr;target+=trial.wr*y;target2+=trial.wr*y*y
   pweight+=trial.wp;ptarget+=trial.wp*packet;ptarget2+=trial.wp*packet*packet;maximum=max(maximum,observed_ready)
  y=trial.envelope-maximum;weight+=trial.we;target+=trial.we*y;target2+=trial.we*y*y
 lag=max(0,target/weight);packet=max(0,ptarget/pweight)
 loss=max(0,target2-2*lag*target+lag*lag*weight)+max(0,ptarget2-2*packet*ptarget+packet*packet*pweight)
 return loss/len(trials),dict(request_work_ns=work_ns,propagation_observation_lag_ns=lag,
     packet_response_ns=packet,packet_queue_fraction=0.,reader_retire_observer_ns=retire)

def fit(trials,activation):
 retire=median(r[5]for t in trials for r in t.rows)
 upper=min(r[4]for t in trials if t.active==1 for r in t.rows)
 grid=[upper*(i/24)**3 for i in range(25)];values=[profile(trials,d,retire)for d in grid]
 best=min(range(len(grid)),key=lambda i:values[i][0]);lo,hi=grid[max(0,best-1)],grid[min(24,best+1)]
 ratio=(math.sqrt(5)-1)/2;a,b=hi-ratio*(hi-lo),lo+ratio*(hi-lo);fa,fb=profile(trials,a,retire),profile(trials,b,retire)
 for _ in range(18):
  if fa[0]<fb[0]:hi,b,fb=b,a,fa;a=hi-ratio*(hi-lo);fa=profile(trials,a,retire)
  else:lo,a,fa=a,b,fb;b=lo+ratio*(hi-lo);fb=profile(trials,b,retire)
 loss,model=min([*values,fa,fb],key=lambda v:v[0]);work=32768+activation*16384
 model.update(request_work_bytes=work,effective_logical_bandwidth_B_per_ns=work/model['request_work_ns']if model['request_work_ns']else None,
     isolated_ready_response_ns=model['request_work_ns']+model['propagation_observation_lag_ns'],
     training_loss=loss,training_fresh=sorted({t.fresh for t in trials}),training_active_ctas=[1,32,64],
     resource_policy='one_fluid_equal_processor_sharing_server_declared_inference',packet_queue_correlation='fixed_zero_not_fitted',
     physical_HBM_bandwidth=False)
 return model,dict(work_grid_ns=grid,loss=[v[0]for v in values])

def conditional(trial,model):
 done=fluid_completions([r[2]for r in trial.rows],model['request_work_ns']);clock={};pred={}
 for row,service_done in zip(trial.rows,done):
  c,s,start,_,_,_=row;ready=max(service_done+model['propagation_observation_lag_ns'],clock.get(c,-math.inf))
  clock[c]=ready+model['reader_retire_observer_ns'];pred[c,s]=(start,start+model['packet_response_ns'],ready)
 return pred

def autonomous(entries,operations,model,consumer_progress_ns=None,consumer_return_ns=None):
 if not entries or operations<1 or model['request_work_ns']<=0:raise ValueError('positive finite work/scope required')
 if any(not math.isfinite(t)for t in entries.values()):raise ValueError('finite source entries required')
 for key in ('request_work_ns','propagation_observation_lag_ns','packet_response_ns','reader_retire_observer_ns'):
  if not math.isfinite(model[key])or model[key]<0:raise ValueError('finite nonnegative service required')
 progress=consumer_progress_ns if consumer_progress_ns is not None else[model['reader_retire_observer_ns']*(s+1)for s in range(operations)]
 if len(progress)!=operations or any(not math.isfinite(x)or x<0 for x in progress)or any(b<a for a,b in zip(progress,progress[1:])):raise ValueError('ordered progress required')
 returned=progress[-1]if consumer_return_ns is None else consumer_return_ns
 if not math.isfinite(returned)or returned<progress[-1]:raise ValueError('consumer return must follow all empty progress')
 heap=[];active=deque();virtual=0.;clock=min(entries.values());pred={};empty={};consumer_clock=dict(entries)
 next_sequence={c:0 for c in entries};packet_ended={(c,-1)for c in entries};emitted=set()
 for c,t in entries.items():heapq.heappush(heap,(t,0,c,0))
 def try_issue(c,now):
  s=next_sequence[c]
  if s>=operations or(c,s)in emitted or(c,s-1)not in packet_ended or(s>=2 and(c,s-2)not in empty):return
  emitted.add((c,s));next_sequence[c]+=1
  pred[c,s]=[now,now+model['packet_response_ns'],None];active.append((virtual+1,c,s))
  heapq.heappush(heap,(pred[c,s][1],1,c,s))
 while heap or active:
  external=heap[0][0]if heap else math.inf
  completion=clock+max(0,active[0][0]-virtual)*len(active)*model['request_work_ns']if active else math.inf
  now=min(external,completion)
  if active:virtual+=(now-clock)/(len(active)*model['request_work_ns'])
  clock=now
  # Complete old byte service before new arrivals at the same logical time.
  while active and active[0][0]<=virtual+1e-12:
   _,c,s=active.popleft();heapq.heappush(heap,(now+model['propagation_observation_lag_ns'],2,c,s))
  batch=[]
  while heap and heap[0][0]==now:batch.append(heapq.heappop(heap))
  for _,kind,c,s in batch:
   if kind==0:try_issue(c,now)
   elif kind==1:packet_ended.add((c,s));try_issue(c,now)
   elif kind==2:
    ready=max(now,consumer_clock[c]);pred[c,s][2]=ready
    step=progress[s]-(progress[s-1]if s else 0);consumer_clock[c]=ready+step
    heapq.heappush(heap,(consumer_clock[c],3,c,s))
   else:empty[c,s]=now;try_issue(c,now)
 returns={c:consumer_clock[c]+returned-progress[-1]for c in entries}
 return dict(packets=pred,empty=empty,consumer_returns=returns)

def metrics(trial,pred,mode,empty=None):
 pe=re=pm=rm=shift=0.;producer=ring=0
 for c,s,start,p,r,_ in trial.rows:
  t,packet,ready=pred[c,s];pe+=abs(packet-start-p);re+=abs(ready-start-r);pm+=p;rm+=r;shift+=t-start
  if s:producer+=pred[c,s-1][1]>t+1e-8
  if s>=2:ring+=(empty[c,s-2]if empty is not None else pred[c,s-2][2])>t+1e-8
 n=len(trial.rows);envelope=max(p[2]for p in pred.values())
 return dict(fresh=trial.fresh,cache=trial.cache,activation=trial.activation,active=trial.active,operations=trial.operations,trial=trial.trial,mode=mode,rows=n,
     packet_mae_ns=pe/n,packet_relative_error=pe/max(1,pm),ready_mae_ns=re/n,ready_relative_error=re/max(1,rm),
     observed_envelope_ns=trial.envelope,predicted_envelope_ns=envelope,envelope_relative_error=abs(envelope/trial.envelope-1),envelope_signed_error=envelope/trial.envelope-1,
     source_start_shift_ns=shift/n,producer_order_violations=producer,ring_ready_violations=ring)

def run(collected,out):
 if out.exists():raise FileExistsError('retain prior model version')
 out.mkdir(parents=True);start=time.monotonic();trials,bindings=load(collected);models={};profiles={};scores=[]
 for split,train_fresh,test_fresh in [('train012_hold34',[0,1,2],[3,4]),('all5_train',list(range(5)),list(range(5)))]:
  models[split]={};profiles[split]={}
  for cache,activation in sorted({(t.cache,t.activation)for t in trials}):
   selected=[t for t in trials if(t.cache,t.activation)==(cache,activation)]
   train=[t for t in selected if t.active in(1,32,64)and t.fresh in train_fresh]
   key=cache+('/A+B'if activation else'/B_only');model,profile_data=fit(train,activation);models[split][key]=model;profiles[split][key]=profile_data
   for t in selected:
    if t.fresh not in test_fresh:continue
    cp=conditional(t,model);ap=autonomous({c:start for c,s,start,_,_,_ in t.rows if s==0},t.operations,model)
    for mode,pred,empty in [('conditional',cp,None),('autonomous',ap['packets'],ap['empty'])]:
     row=metrics(t,pred,mode,empty);row.update(model=split,split='reused_geometry128_validation'if t.active==128 else'independent_process'if split!='all5_train'else'train_support');scores.append(row)
   print(json.dumps(dict(model=split,traffic=key,parameters=model)),flush=True)
  save(out/'models.json',dict(schema=1,models=models,physical_rate_status='inferred_fluid_service_on_literal_probe',geometry128_status='reused_architecture_validation_excluded_from_coefficients',
      packet_future_source_inputs_used=False,production_validated=False,E3_validated=False))
 with(out/'trial_predictions.csv').open('w')as stream:
  writer=csv.DictWriter(stream,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
 groups=defaultdict(list)
 for row in scores:groups[(row['model'],row['cache'],row['activation'],row['split'],row['mode'])].append(row)
 save(out/'validation.json',dict(groups={'|'.join(map(str,k)):summary(v)for k,v in sorted(groups.items())},elapsed_s=time.monotonic()-start,
      fitted_parameters_per_class=3,reader_progress='direct paired retire observer median',coefficient_training_active=[1,32,64],
      independent_processes=[3,4],geometry128_status='reused_validation_not_blind',production_validated=False,E3_validated=False))
 save(out/'objective_profiles.json',profiles);save(out/'input_hashes.json',bindings)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collected',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args();run(args.collected,args.out)
if __name__=='__main__':main()
