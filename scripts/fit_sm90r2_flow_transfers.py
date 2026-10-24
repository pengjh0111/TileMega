#!/usr/bin/env python3
"""Fit a shared admission/response model; issue starts are conditional inputs."""
import argparse,csv,hashlib,heapq,json,math,time
from collections import defaultdict
from pathlib import Path
from statistics import mean,median

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

class Trial:
 def __init__(self,key,rows):
  self.fresh,self.cache,self.activation,self.active,self.operations,self.trial=key
  origin=min(r[2]for r in rows)
  # Subtract integer global timestamps before every conversion/statistic.
  self.rows=[(c,s,t-origin,p-t,r-t,e-r)for c,s,t,p,r,e in rows]
  self.rows.sort(key=lambda r:(r[2],r[0],r[1]))
  self.envelope=max(r[2]+r[4]for r in self.rows)
  n=len(rows);self.wr=1/n/max(1,median(r[4]for r in self.rows))**2
  self.wp=1/n/max(1,median(r[3]for r in self.rows))**2
  self.we=1/self.envelope**2
  self.key=key

def load(collected):
 receipt=json.loads((collected/'receipt.json').read_text());groups=defaultdict(list);bindings={}
 assert receipt['numerical_passes']==5 and len(receipt['processes'])==5
 assert len({(p['boot_id'],p['pid'])for p in receipt['processes']})==5
 for process in receipt['processes']:
  path=Path(process['attempt'])/'unit/probe/raw.csv'
  assert sha(path)==process['receipts']['unit/probe/raw.csv'];bindings[str(path)]=sha(path)
  with path.open()as stream:
   for row in csv.DictReader(stream):
    fresh=int(row['fresh']);assert fresh==process['fresh']
    key=(fresh,row['cache_preparation'],int(row['activation']),int(row['active_ctas']),int(row['operations']),int(row['trial']))
    c,s=int(row['cta']),int(row['sequence'])
    wait,t,p,first,r,e=(int(row[k])for k in('wait_begin','issue_begin','packet_end','first_warp_ready','all_warps_ready','all_warps_retired'))
    assert wait<=t<=p and t<=first<=r<=e
    groups[key].append((c,s,t,p,r,e))
 trials=[]
 for key,rows in sorted(groups.items()):
  active,operations=key[3:5]
  assert len(rows)==active*operations and {(r[0],r[1])for r in rows}=={(c,s)for c in range(active)for s in range(operations)}
  trials.append(Trial(key,rows))
 assert len(trials)==5*4*4*3*16
 return trials,bindings

def profile(trials,occupancy):
 """Constrained weighted least squares for fixed shared admission occupancy."""
 rw=ry=ry2=pw=px=px2=py=py2=pxy=0.0
 for trial in trials:
  free=0.0;maximum=0.0;wr,wp=trial.wr,trial.wp
  for _,_,arrival,packet,response,_ in trial.rows:
   admitted=max(arrival,free);free=admitted+occupancy;wait=admitted-arrival
   target=response-wait;rw+=wr;ry+=wr*target;ry2+=wr*target*target
   pw+=wp;px+=wp*wait;px2+=wp*wait*wait;py+=wp*packet;py2+=wp*packet*packet;pxy+=wp*wait*packet
   maximum=max(maximum,admitted)
  target=trial.envelope-maximum;rw+=trial.we;ry+=trial.we*target;ry2+=trial.we*target*target
 lag=max(0,ry/rw);ready_loss=max(0,ry2-2*lag*ry+lag*lag*rw)
 denominator=px2-px*px/pw
 slope=(pxy-px*py/pw)/denominator if denominator>1e-15 else 0
 candidates=[0.,1.,max(0,min(1,slope))]
 if px2:candidates.append(max(0,min(1,pxy/px2)))
 best=None
 for fraction in candidates:
  base=max(0,(py-fraction*px)/pw)
  loss=max(0,py2+base*base*pw+fraction*fraction*px2+2*base*fraction*px-2*base*py-2*fraction*pxy)
  if best is None or loss<best[0]:best=(loss,base,fraction)
 loss,base,fraction=best
 return (ready_loss+loss)/len(trials),dict(admission_occupancy_ns=occupancy,response_lag_ns=lag,
     packet_base_ns=base,packet_queue_fraction=fraction)

def fit(trials):
 upper=max(r[4]for t in trials for r in t.rows)
 grid=[upper*(i/32)**3 for i in range(33)]
 samples=[profile(trials,x)for x in grid];best=min(range(len(grid)),key=lambda i:samples[i][0])
 lo,hi=grid[max(0,best-1)],grid[min(32,best+1)];ratio=(math.sqrt(5)-1)/2
 a,b=hi-ratio*(hi-lo),lo+ratio*(hi-lo);fa,fb=profile(trials,a),profile(trials,b)
 for _ in range(24):
  if fa[0]<fb[0]:hi,b,fb=b,a,fa;a=hi-ratio*(hi-lo);fa=profile(trials,a)
  else:lo,a,fa=a,b,fb;b=lo+ratio*(hi-lo);fb=profile(trials,b)
 selected=min([*samples,fa,fb],key=lambda v:v[0]);loss,model=selected
 model['reader_retire_observer_ns']=median(r[5]for t in trials for r in t.rows)
 model['training_loss']=loss;model['training_processes']=sorted({t.fresh for t in trials})
 model['training_active_ctas']=[1,32,64]
 model['admission_model']='one_shared_effective_FIFO_server_not_hardware_admission'
 return model,dict(occupancy_grid_ns=grid,training_objective=[v[0]for v in samples],zero_queue_training_loss=samples[0][0],zero_queue_model=samples[0][1])

def predict(trial,model,finite=False,reverse=False):
 """Return source/packet/ready times; finite replay uses observed starts as floors."""
 occupancy=model['admission_occupancy_ns'];lag=model['response_lag_ns'];base=model['packet_base_ns'];fraction=model['packet_queue_fraction']
 rows=trial.rows;free=0.;pred={}
 if not finite:
  ordered=sorted(rows,key=lambda r:(r[2],-r[0]if reverse else r[0],r[1]))
  for c,s,arrival,_,_,_ in ordered:
   admitted=max(arrival,free);free=admitted+occupancy;wait=admitted-arrival
   pred[c,s]=(float(arrival),arrival+base+fraction*wait,admitted+lag)
 else:
  per_cta=defaultdict(dict)
  for row in rows:per_cta[row[0]][row[1]]=row
  queue=[]
  for c,series in per_cta.items():heapq.heappush(queue,(series[0][2],-c if reverse else c,c,0))
  while queue:
   arrival,_,c,s=heapq.heappop(queue);admitted=max(arrival,free);free=admitted+occupancy
   packet=arrival+base+fraction*(admitted-arrival);ready=admitted+lag;pred[c,s]=(arrival,packet,ready)
   if s+1<trial.operations:
    next_start=max(per_cta[c][s+1][2],packet)
    if s+1>=2:next_start=max(next_start,pred[c,s-1][2]+model['reader_retire_observer_ns'])
    heapq.heappush(queue,(next_start,-c if reverse else c,c,s+1))
 return pred

def score(trial,model,mode):
 pred=predict(trial,model,mode!='conditional',mode=='finite_reverse')
 pe=re=pm=rm=shift=0.;producer=ring=0
 for c,s,t,p,r,_ in trial.rows:
  start,packet,ready=pred[c,s];pe+=abs((packet-t)-p);re+=abs((ready-t)-r);pm+=p;rm+=r;shift+=start-t
  if s:producer+=pred[c,s-1][1]>start+1e-9
  if s>=2:ring+=pred[c,s-2][2]>start+1e-9
 n=len(trial.rows);envelope=max(v[2]for v in pred.values())
 return dict(fresh=trial.fresh,cache=trial.cache,activation=trial.activation,active=trial.active,operations=trial.operations,trial=trial.trial,mode=mode,
     rows=n,packet_mae_ns=pe/n,packet_relative_error=pe/max(1,pm),ready_mae_ns=re/n,ready_relative_error=re/max(1,rm),
     observed_envelope_ns=trial.envelope,predicted_envelope_ns=envelope,envelope_relative_error=abs(envelope/trial.envelope-1),
     envelope_signed_error=envelope/trial.envelope-1,source_start_shift_ns=shift/n,
     producer_order_violations=producer,ring_ready_violations=ring)

def summary(rows):
 def stats(key):
  values=sorted(r[key]for r in rows)
  return dict(mean=mean(values),median=median(values),p90=values[int(.9*(len(values)-1))],max=max(values))
 return dict(trials=len(rows),rows=sum(r['rows']for r in rows),packet_relative_error=stats('packet_relative_error'),
     ready_relative_error=stats('ready_relative_error'),envelope_relative_error=stats('envelope_relative_error'),
     envelope_signed_error=stats('envelope_signed_error'),packet_mae_ns=stats('packet_mae_ns'),ready_mae_ns=stats('ready_mae_ns'),
     source_start_shift_ns=stats('source_start_shift_ns'),producer_order_violations=sum(r['producer_order_violations']for r in rows),
     ring_ready_violations=sum(r['ring_ready_violations']for r in rows))

def run(collected,out):
 if out.exists():raise FileExistsError('preserve prior fitting results')
 out.mkdir(parents=True);start=time.monotonic();trials,bindings=load(collected)
 classes=sorted({(t.cache,t.activation)for t in trials});models={};profiles={};scores=[]
 # The same fixed model family is used for all folds; active128 never trains.
 for held in [0,1,2,3,4,None]:
  fold='all5_train'if held is None else'leave_fresh_'+str(held)
  models[fold]={};profiles[fold]={}
  for cache,activation in classes:
   selected=[t for t in trials if(t.cache,t.activation)==(cache,activation)]
   train=[t for t in selected if t.active in(1,32,64)and t.fresh!=held]
   key=cache+('/A+B'if activation else'/B_only');model,profile_data=fit(train)
   models[fold][key]=model;profiles[fold][key]=profile_data
   test=selected if held is None else[t for t in selected if t.fresh==held]
   for trial in test:
    for mode in('conditional','finite_s2','finite_reverse'):
     row=score(trial,model,mode);row.update(fold=fold,split='geometry128_holdout'if trial.active==128 else'process_holdout'if held is not None else'train')
     scores.append(row)
    row=score(trial,profile_data['zero_queue_model'],'conditional');row.update(fold=fold,split='geometry128_holdout'if trial.active==128 else'process_holdout'if held is not None else'train',mode='conditional_no_queue');scores.append(row)
   print(json.dumps(dict(fold=fold,traffic=key,model=model)),flush=True)
  save(out/'models.json',dict(schema=1,models=models,physical_rate_status='inferred_fit_to_literal_transfer_probe_not_production_body',
      issue_starts='actual_observed_external_inputs_or_floor_in_finite_replay',first_byte_observed=False,hardware_admission_observed=False,
      cache_hit_ratio_measured=False,production_validated=False,E3_validated=False))
 with(out/'trial_predictions.csv').open('w')as stream:
  writer=csv.DictWriter(stream,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
 groups=defaultdict(list)
 for row in scores:
  fold='final'if row['fold']=='all5_train'else'LOPO'
  groups[(fold,row['cache'],row['activation'],row['split'],row['mode'])].append(row)
 report={'|'.join(map(str,key)):summary(rows)for key,rows in sorted(groups.items())}
 save(out/'validation.json',dict(schema=1,groups=report,elapsed_s=time.monotonic()-start,
      fit_objective='equal-trial scaled squared paired packet+ready errors plus population envelope error',
      parameter_count_per_cache_traffic_class=4,reader_retire_observer='direct paired median, not exact hardware empty time',
      finite_replay='observed source starts as lower bounds plus modeled producer and S2 feedback; not autonomous source prediction',
      source_arrival_orchestration_predicted=False,production_validated=False,E3_validated=False))
 save(out/'objective_profiles.json',profiles);save(out/'input_hashes.json',bindings)
 return models,report

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collected',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args();run(args.collected,args.out)
if __name__=='__main__':main()
