#!/usr/bin/env python3
"""Autonomous S2 packet/consumer loop: no observed future issue gates."""
import argparse,csv,heapq,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from fit_sm90r2_flow_transfers import load,save,summary

def autonomous(entries,operations,model,consumer_progress_ns=None,consumer_return_ns=None,reverse=False):
 """Entries are supplied stage-entry/source-start times, one per CTA.

 Consumer progress is a paired no-data-stall empty-observer vector. Its entire
 response is owned once; do not append separately priced finalize/epilogue.
 The default is this probe's learned minimal-reader response, not production MMA.
 """
 if operations<1 or not entries:raise ValueError('nonempty finite scope required')
 if any(not math.isfinite(t)for t in entries.values()):raise ValueError('finite source entries required')
 for key in ('admission_occupancy_ns','response_lag_ns','packet_base_ns','reader_retire_observer_ns'):
  if not math.isfinite(model[key])or model[key]<0:raise ValueError('finite nonnegative service required')
 if not math.isfinite(model['packet_queue_fraction'])or not 0<=model['packet_queue_fraction']<=1:
  raise ValueError('packet queue fraction must be in [0,1]')
 progress=consumer_progress_ns if consumer_progress_ns is not None else[model['reader_retire_observer_ns']*(s+1)for s in range(operations)]
 if len(progress)!=operations or any(not math.isfinite(x)or x<0 for x in progress)or any(b<a for a,b in zip(progress,progress[1:])):
  raise ValueError('nonnegative ordered progress required')
 return_ns=progress[-1]if consumer_return_ns is None else consumer_return_ns
 if not math.isfinite(return_ns)or return_ns<progress[-1]:raise ValueError('consumer return must follow all empty progress')
 queue=[(t,-c if reverse else c,c,0)for c,t in entries.items()];heapq.heapify(queue)
 free=min(entries.values());pred={};empty={};consumer_clock=dict(entries)
 while queue:
  start,_,c,s=heapq.heappop(queue);admitted=max(start,free);free=admitted+model['admission_occupancy_ns']
  packet=start+model['packet_base_ns']+model['packet_queue_fraction']*(admitted-start)
  ready=max(admitted+model['response_lag_ns'],consumer_clock[c]);pred[c,s]=(start,packet,ready)
  step=progress[s]-(progress[s-1]if s else 0)
  consumer_clock[c]=ready+step;empty[c,s]=consumer_clock[c]
  if s+1<operations:
   next_start=packet
   if s+1>=2:next_start=max(next_start,empty[c,s-1])
   heapq.heappush(queue,(next_start,-c if reverse else c,c,s+1))
 returns={c:consumer_clock[c]+return_ns-progress[-1]for c in entries}
 if any(returns[c]<pred[c,operations-1][2]for c in entries):raise ValueError('progress/return response violates final readiness')
 return dict(packets=pred,empty=empty,consumer_returns=returns,hierarchy_next_available_ns=free)

def score(trial,model,reverse=False):
 entries={c:t for c,s,t,_,_,_ in trial.rows if not s}
 replay=autonomous(entries,trial.operations,model,reverse=reverse);pred=replay['packets']
 pe=re=pm=rm=shift=absolute_shift=0.;producer=ring=0
 for c,s,t,p,r,_ in trial.rows:
  start,packet,ready=pred[c,s];pe+=abs(packet-t-p);re+=abs(ready-t-r);pm+=p;rm+=r;shift+=start-t;absolute_shift+=abs(start-t)
  if s:producer+=pred[c,s-1][1]>start+1e-9
  if s>=2:ring+=replay['empty'][c,s-2]>start+1e-9
 n=len(trial.rows);envelope=max(p[2]for p in pred.values())-min(entries.values())
 return dict(fresh=trial.fresh,cache=trial.cache,activation=trial.activation,active=trial.active,operations=trial.operations,trial=trial.trial,
     mode='autonomous_reverse'if reverse else'autonomous',rows=n,packet_mae_ns=pe/n,packet_relative_error=pe/max(1,pm),
     ready_mae_ns=re/n,ready_relative_error=re/max(1,rm),observed_envelope_ns=trial.envelope,predicted_envelope_ns=envelope,
     envelope_relative_error=abs(envelope/trial.envelope-1),envelope_signed_error=envelope/trial.envelope-1,
     source_start_shift_ns=shift/n,source_start_mae_ns=absolute_shift/n,producer_order_violations=producer,ring_ready_violations=ring)

def validate(collected,models_path,out):
 if out.exists():raise FileExistsError('preserve prior autonomous evaluation')
 out.mkdir(parents=True);trials,_=load(collected);models=json.loads(models_path.read_text())['models'];rows=[]
 for fold,sets in models.items():
  held=None if fold=='all5_train'else int(fold.rsplit('_',1)[-1])
  for t in trials:
   if held is not None and t.fresh!=held:continue
   key=t.cache+('/A+B'if t.activation else'/B_only')
   for reverse in(False,True):
    row=score(t,sets[key],reverse);row.update(fold=fold,split='reused_geometry128_validation'if t.active==128 else'independent_process'if held is not None else'train_support')
    rows.append(row)
 with(out/'trial_predictions.csv').open('w')as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 groups=defaultdict(list)
 for row in rows:groups[('final'if row['fold']=='all5_train'else'LOPO',row['cache'],row['activation'],row['split'],row['mode'])].append(row)
 result={ '|'.join(map(str,key)):summary(values)for key,values in sorted(groups.items())}
 save(out/'validation.json',dict(groups=result,fitting_performed=False,observed_future_issue_starts_used=False,
     input='only supplied per-CTA entry plus calibrated policies; validation entries are first observed issue',
     geometry128_status='reused validation after first architecture failure; excluded from every coefficient fit',
     production_validated=False,E3_validated=False))
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--validate-collected',type=Path);parser.add_argument('--entries',type=Path);parser.add_argument('--operations',type=int,default=2)
 parser.add_argument('--traffic',default='warm_touch/A+B');parser.add_argument('--consumer-progress',type=Path)
 args=parser.parse_args()
 if args.validate_collected:validate(args.validate_collected,args.models,args.out);return
 if not args.entries:parser.error('--entries or --validate-collected required')
 entries={int(c):float(t)for c,t in json.loads(args.entries.read_text()).items()}
 policy=json.loads(args.models.read_text())['models']['all5_train'][args.traffic]
 consume=json.loads(args.consumer_progress.read_text())if args.consumer_progress else{}
 replay=autonomous(entries,args.operations,policy,consume.get('empty_progress_ns'),consume.get('return_ns'))
 output={k:({str(x):v for x,v in values.items()}if isinstance(values,dict)else values)for k,values in replay.items()}
 save(args.out,output)
if __name__=='__main__':main()
