#!/usr/bin/env python3
"""Collect literal joint production-role observations, without phase attribution."""
import argparse,csv,hashlib,json,shutil,subprocess,sys
from collections import defaultdict
from pathlib import Path
from statistics import median
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from collect_sm90r2_flow_consumers import sha,read,write,require,summarize,csv_write,validate_process
ACTIVE=(1,32,64,128);MODES=('warm_touch','scrub_surrogate')
PROBE_SHA='477857c677f3e01ea5b5fcce5a6760295354193ea2401b8d6ba23a381cf96905'
SOURCE_SHA='4d1ee58e889df1d7cdf692515767b898d6636584f8af39f13da1f041a127126f'
KEY=('active_ctas','cache_mode','trial');CTA_KEY=KEY+('cta',)
RAW=('fresh','active_ctas','cache_mode','trial','cta','smid','warp','consumer_entry','consumer_returned','producer_entry','producer_returned','setup_acquired','roles_join_observed','v3_arrive','v3_acquired')
CTAS=('fresh','active_ctas','cache_mode','trial','cta','smid','has_work','setup_acquired','producer_entry','producer_returned','roles_join_observed','producer_phase0','producer_phase1','consumer_phase0','consumer_phase1','v3_arrive','v3_acquired')
CONTEXTS=('fresh','active_ctas','cache_mode','trial','setup_arrivals','final_arrivals','cache_checksum','checked_outputs','checked_sentinels')
COMMON=('fresh',)+CTA_KEY+('smid','producer_entry','producer_returned','setup_acquired','roles_join_observed','v3_arrive','v3_acquired')


def launches():return [(a,MODES[i^(t&1)],t) for a in ACTIVE for t in range(1,17) for i in range(2)]

def parse(row,fields,fresh):
 require(set(row)==set(fields) and row['cache_mode'] in MODES,'wrong columns/cache mode')
 require(all(isinstance(v,str) and v.isascii() and v.isdecimal() for k,v in row.items() if k!='cache_mode'),'non-unsigned CSV integer')
 r={k:v if k=='cache_mode' else int(v) for k,v in row.items()}
 require(r['fresh']==fresh and r['active_ctas'] in ACTIVE and 1<=r['trial']<=16 and all(0<=v<2**64 for k,v in r.items() if k!='cache_mode'),'invalid fresh/coordinates')
 if fields==CONTEXTS:
  require(r['setup_arrivals']==r['final_arrivals']==132,'native V3 counters are not132')
  require(r['checked_outputs']==16*r['active_ctas']*32 and r['checked_sentinels']==16*(4096-r['active_ctas']*32)+128,'wrong output/sentinel coverage')
 else:
  require(0<=r['cta']<132 and 0<r['setup_acquired']<=r['roles_join_observed']<=r['v3_arrive']<=r['v3_acquired'],'invalid CTA/join/V3 timestamps')
  work=r['cta']<r['active_ctas']
  if work:require(r['setup_acquired']<=r['producer_entry']<r['producer_returned']<=r['roles_join_observed'],'invalid paired producer interval')
  else:require(r['producer_entry']==r['producer_returned']==0,'idle CTA has producer interval')
  if fields==CTAS:
   require(r['has_work']==int(work) and all(r[k]==int(work) for k in ('producer_phase0','producer_phase1','consumer_phase0','consumer_phase1')),'wrong work/generation values')
  else:require(work and 0<=r['warp']<4 and r['setup_acquired']<=r['consumer_entry']<r['consumer_returned']<=r['roles_join_observed'],'invalid paired consumer interval')
 return r

def load(path,fields,fresh):
 with Path(path).open() as stream:
  rows=csv.DictReader(stream);require(rows.fieldnames==list(fields),'wrong header: '+str(path));return [parse(r,fields,fresh) for r in rows]

def validate_rows(raw,ctas,contexts):
 order=launches()
 require([tuple(r[k] for k in KEY) for r in contexts]==order,'missing/duplicate/reordered interleaved contexts')
 require([tuple(r[k] for k in CTA_KEY) for r in ctas]==[key+(c,) for key in order for c in range(132)],'incomplete/duplicate/reordered CTA cohort')
 require([tuple(r[k] for k in CTA_KEY+('warp',)) for r in raw]==[key+(c,w) for key in order for c in range(key[0]) for w in range(4)],'incomplete/duplicate/reordered raw cohort')
 mapping={tuple(r[k] for k in CTA_KEY):r for r in ctas};prior_end=0
 for key in order:
  group=[mapping[key+(c,)] for c in range(132)]
  require(len({r['smid'] for r in group})==132,'resident CTAs lack132 distinct actual SMIDs')
  require(max(r['v3_arrive'] for r in group)<=min(r['v3_acquired'] for r in group),'native acquire predates last CTA arrival')
  require(prior_end<=min(r['setup_acquired'] for r in group),'separate interleaved launches overlap or reverse global time')
  prior_end=max(r['v3_acquired'] for r in group)
 groups=defaultdict(dict)
 for r in raw:
  key=tuple(r[k] for k in CTA_KEY)
  require(all(r[k]==mapping[key][k] for k in COMMON),'raw/CTA metadata or producer stamps differ')
  groups[key][r['warp']]=r
 observations={}
 for key,warps in groups.items():
  require(set(warps)==set(range(4)),'missing consumer warp');r=dict(mapping[key])
  cbegin=min(v['consumer_entry'] for v in warps.values());cend=max(v['consumer_returned'] for v in warps.values())
  begin=min(cbegin,r['producer_entry']);end=max(cend,r['producer_returned'])
  # Every subtraction is between Python integers before any statistic/ratio.
  r.update(consumer_entry_min=cbegin,consumer_return_max=cend,joint_entry_min=begin,last_role_return=end,
   producer_call_ns=r['producer_returned']-r['producer_entry'],consumer_cta_envelope_ns=cend-cbegin,
   joint_entry_to_last_role_return_ns=end-begin,joint_entry_to_join_observed_ns=r['roles_join_observed']-begin,
   join_observation_tail_ns=r['roles_join_observed']-end,
   producer_entry_relative_to_first_consumer_ns=r['producer_entry']-cbegin,
   after_join_to_v3_arrive_ns=r['v3_arrive']-r['roles_join_observed'],
   native_v3_arrive_to_acquire_ns=r['v3_acquired']-r['v3_arrive'],
   consumer_warp_durations=[warps[w]['consumer_returned']-warps[w]['consumer_entry'] for w in range(4)])
  observations[key]=r
 return observations,mapping


def metadata(meta):
 require(tuple(meta.get(k) for k in ('schema','grid','contexts','threads','shared_bytes','record_offset','slots','iterations','tasks_per_active_cta','registers','local_bytes','observed_residency','measured_trials','scrub_bytes'))==(1,132,8,160,199808,448,2,2,1,142,0,1,16,134217728),'wrong joint geometry/resources')
 require(meta.get('active_ctas')==list(ACTIVE) and meta.get('cache_modes')==list(MODES) and meta.get('epsilon')==0 and meta.get('input_norm') is True and meta.get('output_sums') is False and meta.get('prefix_bytes')==0,'wrong joint flags/contexts')
 require(meta.get('real_transfers') is True and meta.get('dummy_producer') is False and meta.get('reference')=='independent_exact_integer_dot_power_of_two_norm_BF16' and meta.get('observation')=='joint_actual_Produce_Consume_role_entry_return_under_native_V3' and all(meta.get(k) is False for k in ('verified_cold_cache','first_byte_observed','hardware_admission_observed','pure_compute_observed','E2E_offset_fit','performance_eligible')),'wrong joint interpretation')
 require(meta.get('roles_join_observed')=='thread_zero_timestamp_after_all160_threads_join' and meta.get('v3_acquired')=='thread_zero_after_global_acquire_before_final_CTA_join' and meta.get('output_and_sentinel_checks')=='every_launch_including_warmup' and meta.get('counter_values')=='serialized_per_context','wrong join/counter/numerical scope')


def binding_checks(binding):
 for path,digest in binding['files'].items():require(sha(path)==digest,'frozen probe input changed: '+path)
 h=Path(binding['source_file']).parent
 paths=dict(source=Path(binding['source_file']),binary=h/'probe',runner=h/'run.py',review=Path(binding['function_review']))
 require(sha(paths['binary'])==PROBE_SHA and sha(paths['source'])==SOURCE_SHA,'wrong literal jointly running probe')
 review=read(paths['review']);audit=read(binding['audit_requirements']);request=read(binding['build_request']);build=read(binding['build_receipt']);snapshot=read(binding['source_snapshot'])
 require(review['status']=='passed' and review['probe_binary_sha256']==PROBE_SHA and review['reference_binary_sha256']==audit['reference_binary_sha256']=='9376bc4772812277f521631c93b9baafa6d8bd19343161a2e1ed0ffe84aef38a','wrong actual function audit/reference')
 expected={r['role']:r['symbol'] for r in audit['functions']}
 require(len(review['comparisons'])==2 and {r['role'] for r in review['comparisons']}==set(expected)=={'Produce','Consume'},'missing unique joint function audit')
 for r in review['comparisons']:
  require(r['key']==expected[r['role']] and r['left_names']==r['right_names']==[expected[r['role']]] and all(r[k] is True for k in ('unique','bytecode_equal','resource_metadata_equal')),'joint full-symbol/function bytes/resources differ')
  require(r['left_flags']==r['right_flags']==([['true','false']] if r['role']=='Consume' else []),'wrong actual function flags')
 require(build['status']=='completed' and build['exit_code']==0 and build['command']==request['command'] and snapshot['source_head']==binding['source_head'] and snapshot['source_sha256']==SOURCE_SHA and snapshot['request_sha256']==sha(binding['build_request']),'build/source snapshot differs')
 headers=snapshot['header_hashes'];names=list(headers);head=snapshot['source_head'];data=subprocess.check_output(['git','-C',str(REPO),'cat-file','--batch'],input=''.join(head+':'+n+'\n' for n in names).encode());cursor=0
 for name in names:
  end=data.index(b'\n',cursor);info=data[cursor:end].split();cursor=end+1;require(len(info)==3 and info[1]==b'blob','missing historical header')
  size=int(info[2]);blob=data[cursor:cursor+size];cursor+=size+1;require(hashlib.sha256(blob).hexdigest()==headers[name],'historical header mismatch')
 require(all(n in headers for n in binding['transitive_tilemega_headers']),'missing transitive source header')
 require(binding['input_manifest_sha256']==sha(h/'inputs/manifest.json'),'prepared inputs differ')
 return paths,review


def metrics(observations,ctas,fresh):
 samples=defaultdict(list);trial_rows=[];pairs=[];placements=[]
 def add(a,m,scope,key,value):samples[(a,m,scope,key)].append(value)
 measured=('producer_call_ns','consumer_cta_envelope_ns','joint_entry_to_last_role_return_ns','joint_entry_to_join_observed_ns','join_observation_tail_ns','producer_entry_relative_to_first_consumer_ns','after_join_to_v3_arrive_ns','native_v3_arrive_to_acquire_ns')
 for (a,m,t,c),r in observations.items():
  for name in measured:add(a,m,'cta',name,r[name])
  for value in r['consumer_warp_durations']:add(a,m,'warp','paired_consumer_entry_to_return_ns',value)
 for (a,m,t,c),r in ctas.items():
  role='active' if r['has_work'] else 'idle';span=r['v3_acquired']-r['v3_arrive'];add(a,m,role+'_cta','native_v3_arrive_to_acquire_ns',span)
  obs=observations.get((a,m,t,c));placements.append(dict(r,native_v3_arrive_to_acquire_ns=span,joint_entry_to_join_observed_ns=obs['joint_entry_to_join_observed_ns'] if obs else ''))
 for a in ACTIVE:
  for t in range(1,17):
   values={}
   for m in MODES:
    pop=[observations[a,m,t,c] for c in range(a)];allctas=[ctas[a,m,t,c] for c in range(132)]
    begin=min(r['joint_entry_min'] for r in pop);last_return=max(r['last_role_return'] for r in pop);last_join=max(r['roles_join_observed'] for r in pop);last_acquire=max(r['v3_acquired'] for r in allctas)
    row=dict(fresh=fresh,active_ctas=a,cache_mode=m,trial=t,earliest_role_entry=begin,last_role_return=last_return,last_active_join_observed=last_join,last_native_acquire=last_acquire,
     population_joint_entry_to_last_role_return_ns=last_return-begin,population_joint_entry_to_last_active_join_ns=last_join-begin,population_joint_entry_to_last_native_acquire_ns=last_acquire-begin,
     population_consumer_entry_to_return_ns=max(r['consumer_return_max'] for r in pop)-min(r['consumer_entry_min'] for r in pop))
    for name,value in row.items():
     if name.startswith('population_'):add(a,m,'trial',name,value)
    trial_rows.append(row);values[m]=row['population_joint_entry_to_last_active_join_ns']
   add(a,'paired_cache_control','trial','scrub_over_warm_population_join_span_ratio',values['scrub_surrogate']/values['warm_touch'])
   for c in range(a):
    warm=observations[a,'warm_touch',t,c];scrub=observations[a,'scrub_surrogate',t,c]
    ratio=scrub['joint_entry_to_join_observed_ns']/warm['joint_entry_to_join_observed_ns']
    pairs.append(dict(fresh=fresh,active_ctas=a,trial=t,cta=c,warm_smid=warm['smid'],scrub_smid=scrub['smid'],same_smid=warm['smid']==scrub['smid'],warm_join_span_ns=warm['joint_entry_to_join_observed_ns'],scrub_join_span_ns=scrub['joint_entry_to_join_observed_ns'],scrub_over_warm_ratio=ratio))
    add(a,'paired_cache_control','cta','scrub_over_warm_join_span_ratio',ratio)
 stats=[dict(fresh=fresh,active_ctas=k[0],cache_mode=k[1],scope=k[2],metric=k[3],**summarize(v)) for k,v in sorted(samples.items())]
 return stats,trial_rows,pairs,placements


def aggregate(rows,freshes):
 grouped=defaultdict(list)
 for row in rows:grouped[tuple(row[k] for k in ('active_ctas','cache_mode','scope','metric'))].append(row)
 result=[]
 for k,values in sorted(grouped.items()):
  require(len(values)==len(freshes) and {r['fresh'] for r in values}==set(freshes),'missing/mixed process summary')
  result.append(dict(active_ctas=k[0],cache_mode=k[1],scope=k[2],metric=k[3],processes=len(values),median_of_process_medians=median(r['median'] for r in values),median_of_process_p99=median(r['p99'] for r in values),min_process_median=min(r['median'] for r in values),max_process_median=max(r['median'] for r in values),aggregation='process_statistics_not_pooled'))
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('state','bindings','queue','out'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--allow-partial',action='store_true');a=p.parse_args();require(not a.out.exists(),'preserve collection')
 collector_files={str(Path(__file__).resolve()):sha(__file__),str(REPO/'scripts/collect_sm90r2_flow_consumers.py'):sha(REPO/'scripts/collect_sm90r2_flow_consumers.py')}
 inputs={k:getattr(a,k).read_bytes() for k in ('state','bindings','queue')};state,binding,queue=[json.loads(inputs[k]) for k in ('state','bindings','queue')]
 names=[f'flow-joint-qkv-v1-f{i}' for i in range(5)];jobs={j['name']:j for j in queue}
 require(len(queue)==5 and set(jobs)==set(state)==set(names),'wrong finite joint cohort')
 require(all(state[n]['status'] in ('pending','running','done') for n in names),'failed/rejected scheduler state')
 selected={n:state[n] for n in names if state[n]['status']=='done'};require(selected and (a.allow_partial or len(selected)==5),'five completed processes required; --allow-partial is provisional')
 paths,review=binding_checks(binding);evidence=[];stats=[];trials=[];pairs=[];placement=[]
 for fresh,name in enumerate(names):
  if name not in selected:continue
  folder=Path(selected[name]['out']);receipt=validate_process(folder,fresh,selected[name],jobs[name],binding,paths,metadata)
  unit=folder/'unit/probe';process=read(folder/'unit/probe-process.json')
  require(process['input_manifest_sha256']==binding['input_manifest_sha256'] and process['function_review_sha256']==sha(paths['review']),'runner input/audit binding differs')
  require(sha(unit/'input-manifest.json')==binding['input_manifest_sha256'],'executed input manifest differs')
  receipt['receipts'].update({str(unit/n):sha(unit/n) for n in ('contexts.csv','input-manifest.json')})
  observations,ctas=validate_rows(load(unit/'raw.csv',RAW,fresh),load(unit/'ctas.csv',CTAS,fresh),load(unit/'contexts.csv',CONTEXTS,fresh))
  rows=metrics(observations,ctas,fresh)
  for destination,value in zip((stats,trials,pairs,placement),rows):destination.extend(value)
  evidence.append(receipt)
 freshes=[e['fresh'] for e in evidence]
 require(len({(e['boot_id'],e['pid']) for e in evidence})==len(evidence) and len({e['ppid'] for e in evidence})==len(evidence),'not fresh child/wrapper processes')
 require(len({tuple(e['physical_identity'][k] for k in ('gpu_index','cuda_uuid','cuda_pci_bus_id','numa_node')) for e in evidence})==1,'mixed physical cohort')
 require({n:read(a.state)[n] for n in selected}==selected,'completed scheduler rows changed')
 for path,digest in binding['files'].items():require(sha(path)==digest,'bound input changed during collection')
 for path,digest in collector_files.items():require(sha(path)==digest,'collector changed while running')
 for e in evidence:
  for path,digest in e['receipts'].items():require(sha(path)==digest,'process receipt changed')
 a.out.mkdir(parents=True);(a.out/'inputs').mkdir();(a.out/'raw').mkdir()
 for name,data in inputs.items():(a.out/'inputs'/(name+'.json')).write_bytes(data)
 for e in evidence:
  for name in ('raw.csv','ctas.csv','contexts.csv'):shutil.copyfile(Path(e['attempt'])/'unit/probe'/name,a.out/'raw'/f"f{e['fresh']}-{name}")
 for name,rows in [('process-statistics.csv',stats),('metrics.csv',aggregate(stats,freshes)),('trial-observations.csv',trials),('paired-cache-controls.csv',pairs),('sm-placement.csv',placement)]:csv_write(a.out/name,rows)
 write(a.out/'receipt.json',dict(schema=1,complete_cohort=len(evidence)==5,numerical_passes=len(evidence),numerical_processes_expected=5,freshes=freshes,pending=[n for n in names if n not in selected],collector_files=collector_files,selected_scheduler_rows=selected,bindings_sha256=hashlib.sha256(inputs['bindings']).hexdigest(),queue_sha256=hashlib.sha256(inputs['queue']).hexdigest(),processes=evidence,function_comparison=review,source_head=binding['source_head'],ownership_verified=all(e['guard']['ownership_verified'] for e in evidence),guard_scope='original accepted/provisional classification preserved',counter_scope='both actual counter values132 serialized and validated every measured context; host checks include warmups',output_scope='source-bound exact CPU reference plus guarded child exit0 and serialized active/sentinel counts; output tensor not separately dumped',producer_return_scope='outlined producer-loop completion/generation publication, not last operand readiness',join_scope='join observed after role stamp stores and160-thread CTA join; includes observation overhead',acquire_scope='global acquire observation before final CTA join, not a release timestamp',cache_scope='actual source touch versus128MiB scrub surrogate; residency/cold cache unverified',aggregation='integer paired deltas first; per-process median/p99 then median across accepted processes, full cohort five; not pooled',quantile='sorted[min(n-1,floor(.99*n))]',pure_compute=False,first_byte_observed=False,hardware_admission_observed=False,marginal_phase_subtraction=False,component_attribution_or_offset_fit=False,E2E_performance_acceptance=False))
 print(json.dumps(dict(processes=len(evidence),complete=len(evidence)==5,placement_rows=len(placement),process_statistic_rows=len(stats),out=str(a.out))))

if __name__=='__main__':main()
