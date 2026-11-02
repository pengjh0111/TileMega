"""Build sourced, context-bounded native task prices from actual probe/profile jobs."""
import argparse
import json
import math
from pathlib import Path
import statistics
import struct
from tilemega.bench.as2_models import RUNS, controlled
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_body_profile import logical_bytes
from tilemega.bench.as2_precision import REPLAY, validate_plan_precision


def combined_timing_status(statuses):
    values=list(statuses)
    if not values:raise ValueError('Missing timing observations')
    if 'contended' in values:return 'contended'
    return 'uncontended_observed' if all(v=='uncontended_observed' for v in values) else 'load_not_fully_characterized'


def task_overheads(stage,profile,cost):
    composite=stage[17]==13 and stage[31] in (4,5,6,7,8,9,12)
    if bool(profile.get('internal_mix_coordination_included'))!=composite:
        raise ValueError('Measured coordination span differs from implementation')
    cube_owner=stage[17]==0 or composite
    overhead=cost['dispatcher_aic_ns'] if cube_owner else cost['dispatcher_aiv_ns']
    # The cooperative owner's reservation already contains nested participant
    # coordination. Disjoint GEMM/epilogue spans exclude both dispatchers.
    if stage[17]==0 and stage[18]:overhead+=cost['dispatcher_aiv_ns']
    internal=2*cost['cross_worker_ready_ns'] if stage[17]==0 and stage[18] else 0.0
    return overhead,internal


def cost_measurement(value,job):
    groups=value['groups'];workers=3*groups
    if (value['status']!='passed' or value['device_abort'] or groups not in (8,14,28) or
            value['workers']!=workers or value['payload_samples_passed']!=groups*1024):
        raise ValueError('Unverified copy/control calibration')
    dispatch=value['dispatcher_ticks_sum'];copy=value['aiv_copy_ticks'];repeats=value['repeats']
    if (len(dispatch)!=workers or len(copy)!=2*groups or repeats<=0 or
            min(dispatch)<=0 or min(copy)<=0 or value['roundtrip_ticks_sum']<=0 or
            value['copy_logical_read_write_bytes']<=0):
        raise ValueError('Missing worker measurements')
    return dict(groups=groups,source_binary_sha256=job['executable_sha256'],
        dispatcher_aic_ns=statistics.median(dispatch[:groups])/repeats,
        dispatcher_aiv_ns=statistics.median(dispatch[groups:])/repeats,
        cross_worker_ready_ns=value['roundtrip_ticks_sum']/(2*repeats),
        gm_bytes_per_ns=value['copy_logical_read_write_bytes']/max(copy),
        timing_status=job['timing_status'],physical_device_id=job['physical_device_id'],
        started_at=job['started_at'],finished_at=job['finished_at'],pid=job['pid'],start_ticks=job['start_ticks'])


def costs(jobs,out):
    out=controlled(out,[RUNS])
    if out.exists():raise FileExistsError(out)
    if len(jobs)<5:raise ValueError('Require five fresh finite calibration processes')
    resources=json.loads((RUNS/'probes/resource_summary.json').read_text());hz=resources['counter_hz']
    if hz!=1_000_000_000:raise ValueError('Require the verified timer differential')
    rows=[];receipts=[]
    for directory in jobs:
        directory=controlled(directory,[RUNS]);job=json.loads((directory/'job.json').read_text())
        if job['status']!='passed':raise ValueError('Failed calibration process')
        lines=(directory/'stdout.log').read_text().splitlines()
        value=next(json.loads(x) for x in lines if x.startswith('{"schema":"tilemega.as2.native_cost_probe.v1"'))
        rows.append(dict(cost_measurement(value,job),job=str(directory)))
        receipts.append(job)
    ordered=sorted(rows,key=lambda x:x['started_at'])
    if len({(x['pid'],x['start_ticks']) for x in rows})!=len(rows) or any(a['finished_at']>b['started_at'] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Calibration processes were reused or overlapped')
    if (len({r['physical_device_id'] for r in rows})!=1 or len({r['source_binary_sha256'] for r in rows})!=1 or
            len({r['groups'] for r in rows})!=1):
        raise ValueError('Mixed calibration device/binary')
    result=dict(schema='tilemega.as2.native_scheduling_costs.v1',status='measured_empirical_context',
        groups=rows[0]['groups'],physical_device_id=rows[0]['physical_device_id'],counter_hz=hz,rows=rows,
        provenance=[str(controlled(j,[RUNS])/'job.json') for j in jobs],
        timing_status=combined_timing_status(r['timing_status'] for r in rows),
        transferability='same private toolchain/device/code paths only; no hardware-peak claim',
        definitions=dict(dispatcher='AcquireData + Stage POD read + single-owner publication; excludes queue/dependency waits',
            cross_worker_ready='half AIC0 / AIV(group1,lane0) round-trip median; symmetric approximation',
            gm='aggregate logical read+write bytes / slowest worker copy duration; working set64MiB/group, GM/L2 path'))
    for key in ('dispatcher_aic_ns','dispatcher_aiv_ns','cross_worker_ready_ns','gm_bytes_per_ns'):
        result[key]=statistics.median(r[key] for r in rows)
        if not math.isfinite(result[key]) or result[key]<=0:raise ValueError('Unknown/nonpositive native cost')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


def tile_body_prices(stage,profile):
    if stage[17]!=0 or stage[31] not in (2,3):return None
    qtail=stage[31]==3
    if qtail and (tuple(stage[20:22])!=(1024,4096) or stage[19]!=2048 or not ((stage[22],stage[18])==(4096,4) or stage[22] in (4096,14336) and stage[18]==1)):
        raise ValueError('Unregistered local Q Body price domain')
    if not qtail and (tuple(stage[20:22])!=(1024,2048) or stage[22] not in (6144,8192) or stage[19]!=1024):
        raise ValueError('Unregistered local Body price domain')
    buckets=profile['body_service_regions'];result=[]
    for name,count in (('regular',1792 if qtail else 896),('tail',256 if qtail else 128)):
        row=buckets[name]
        if row['tasks']!=count or not math.isfinite(row['median']) or row['median']<=0:
            raise ValueError('Incomplete or nonpositive local Body region price')
    for tile in range(stage[19]):result.append(buckets['tail' if tile%(32 if qtail else 16)>=(28 if qtail else 14) else 'regular']['median'])
    return result


def measured_prices(plan,profile_path,cost_path):
    plan,profile_path,cost_path=(controlled(p,[RUNS]) for p in (plan,profile_path,cost_path))
    profile=json.loads(profile_path.read_text());cost=json.loads(cost_path.read_text());imported=json.loads((plan.parent/'import.json').read_text())
    precision=imported.get('precision_contract_id','as2_llm_fp32_replay_bf16_casts_v1')
    if profile.get('precision_contract_id',REPLAY)!=precision:
        raise ValueError('Profile and Plan precision contracts differ')
    m=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes());stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
    validate_plan_precision(precision,stages)
    job=json.loads((Path(profile['job'])/'job.json').read_text())
    if profile['status']!='measured_body_only' or job['status']!='passed' or cost['status']!='measured_empirical_context':
        raise ValueError('Require actual successful Body and control measurements')
    for key in ('model_id','config_sha256','batch','phase'):
        if profile[key]!=imported[key]:raise ValueError('Profile/import identity mismatch: '+key)
    if profile['groups']!=m[1] or cost['groups']!=m[1] or profile['physical_device_id']!=cost['physical_device_id']:
        raise ValueError('Profile/control topology or physical device mismatch')
    if profile['tasks_checked']!=m[3] or len(profile['stages'])!=len(stages):raise ValueError('Unmeasured task domain')
    binding=Path(job['command'][2]);phase=profile['phase']
    observed_metadata=struct.unpack('<12I3Q',(binding/phase/'metadata.bin').read_bytes())
    if observed_metadata[0]!=m[0]:raise ValueError('Profile/runtime ABI mismatch')
    if digest(binding/phase/'stages.bin')!=digest(plan/'stages.bin'):
        # Bound plans relocate addresses, so compare all non-address fields.
        observed=list(struct.iter_unpack('<17Q16I',(binding/phase/'stages.bin').read_bytes()))
        if len(observed)!=len(stages) or any(a[17:]!=b[17:] for a,b in zip(observed,stages)):
            raise ValueError('Measured implementation/shape differs from priced CTG')
    rows=[]
    for i,(s,r) in enumerate(zip(stages,profile['stages'])):
        if r['stage']!=i or r['kind']!=s[17] or r['tasks']!=s[19]:raise ValueError('Unmeasured stage')
        overhead,internal=task_overheads(s,r,cost)
        service=r['body_service_ns']['median']+overhead+internal
        if not math.isfinite(service) or service<=0:raise ValueError('Nonpositive task price')
        rows.append(dict(stage=i,kind=s[17],variant=s[31],tasks=s[19],rows=s[20],columns=s[21],k=s[22],
            service_ns=service,body_ns=r['body_service_ns']['median'],dispatcher_ns=overhead,
            internal_handoff_estimate_ns=internal,
            internal_sync_source='included in measured composite span' if r.get('internal_mix_coordination_included') else 'cross-group symmetric readiness used conservatively for two same-group GEMM messages' if internal else 'no composite handoff',
            logical_gm_bytes=[logical_bytes(s,m,tile,profile['past']) for tile in range(s[19])]))
        per_tile=tile_body_prices(s,r)
        if per_tile is not None:
            rows[-1]['tile_service_ns']=[v+overhead+internal for v in per_tile]
            rows[-1]['tile_price_source']='measured regular/tail region medians plus sourced overhead'
    result=dict(schema='tilemega.as2.native_task_prices.v1',status='sourced_empirical_context',
        precision_contract_id=precision,
        price_provider_version=4,dispatcher_accounting='actual_owner_role_and_disjoint_participant',
        model_id=profile['model_id'],config_sha256=profile['config_sha256'],batch=m[8],seq=m[9],groups=m[1],runtime_abi=m[0],
        phase=phase,past_min=0,past_max=profile['past'],stages=rows,
        source_profile=str(profile_path),source_profile_sha256=digest(profile_path),
        source_costs=str(cost_path),source_costs_sha256=digest(cost_path),
        source_plan=str(plan),source_stages_sha256=digest(plan/'stages.bin'),
        swiglu_l1_k=256 if any(s[17]==0 and s[31]==5 for s in stages) else 128,
        gm_bytes_per_ns=cost['gm_bytes_per_ns'],cross_worker_ready_ns=cost['cross_worker_ready_ns'],
        timing_status=combined_timing_status([cost['timing_status'],profile['timing_status']]),accuracy='stage or explicit local-region median plus measured metadata/publication overhead; conservative active-KV upper bound',
        limitations=['no automatic transfer to another model/G/geometry','MIX service sums serial Cube/epilogue phases',
            'GM calendar is conservative and not a detailed pipelined interconnect model','contended costs are context-local, not portable hardware constants'])
    return result


def prices(plan,profile_path,cost_path,out):
    out=controlled(out,[RUNS])
    if out.exists():raise FileExistsError(out)
    result=measured_prices(plan,profile_path,cost_path)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='stages'}))


def context_weights(buckets,minimum=64,maximum=1086):
    cursor=minimum;weights=[]
    for row in buckets:
        low,high,past=(row[k] for k in ('past_min','past_max','past'))
        if (any(type(x)!=int for x in (low,high,past)) or low!=cursor or
                not low<=past<=high<=maximum):
            raise ValueError('Context buckets must cover the decode interval once with an in-bucket measurement')
        weights.append(high-low+1);cursor=high+1
    if not buckets or cursor!=maximum+1:raise ValueError('Incomplete decode context coverage')
    return weights


def weighted_stage_prices(tables,weights):
    if not tables or len(tables)!=len(weights) or any(w<=0 for w in weights):
        raise ValueError('Missing context prices or positive counts')
    total=sum(weights);result=[]
    if len({len(t) for t in tables})!=1:raise ValueError('Context stage count changed')
    def mean(values):
        if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Unknown/nonpositive context service')
        return sum(v*w for v,w in zip(values,weights))/total
    for rows in zip(*tables):
        base=rows[0]
        identity=('stage','kind','variant','tasks','rows','columns','k')
        if any(any(r[k]!=base[k] for k in identity) for r in rows):
            raise ValueError('Implementation/geometry changed between contexts')
        item=dict(base)
        for key in ('service_ns','body_ns','dispatcher_ns'):
            item[key]=mean([r[key] for r in rows])
        item['internal_handoff_estimate_ns']=sum(r['internal_handoff_estimate_ns']*w for r,w in zip(rows,weights))/total
        if any('tile_service_ns' in r for r in rows):
            if any(len(r.get('tile_service_ns',[]))!=base['tasks'] for r in rows):
                raise ValueError('Incomplete context tile services')
            item['tile_service_ns']=[mean(v) for v in zip(*(r['tile_service_ns'] for r in rows))]
        result.append(item)
    return result


def contexts(plan,profile_paths,bounds,cost_path,out):
    plan,cost_path,out=(controlled(p,[RUNS]) for p in (plan,cost_path,out))
    bundle=out.with_name(out.stem+'_contexts.json')
    if out.exists() or bundle.exists():raise FileExistsError(out)
    if len(profile_paths)!=len(bounds):raise ValueError('One observed profile per declared context bucket required')
    tables=[];buckets=[];identities=[]
    for path,(low,high) in zip(profile_paths,bounds):
        path=controlled(path,[RUNS]);profile=json.loads(path.read_text())
        if profile['phase']!='decode':raise ValueError('Context weighting applies to decode, not prefill')
        tables.append(measured_prices(plan,path,cost_path));identities.append(profile)
        buckets.append(dict(past_min=low,past_max=high,past=profile['past'],source_profile=str(path),
            source_profile_sha256=digest(path)))
    weights=context_weights(buckets)
    identity=('model_id','config_sha256','batch','groups','runtime_abi','physical_device_id','binary_sha256','precision_contract_id')
    if any(any(p.get(k)!=identities[0].get(k) for k in identity) for p in identities):
        raise ValueError('Context profiles use different model/implementation/device identities')
    result=dict(tables[0]);result['stages']=weighted_stage_prices([t['stages'] for t in tables],weights)
    m=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes())
    stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
    # Traffic is known from the actual access recipe, so average all 1023
    # lengths exactly; service time remains an explicitly piecewise estimate.
    for row,s in zip(result['stages'],stages):
        if s[17]==13:
            value=math.ceil(sum(logical_bytes(s,m,0,p) for p in range(64,1087))/1023)
            row['logical_gm_bytes']=[value]*s[19]
    timing=combined_timing_status([p['timing_status'] for p in identities]+[result['timing_status']])
    source=dict(schema='tilemega.as2.native_body_contexts.v1',status='measured_body_contexts',
        phase='decode',past_min=64,past_max=1086,forward_count=1023,buckets=buckets,
        weight_rule='one occurrence per actual decode past length',timing_status=timing)
    bundle.write_text(json.dumps(source,indent=2)+'\n')
    result.update(price_provider_version=5,past_min=64,past_max=1086,source_profile=str(bundle),
        source_profile_sha256=digest(bundle),timing_status=timing,
        context_aggregation=dict(forwards=1023,bucket_counts=weights,body_service='piecewise constant observed median',
            logical_bytes='exact recipe average across past64..1086, rounded upward to integer bytes'),
        accuracy='context-count-weighted measured service plus sourced overhead; exact logical traffic recipe average')
    result['limitations']+=['three-context service approximation does not establish every-length latency; full-request admission remains required']
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='stages'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('costs');c.add_argument('--jobs',type=Path,nargs='+',required=True);c.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('prices')
    for name in ('plan','profile','costs','out'):q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('contexts')
    for name in ('plan','costs','out'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--profiles',type=Path,nargs='+',required=True)
    q.add_argument('--bounds',nargs='+',required=True,help='Inclusive ranges, e.g. 64:127 128:511 512:1086')
    a=p.parse_args()
    if a.command=='costs':costs(a.jobs,a.out)
    elif a.command=='prices':prices(a.plan,a.profile,a.costs,a.out)
    else:contexts(a.plan,a.profiles,[tuple(map(int,b.split(':'))) for b in a.bounds],a.costs,a.out)
