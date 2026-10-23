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


def task_overheads(stage,profile,cost):
    composite=stage[17]==13 and stage[31] in (4,5,6)
    if bool(profile.get('internal_mix_coordination_included'))!=composite:
        raise ValueError('Measured coordination span differs from implementation')
    cube_owner=stage[17]==0 or composite
    overhead=cost['dispatcher_aic_ns'] if cube_owner else cost['dispatcher_aiv_ns']
    # The cooperative owner's reservation already contains nested participant
    # coordination. Disjoint GEMM/epilogue spans exclude both dispatchers.
    if stage[17]==0 and stage[18]:overhead+=cost['dispatcher_aiv_ns']
    internal=2*cost['cross_worker_ready_ns'] if stage[17]==0 and stage[18] else 0.0
    return overhead,internal


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
        if value['status']!='passed' or value['device_abort'] or value['groups']!=8 or value['payload_samples_passed']!=8192:
            raise ValueError('Unverified copy/control calibration')
        if len(value['dispatcher_ticks_sum'])!=24 or len(value['aiv_copy_ticks'])!=16 or min(value['aiv_copy_ticks'])<=0:
            raise ValueError('Missing worker measurements')
        rows.append(dict(job=str(directory),source_binary_sha256=job['executable_sha256'],
            dispatcher_aic_ns=statistics.median(value['dispatcher_ticks_sum'][:8])/value['repeats'],
            dispatcher_aiv_ns=statistics.median(value['dispatcher_ticks_sum'][8:])/value['repeats'],
            cross_worker_ready_ns=value['roundtrip_ticks_sum']/(2*value['repeats']),
            gm_bytes_per_ns=value['copy_logical_read_write_bytes']/max(value['aiv_copy_ticks']),
            timing_status=job['timing_status'],physical_device_id=job['physical_device_id'],
            started_at=job['started_at'],finished_at=job['finished_at'],pid=job['pid'],start_ticks=job['start_ticks']))
        receipts.append(job)
    ordered=sorted(rows,key=lambda x:x['started_at'])
    if len({(x['pid'],x['start_ticks']) for x in rows})!=len(rows) or any(a['finished_at']>b['started_at'] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Calibration processes were reused or overlapped')
    if len({r['physical_device_id'] for r in rows})!=1 or len({r['source_binary_sha256'] for r in rows})!=1:
        raise ValueError('Mixed calibration device/binary')
    result=dict(schema='tilemega.as2.native_scheduling_costs.v1',status='measured_empirical_context',
        groups=8,physical_device_id=rows[0]['physical_device_id'],counter_hz=hz,rows=rows,
        provenance=[str(controlled(j,[RUNS])/'job.json') for j in jobs],
        timing_status='contended' if any(r['timing_status']=='contended' for r in rows) else 'uncontended_observed',
        transferability='same private toolchain/device/code paths only; no hardware-peak claim',
        definitions=dict(dispatcher='AcquireData + Stage POD read + single-owner publication; excludes queue/dependency waits',
            cross_worker_ready='half AIC0 / AIV(group1,lane0) round-trip median; symmetric approximation',
            gm='aggregate logical read+write bytes / slowest worker copy duration, 512MiB working set; GM/L2 path'))
    for key in ('dispatcher_aic_ns','dispatcher_aiv_ns','cross_worker_ready_ns','gm_bytes_per_ns'):
        result[key]=statistics.median(r[key] for r in rows)
        if not math.isfinite(result[key]) or result[key]<=0:raise ValueError('Unknown/nonpositive native cost')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


def prices(plan,profile_path,cost_path,out):
    plan,profile_path,cost_path,out=(controlled(p,[RUNS]) for p in (plan,profile_path,cost_path,out))
    if out.exists():raise FileExistsError(out)
    profile=json.loads(profile_path.read_text());cost=json.loads(cost_path.read_text());imported=json.loads((plan.parent/'import.json').read_text())
    m=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes());stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
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
    result=dict(schema='tilemega.as2.native_task_prices.v1',status='sourced_empirical_context',
        price_provider_version=2,dispatcher_accounting='actual_owner_role_and_disjoint_participant',
        model_id=profile['model_id'],config_sha256=profile['config_sha256'],batch=m[8],seq=m[9],groups=m[1],runtime_abi=m[0],
        phase=phase,past_min=0,past_max=profile['past'],stages=rows,
        source_profile=str(profile_path),source_profile_sha256=digest(profile_path),
        source_costs=str(cost_path),source_costs_sha256=digest(cost_path),
        source_plan=str(plan),source_stages_sha256=digest(plan/'stages.bin'),
        gm_bytes_per_ns=cost['gm_bytes_per_ns'],cross_worker_ready_ns=cost['cross_worker_ready_ns'],
        timing_status=cost['timing_status'],accuracy='per-stage median plus measured metadata/publication overhead; conservative active-KV upper bound',
        limitations=['no automatic transfer to another model/G/geometry','MIX service sums serial Cube/epilogue phases',
            'GM calendar is conservative and not a detailed pipelined interconnect model','contended costs are context-local, not portable hardware constants'])
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='stages'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('costs');c.add_argument('--jobs',type=Path,nargs='+',required=True);c.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('prices')
    for name in ('plan','profile','costs','out'):q.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.command=='costs':costs(a.jobs,a.out)
    else:prices(a.plan,a.profile,a.costs,a.out)
