"""Extract measured Body service intervals; never substitute them for total task prices."""
import argparse
import json
from pathlib import Path
import statistics
import struct
from tilemega.bench.as2_models import RUNS, controlled
from tilemega.bench.as2_native import digest


def logical_bytes(s,m,tile,past):
    absent=(1<<64)-1
    kind,epilogue,_,rows,columns,k,extent,width,group,_,query_rows=s[17:28]
    if kind in (0,6):
        tm,tn=(16,128) if s[31] else (32,64)
        if extent>1:
            if extent not in (2,3,7) or width!=(k+extent-1)//extent or group!=1 or s[31]!=1:
                raise ValueError('Unregistered split/combine traffic contract')
            if extent==7 and (columns,k,epilogue)!=(1024,4096,0):
                raise ValueError('Seven-part traffic requires the observed KV store family')
            if kind==0:part=tile%extent;tile//=extent
        nt=(columns+tn-1)//tn;mr=min(tm,rows-tile//nt*tm);nr=min(tn,columns-tile%nt*tn)
        if mr<=0 or nr<=0:raise ValueError('GEMM traffic coordinate outside output')
        if kind==6:
            if (extent not in (2,3) or epilogue!=1) and not (extent==7 and columns==1024 and k==4096 and epilogue==0):
                raise ValueError('Unregistered combine epilogue traffic')
            # FP32 partial reads, BF16 raw write, then raw/residual reads+output.
            return (4*extent+2+(6 if epilogue else 0))*mr*nr
        if extent>1:
            if epilogue:raise ValueError('Partial tasks cannot publish a final epilogue')
            size=min(width,k-part*width)
            return 2*(mr*size+nr*size)+4*mr*nr
        n=2*(mr*k+nr*k+mr*nr)
        if s[31]==2:
            if (rows,columns,epilogue)!=(1024,2048,1) or k not in (6144,8192) or extent>1:
                raise ValueError('Unregistered local tail reduction traffic')
            if tile%nt*tn>=1792:n+=56*mr*nr
        if epilogue==1:n+=6*mr*nr
        elif epilogue==2:n+=3*mr*nr
        elif epilogue==3:n+=2*mr*nr+8*mr
        return n
    if kind==1:return 6*width
    if kind==11:return 4*width
    if kind==15:return 8*width+4
    if kind==25:
        if (extent,width,group,query_rows,s[26])!=(8,768,1,4,128) or not 0<=tile<rows:
            raise ValueError('Unregistered QKV packing traffic domain')
        return 4*extent*width
    if kind!=13:raise ValueError('No logical GM traffic contract for stage kind '+str(kind))
    if s[26]!=1088:raise ValueError('Only the actual unified KV body is profiled')
    qrotate=(14 if s[5]!=absent else 8)*width;krotate=(14 if s[6]!=absent else 8)*width
    if s[31] in (4,5,6):
        if m[9]==1:
            positions=1;rows=group;keys=1088;valid=past+1
            n=krotate+4*width+12*valid*width+8*(keys-valid)*width
            valid_scores=rows*valid
        else:
            blocks=(m[9]*group+query_rows-1)//query_rows;qb=(tile//extent)%blocks
            positions=query_rows//group;rows=query_rows;keys=valid=m[9]
            n=valid*(krotate+4*width)+positions*8*width+12*valid*width
            valid_scores=sum(qb*positions+r//group+1 for r in range(rows))
        n+=rows*(qrotate+6*width)
        mt=(rows+31)//32;nt=(keys+63)//64
        n+=4*(rows*width*nt+keys*width*mt+rows*keys)
        n+=4*valid_scores+4*rows*keys
        n+=4*(rows*keys+keys*width*mt+rows*width)
        return n+6*rows*width
    def attend(valid):
        chunks=(valid+31)//32
        if s[31]==1:return qrotate+10*chunks*width+4*valid*width+20*chunks+2*width
        if s[31]==3:return qrotate+10*chunks*width+4*valid*width+16*valid+2*width
        return qrotate+10*chunks*width+4*valid*width+12*valid+2*width
    if m[9]==1:return krotate+4*width+group*attend(past+1)
    blocks=(m[9]*group+query_rows-1)//query_rows;qb=(tile//extent)%blocks
    first=qb*query_rows;end=min((qb+1)*query_rows,m[9]*group)
    owned_positions=sum(pos*group//query_rows==qb for pos in range(m[9]))
    return m[9]*(krotate+4*width)+owned_positions*8*width+sum(attend(q//group+1) for q in range(first,end))


def extract(binding,job_path,phase,repetition,out):
    binding,job_path,out=(controlled(p,[RUNS]) for p in (binding,job_path,out))
    if out.exists():raise FileExistsError(out)
    job=json.loads((job_path/'job.json').read_text());command=job['command']
    if job['status']!='passed' or len(command)!=8 or command[1]!='request' or command[2]!=str(binding) or command[7]!='2':
        raise ValueError('Require an actual successful Body-profile request')
    identity=json.loads((binding/'request.json').read_text());plan=binding/phase
    for name,sha in identity['phases'][phase]['bound_hashes'].items():
        if digest(plan/name)!=sha:raise ValueError('Profile plan identity changed')
    m=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes())
    tasks=list(struct.iter_unpack('<7I',(plan/'tasks.bin').read_bytes()))
    stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
    offsets=struct.unpack('<'+'I'*(m[2]+1),(plan/'offsets.bin').read_bytes())
    queue=struct.unpack('<'+'I'*m[6],(plan/'queue.bin').read_bytes())
    data=(Path(command[3])/f'request_{repetition}'/phase/'profile.bin').read_bytes()
    if len(data)!=m[6]*512:raise ValueError('Wrong queue-profile byte extent')
    resources=json.loads((RUNS/'probes/resource_summary.json').read_text())
    timer_job=Path(resources['counter_evidence'])
    timer_receipt=json.loads((timer_job/'job.json').read_text())
    if timer_receipt['status']!='passed' or resources['counter_hz']<=0:raise ValueError('Require executed timer evidence')
    ns_per_tick=1e9/resources['counter_hz'];expected_epoch=0 if phase=='prefill' else int(command[4])-2
    past=0 if phase=='prefill' else int(command[4])+62
    intervals=[{} for _ in tasks];worker_windows=[]
    for w in range(m[2]):
        starts=[];ends=[];busy=0
        for slot in range(offsets[w],offsets[w+1]):
            r=struct.unpack_from('<8Q',data,slot*512);id=queue[slot];task=tasks[id]
            participant=int(w!=task[2])
            if r[:3]!=(w,task[0],task[1]) or r[5:]!=(expected_epoch,id,participant) or r[4]<=r[3]:
                raise ValueError('Corrupt/uninitialized Body profile or ownership mismatch')
            if participant and not (task[6]==1 and task[2]<m[1] and w==m[1]+2*task[2]):
                raise ValueError('Unregistered participant in measured Body profile')
            if w in intervals[id]:raise ValueError('Duplicate worker service record')
            intervals[id][w]=(r[4]-r[3])*ns_per_tick
            starts.append(r[3]);ends.append(r[4]);busy+=(r[4]-r[3])*ns_per_tick
        window=(max(ends)-min(starts))*ns_per_tick if starts else None
        worker_windows.append(dict(worker=w,body_ns=busy,profiled_queue_window_ns=window,
            body_fraction=busy/window if window else None,
            note='excludes metadata/waits/frontiers/profile writes; not a hardware utilization counter'))
    service=[[] for _ in stages];traffic=[[] for _ in stages];regions={}
    for id,(task,values) in enumerate(zip(tasks,intervals)):
        if len(values)!=1+task[6]:raise ValueError('Incomplete owner/participant service profile')
        # A single GEMM's Cube/epilogue intervals are disjoint. A cooperative
        # attention owner span already contains its nested participant phases.
        composite=stages[task[0]][17]==13 and stages[task[0]][31] in (4,5,6)
        measured=values[task[2]] if composite else sum(values.values())
        service[task[0]].append(measured)
        if stages[task[0]][17]==0 and stages[task[0]][31]==2:
            buckets=regions.setdefault(task[0],dict(regular=[],tail=[]))
            buckets['tail' if task[1]%16>=14 else 'regular'].append(measured)
        traffic[task[0]].append(logical_bytes(stages[task[0]],m,task[1],past))
    rows=[]
    for i,(s,values,bytes_) in enumerate(zip(stages,service,traffic)):
        if not values:raise ValueError('Unprofiled stage cannot receive an invented price')
        ordered=sorted(values)
        rows.append(dict(stage=i,kind=s[17],epilogue=s[18],tasks=len(values),
            shape=dict(rows=s[20],columns=s[21],k=s[22],extent=s[23],width=s[24],group=s[25]),
            body_service_ns=dict(min=min(values),median=statistics.median(values),p95=ordered[min(len(ordered)-1,int(.95*len(ordered)))],max=max(values)),
            logical_gm_payload_bytes=dict(min=min(bytes_),median=statistics.median(bytes_),max=max(bytes_),sum=sum(bytes_)),
            service_measurement='composite_owner_reservation_span' if s[17]==13 and s[31] in (4,5,6) else 'serial_compute_copy_intervals',
            internal_mix_coordination_included=s[17]==13 and s[31] in (4,5,6),
            dispatcher_sync_ns=None,total_task_service_ns=None,calibration_status='body_only_not_total_task_price'))
        if i in regions:
            rows[-1]['body_service_regions']={name:dict(tasks=len(samples),median=statistics.median(samples),
                min=min(samples),max=max(samples)) for name,samples in regions[i].items()}
    result=dict(schema='tilemega.as2.native_body_profile.v1',status='measured_body_only',phase=phase,past=past,
        model_id=identity['model_id'],config_sha256=identity['config_sha256'],batch=m[8],groups=m[1],runtime_abi=m[0],
        job=str(job_path),binary_sha256=job['executable_sha256'],physical_device_id=job['physical_device_id'],
        timing_status=job['timing_status'],counter_hz=resources['counter_hz'],counter_evidence=str(timer_job),
        profile_sha256=__import__('hashlib').sha256(data).hexdigest(),queue_refs_checked=m[6],tasks_checked=m[3],
        task_pricing_admitted=False,shared_gm_bytes_per_ns=None,cross_worker_ready_ns=None,
        exclusions=['external_dependency_waits','plan_metadata_acquire','frontier_publish','diagnostic_writes'],
        mix_timing_contract='single-phase GEMM excludes ready/done; multi-phase attention owner span includes internal coordination without adding its overlapping participant span',
        gm_bytes_contract='logical DataCopy payload only; excludes padding/metadata/cache effects; not measured HBM traffic',
        worker_windows=worker_windows,stages=rows)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='stages'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','job','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('prefill','decode'),required=True);p.add_argument('--repetition',type=int,default=0)
    a=p.parse_args();extract(a.binding,a.job,a.phase,a.repetition,a.out)
