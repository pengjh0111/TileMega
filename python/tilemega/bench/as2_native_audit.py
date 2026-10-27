"""Audit existing native launch/frontier receipts; never submit device work."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
from tilemega.bench.as2_models import RUNS, controlled


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def mix_prefix(metadata,tasks,stages,group,epoch):
    if metadata[0] not in (2,3,4):raise ValueError('Unknown internal MIX token ABI')
    last=max((t for t in tasks if t[2]==group and t[6]),key=lambda t:t[3],default=None)
    if last is None:return 0
    prefix=epoch*metadata[7]+last[3]
    if metadata[0]==4:
        stage=stages[last[0]];phase=2 if stage[17]==13 and stage[31] in (4,5,6) else 0
        prefix=4*prefix+phase
    return prefix


def audit(plan,jobs,out):
    if len(jobs)<5:raise ValueError('Synchronization admission needs five fresh processes')
    m=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes());groups,workers,count,stride=m[1],m[2],m[3],m[7]
    offsets=struct.unpack('<'+'I'*(workers+1),(plan/'offsets.bin').read_bytes())
    tasks=list(struct.iter_unpack('<7I',(plan/'tasks.bin').read_bytes()))
    stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
    def group(w):return w if w<groups else (w-groups)//2
    waits=list(struct.iter_unpack('<2I',(plan/'waits.bin').read_bytes()))
    cross=sum(group(w)!=group(t[2]) for t in tasks for w,p in waits[t[4]:t[4]+t[5]])
    records=[];identities=set();binary=None;previous_end=None;canonical=None
    for directory in jobs:
        job=json.loads((directory/'job.json').read_text());command=job['command']
        if job['status']!='passed' or len(command)!=7 or command[1]!=str(plan) or int(command[5])<2:
            raise ValueError('Require completed multi-forward native jobs of this plan')
        identity=(job['pid'],job['start_ticks'])
        if identity in identities:raise ValueError('Each synchronization trial must use a fresh process')
        identities.add(identity)
        started=datetime.datetime.fromisoformat(job['started_at']);ended=datetime.datetime.fromisoformat(job['finished_at'])
        if previous_end is not None and started<previous_end:raise ValueError('Selected project jobs overlapped')
        previous_end=ended
        if binary is not None and binary!=job['executable_sha256']:raise ValueError('Mixed native binary identities')
        binary=job['executable_sha256'];output=controlled(Path(command[3]),[RUNS]);forwards=int(command[5])
        for epoch in range(forwards):
            data=(output/f'control_{epoch}.bin').read_bytes();control=struct.unpack('<'+'I'*(len(data)//4),data)
            if control[(workers+2*groups)*128]:raise ValueError('Device abort was recorded')
            for w in range(workers):
                if control[w*128]!=epoch*stride+offsets[w+1]-offsets[w]:raise ValueError('Incorrect FIFO completion prefix')
            for g in range(groups):
                last=mix_prefix(m,tasks,stages,g,epoch)
                for cell in (workers+g,workers+groups+g):
                    if control[cell*128]!=last:raise ValueError('MIX ready/done epoch did not match its owner')
        launches=[json.loads(line) for line in (output/'launches.jsonl').read_text().splitlines()]
        if len(launches)!=forwards or any(x['kernel']!='tilemega_llm' or x['launch_index']!=i or x['past']!=int(command[4])+i for i,x in enumerate(launches)):
            raise ValueError('Actual launch receipts disagree with forward count')
        if int(command[6]):
            trace=list(struct.iter_unpack('<8Q',(output/'trace.bin').read_bytes()))
            if len(trace)!=count or any((r[0],r[1],r[2],r[5])!=(t[2],t[0],t[1],forwards-1) or r[4]<r[3] or r[6:8]!=stages[t[0]][:2] for t,r in zip(tasks,trace)):
                raise ValueError('Diagnostic task ownership/ABI roundtrip is invalid')
        hashes={p.name:digest(p) for p in output.glob('*.bin') if p.name!='trace.bin'}
        if canonical is not None and hashes!=canonical:raise ValueError('Payload/KV/token/control differs across fresh processes')
        canonical=hashes
        summary=next(json.loads(line) for line in (directory/'stdout.log').read_text().splitlines() if line.startswith('{"status":'))
        records.append(dict(job=str(directory),pid=job['pid'],start_ticks=job['start_ticks'],
            diagnostic=bool(int(command[6])),forwards=forwards,host_launches=len(launches),
            physical_device=job['physical_device_id'],forward_host_ms=summary['forward_host_ms'],
            wrapper_elapsed_s=job['elapsed_s'],timing_status=job['timing_status'],status='pass'))
    owners=[0]*workers
    for t in tasks:owners[t[2]]+=1
    result=dict(schema='tilemega.as2.native_stability.v1',status='pass',scope='decode_frontier_mix_lifecycle_four_forwards_not_model_numeric_admission',
        processes_passed=len(records),processes_total=len(records),tasks_per_forward=count,
        groups=groups,owner_tasks_per_worker=owners,queue_references_per_worker=[offsets[w+1]-offsets[w] for w in range(workers)],
        lowered_cross_group_waits=cross,compared_buffers=len(canonical),payload_bitwise_equal=True,
        selected_submissions_sequential=True,device_locked=False,binary_sha256=binary,
        numerical_gate='not_implied',performance_gate='not_implied',records=records)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))


def compare_capture_payloads(diagnostic,fast,required,active_extents=None):
    if not fast or not fast.keys()<=diagnostic.keys() or not required<=fast.keys():
        raise ValueError('Missing mandatory or unmatched fast capture')
    if any(value!=diagnostic[name] for name,value in fast.items()):
        raise ValueError('Fast payload, KV or token differs')
    extra=sorted(diagnostic.keys()-fast.keys())
    # Active diagnostic copies require explicit phase extents. Fast KV,
    # tokens, controls and logits remain mandatory and must match exactly.
    for name in extra:
        path=Path(name)
        if active_extents and name in active_extents:
            if diagnostic[name][1]!=active_extents[name] or not 0<active_extents[name]<=1<<20:
                raise ValueError('Active diagnostic capture extent changed')
            continue
        if path.parent!=Path('decode') or not path.stem.isdecimal() or path.suffix!='.bin' or diagnostic[name][1]<=1<<20:
            raise ValueError('Unexpected diagnostic-only capture: '+name)
    return extra


def audit_request(binding,jobs,out,reset_admission=True):
    if len(jobs)<(5 if reset_admission else 2):raise ValueError('Insufficient fresh processes for the selected admission')
    records=[];identities=set();canonical=None;binary=None;previous_end=None;scope=None;captures={}
    description=json.loads((binding/'request.json').read_text())
    ss=list(struct.iter_unpack('<17Q16I',(binding/'decode/stages.bin').read_bytes()))
    heads=[s for s in ss if s[17]==0 and s[18]==3]
    if len(heads)!=1:raise ValueError('Require one complete vocabulary projection')
    logit=next(b for b in description['buffers'] if b['offset_bytes']==heads[0][16])
    required={'tokens.bin','prefill/control.bin','decode/control.bin',f'decode/{logit["id"]}.bin'}
    for phase,entry in description['phases'].items():
        if any(digest(binding/phase/name)!=sha for name,sha in entry['bound_hashes'].items()):
            raise ValueError('Bound plan changed before capture audit')
    for directory in jobs:
        job=json.loads((directory/'job.json').read_text());command=job['command']
        if job['status']!='passed' or len(command)!=8 or command[1]!='request' or command[2]!=str(binding):
            raise ValueError('Require completed native request jobs of this shared binding')
        identity=(job['pid'],job['start_ticks'])
        if identity in identities:raise ValueError('Request trials require fresh processes')
        identities.add(identity)
        started=datetime.datetime.fromisoformat(job['started_at']);ended=datetime.datetime.fromisoformat(job['finished_at'])
        if previous_end is not None and started<previous_end:raise ValueError('Selected request jobs overlapped')
        previous_end=ended
        if binary is not None and binary!=job['executable_sha256']:raise ValueError('Mixed request binary identities')
        binary=job['executable_sha256'];output=controlled(Path(command[3]),[RUNS])
        outputs,warmup,repetitions,diagnostic=map(int,command[4:])
        if diagnostic not in (0,1) or (not reset_admission and outputs>65):
            raise ValueError('Capture comparison requires short diagnostic/fast requests with equal KV history extents')
        if outputs<2 or warmup+repetitions<(2 if reset_admission else 1):raise ValueError('Insufficient complete independent requests')
        current=(outputs,warmup,repetitions)
        if scope is not None and current!=scope:raise ValueError('Incompatible request trial scopes')
        scope=current
        launches=[json.loads(line) for line in (output/'launches.jsonl').read_text().splitlines()]
        if len(launches)!=(warmup+repetitions)*outputs:raise ValueError('Incorrect actual model-forward launch count')
        for repetition in range(-warmup,repetitions):
            request=output/f'request_{repetition}'
            for phase,epoch in [('prefill',0),('decode',outputs-2)]:
                plan=binding/phase;m=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes())
                groups,workers,count,stride=m[1],m[2],m[3],m[7]
                offsets=struct.unpack('<'+'I'*(workers+1),(plan/'offsets.bin').read_bytes())
                tasks=list(struct.iter_unpack('<7I',(plan/'tasks.bin').read_bytes()))
                stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
                data=(request/phase/'control.bin').read_bytes();control=struct.unpack('<'+'I'*(len(data)//4),data)
                if control[(workers+2*groups)*128]:raise ValueError('Request device abort')
                for w in range(workers):
                    if control[w*128]!=epoch*stride+offsets[w+1]-offsets[w]:raise ValueError('Incorrect request FIFO prefix')
                for g in range(groups):
                    last=mix_prefix(m,tasks,stages,g,epoch)
                    if any(control[cell*128]!=last for cell in (workers+g,workers+groups+g)):
                        raise ValueError('Incorrect request MIX lifecycle prefix')
                if diagnostic:
                    trace=list(struct.iter_unpack('<8Q',(request/phase/'trace.bin').read_bytes()))
                    if len(trace)!=count or any((r[0],r[1],r[2],r[5])!=(t[2],t[0],t[1],epoch) or r[4]<r[3] or r[6:8]!=stages[t[0]][:2] for t,r in zip(tasks,trace)):
                        raise ValueError('Invalid request task ownership/ABI receipt')
            hashes={str(p.relative_to(request)):(digest(p),p.stat().st_size) for p in request.rglob('*.bin') if p.name!='trace.bin'}
            previous=canonical if reset_admission else captures.get(diagnostic)
            if previous is not None and hashes!=previous:raise ValueError('Request reset payload, KV or token differs')
            if reset_admission:canonical=hashes
            else:captures[diagnostic]=hashes
        for i,entry in enumerate(launches):
            step=i%outputs;rep=i//outputs-warmup
            if entry['repetition']!=rep or entry['forward_index']!=step or entry['past']!=(63+step if step else 0) or entry['epoch']!=(step-1 if step else 0) or entry['seq']!=(1 if step else 64):
                raise ValueError('Request launch/position/epoch mismatch')
        records.append(dict(job=str(directory),pid=job['pid'],start_ticks=job['start_ticks'],diagnostic=bool(diagnostic),
            requests=warmup+repetitions,outputs_per_request=outputs,host_launches=len(launches),
            wrapper_elapsed_s=job['elapsed_s'],physical_device=job['physical_device_id'],status='pass'))
    if not reset_admission and {r['diagnostic'] for r in records}!={False,True}:
        raise ValueError('C2 needs both diagnostic and fast executions')
    extra=[]
    if not reset_admission:
        active_extents={}
        extent_path=binding/'decode/active_bytes.bin'
        if extent_path.exists():
            extents=list(struct.iter_unpack('<Q',extent_path.read_bytes()))
            if len(extents)!=len(description['buffers']):raise ValueError('Invalid active diagnostic extent count')
            for b,(size,) in zip(description['buffers'],extents):
                if size>b['bytes']:raise ValueError('Active extent exceeds shared allocation')
                if not b.get('required_input') and 0<size<=1<<20:
                    active_extents[f'decode/active/{b["id"]}.bin']=size
        extra=compare_capture_payloads(captures[1],captures[0],required,active_extents);canonical=captures[0]
    result=dict(schema='tilemega.as2.request_stability.v1' if reset_admission else 'tilemega.as2.request_path_c2.v1',status='pass',
        scope='prefill_decode_stream_and_request_reset_not_numeric_admission' if reset_admission else 'diagnostic_fast_bitwise_equivalence_not_C3',
        processes_passed=len(records),processes_total=len(records),complete_requests=sum(r['requests'] for r in records),
        compared_files_per_request=len(canonical),payload_kv_tokens_controls_bitwise_equal=True,
        selected_submissions_sequential=True,device_locked=False,binary_sha256=binary,records=records,
        numerical_gate='not_implied',performance_gate='not_implied')
    if not reset_admission:
        result['diagnostic_only_large_capture_paths']=extra
        result['diagnostic_only_large_capture_count']=len(extra)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','diagnostic_only_large_capture_paths')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);kind=p.add_mutually_exclusive_group(required=True)
    kind.add_argument('--plan',type=Path);kind.add_argument('--request',type=Path);kind.add_argument('--compare-request',type=Path)
    p.add_argument('--jobs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();function=audit_request if a.request or a.compare_request else audit
    kwargs={'reset_admission':False} if a.compare_request else {}
    function(controlled(a.request or a.compare_request or a.plan,[RUNS]),[controlled(j,[RUNS]) for j in a.jobs],controlled(a.out,[RUNS]),**kwargs)

if __name__=='__main__':main()
