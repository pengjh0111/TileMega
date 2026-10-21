"""Bounded ABI4 MIX Body/protocol probe; synthetic queues are not an E2E Plan."""
import argparse
import json
import math
import struct
from pathlib import Path
import torch
from tilemega.bench.as2_models import RUNS,REPO,controlled
from tilemega.bench.as2_native import digest

ABSENT=(1<<64)-1
G=8
D=128
CAP=1088


def read(path,shape,dtype=torch.bfloat16):
    count=math.prod(shape)
    if path.stat().st_size!=count*torch.empty((),dtype=dtype).element_size():raise ValueError('Tensor extent mismatch')
    return torch.from_file(str(path),shared=False,size=count,dtype=dtype).reshape(shape)


def save(path,value):
    path.write_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())


def prepare(source,rope,out):
    source=controlled(source,[RUNS]);rope=controlled(rope,[RUNS])
    receipt=json.loads((source/'reference.json').read_text())
    observed=next(v for v in receipt['files'] if v['name']=='l0.qkv')
    if receipt['model_id']!='qwen3_0_6b' or receipt['seq']!=64 or receipt['batch']!=1 or digest(source/observed['file'])!=observed['sha256']:
        raise ValueError('Require actual independently observed Qwen first-layer QKV')
    table=json.loads((rope/'table.json').read_text())
    if table['status']!='pass' or table['config_sha256']!=receipt['config_sha256']:raise ValueError('Wrong verified RoPE constants')
    constants={}
    for row in table['files']:
        p=rope/row['file']
        if digest(p)!=row['sha256']:raise ValueError('RoPE table changed')
        constants[row['name']]=read(p,(CAP,D))
    qkv=read(source/observed['file'],(64,8,4,D))
    out.mkdir();records={}
    generator=torch.Generator().manual_seed(20261016)
    ax=torch.randn(1,14336,generator=generator).bfloat16()
    weight=torch.randn(128,14336,generator=generator).bfloat16()*0.02
    for seq in (64,1):
        root=out/('prefill' if seq==64 else 'decode');root.mkdir();plan=root/'plan';plan.mkdir();inputs=root/'inputs';inputs.mkdir()
        buffers=[];offset=0
        def allocation(name,shape,dtype=torch.bfloat16,value=None):
            nonlocal offset
            n=math.prod(shape)*torch.empty((),dtype=dtype).element_size();index=len(buffers)
            entry=dict(id=index,name=name,offset=offset,bytes=n,allocated=(n+1023)//512*512,
                shape=list(shape),dtype=str(dtype),required=value is not None)
            buffers.append(entry);offset+=entry['allocated']
            if value is not None:save(inputs/f'{index}.bin',value)
            return entry['offset']
        q=allocation('qkv',(seq,8,4,D),value=qkv if seq==64 else qkv[-1:])
        k=allocation('k',(8,CAP,D));v=allocation('v',(8,CAP,D))
        cos=allocation('cos',(CAP,D),value=constants['cos']);sin=allocation('sin',(CAP,D),value=constants['sin'])
        gamma=allocation('gamma',(D,),value=torch.ones(D,dtype=torch.bfloat16))
        context=allocation('context',(seq,16,D));work=allocation('attention_workspace',(G,2<<20),torch.uint8)
        a=allocation('bf16_large_k_input',(1,14336),value=ax)
        b=allocation('bf16_large_k_weight',(G*128,14336),value=weight.repeat(G,1))
        outputs=[allocation(f'bf16_output_{i}',(1,G*128)) for i in range(4 if seq==64 else 1)]
        def stage(operands,workspace,**kw):
            fields=dict(kind=0,epilogue=0,tasks=0,rows=0,columns=0,k=0,extent=0,width=0,group=0,
                kv_block=0,query_rows=0,row_stride=0,row_offset=0,batch_rows=0,variant=0,epsilon=0.0)
            fields.update(kw)
            return struct.pack('<17Q15If',*(operands+[ABSENT]*(16-len(operands))),workspace,*fields.values())
        attn_tasks=32 if seq==64 else 8
        stages=[stage([q,k,v,cos,sin,gamma,gamma,context],work,kind=13,tasks=attn_tasks,extent=8,width=D,
            group=2,kv_block=CAP,query_rows=32 if seq==64 else 2,variant=4,epsilon=1e-6)]
        for c in outputs:stages.append(stage([a,b,ABSENT,c],ABSENT,tasks=G,rows=1,columns=G*128,k=14336,variant=1))
        tasks=[];queues=[[] for _ in range(3*G)]
        for group in range(G):
            for step in range(len(outputs)):
                for s,tile,mix in ((step+1,group,0),(0,step*G+group,1)):
                    index=len(tasks);ordinal=len(queues[group])+1
                    tasks.append((s,tile,group,ordinal,0,0,mix));queues[group].append(index)
                    if mix:queues[G+2*group].append(index)
        stride=1+max(map(len,queues));flat=[];offsets=[0]
        for queue in queues:flat+=queue;offsets.append(len(flat))
        (plan/'metadata.bin').write_bytes(struct.pack('<12I3Q',4,G,3*G,len(tasks),len(stages),0,len(flat),stride,1,seq,CAP,1,offset,131072,2000000000))
        (plan/'stages.bin').write_bytes(b''.join(stages));(plan/'tasks.bin').write_bytes(b''.join(struct.pack('<7I',*t) for t in tasks))
        (plan/'waits.bin').write_bytes(b'');(plan/'queue.bin').write_bytes(struct.pack('<'+'I'*len(flat),*flat))
        (plan/'offsets.bin').write_bytes(struct.pack('<'+'I'*len(offsets),*offsets))
        (plan/'allocations.bin').write_bytes(b''.join(struct.pack('<3Q2I',r['offset'],r['bytes'],r['allocated'],int(r['required']),0) for r in buffers))
        records[str(seq)]=dict(buffers=buffers,task_count=len(tasks),stages=len(stages),queue_lengths=list(map(len,queues)),epoch_stride=stride,
            plan_sha256={p.name:digest(p) for p in plan.iterdir()},input_sha256={p.name:digest(p) for p in inputs.iterdir()})
    identity=dict(schema='tilemega.as2.mix_attention_probe.v1',tilemega_pipeline=False,
        scope='Body/protocol test only; explicitly constructed probe queues',model='qwen3_0_6b',head_dim=D,gqa=2,groups=G,
        source=str(source),source_sha256=digest(source/'reference.json'),rope=str(rope),rope_sha256=digest(rope/'table.json'),
        phases=records,source_qkv_sha256=observed['sha256'],dtype='bf16 inputs and output; full FP32 QK/softmax/PV',
        tolerance=dict(max_abs=0.0,bitwise_expected=True),device_forwards=dict(prefill=1,decode=4))
    (out/'probe.json').write_text(json.dumps(identity,indent=2)+'\n')


def reference(root,out):
    import torch_npu
    import torch.nn.functional as F
    torch.npu.set_device(0)
    if torch.npu.matmul.allow_hf32:raise ValueError('Require full FP32 arithmetic')
    record=validate(root);desc=record['phases']['64'];buf={v['name']:v for v in desc['buffers']}
    inputs=root/'prefill/inputs'
    def value(name):
        r=buf[name];return read(inputs/f"{r['id']}.bin",r['shape']).to('npu:0')
    out.mkdir();context_rows=[]
    with torch.inference_mode():
        packed=value('qkv');cos=value('cos');sin=value('sin');gamma=value('gamma')
        key=torch.zeros((1,8,CAP,D),dtype=torch.bfloat16,device='npu:0');val=torch.zeros_like(key)
        def rotate(x,positions):
            x=torch_npu.npu_rms_norm(x,gamma,epsilon=1e-6)[0]
            return x*cos[positions,None,None,:]+torch.cat((-x[...,D//2:],x[...,:D//2]),dim=-1)*sin[positions,None,None,:]
        def attention(qkv,positions):
            query=rotate(qkv[:,:,:2],positions).reshape(-1,16,D).permute(1,0,2)[None]
            kval=rotate(qkv[:,:,2:3],positions).squeeze(2).permute(1,0,2)
            vval=qkv[:,:,3].permute(1,0,2)
            key[0,:,positions,:]=kval;val[0,:,positions,:]=vval
            mask=torch.zeros((len(positions),CAP),dtype=torch.bfloat16,device='npu:0')
            mask.masked_fill_(torch.arange(CAP,device='npu:0')[None,:]>positions[:,None],torch.finfo(torch.bfloat16).min)
            result=F.scaled_dot_product_attention(query,key.repeat_interleave(2,dim=1),val.repeat_interleave(2,dim=1),attn_mask=mask)
            return result[0].permute(1,0,2).contiguous()
        result=attention(packed,torch.arange(64,device='npu:0'))
        save(out/'context_prefill.bin',result.cpu());save(out/'k_prefill.bin',key.cpu());save(out/'v_prefill.bin',val.cpu())
        for pos in range(64,68):context_rows.append(attention(packed[-1:],torch.tensor([pos],device='npu:0')).cpu())
        save(out/'context_decode_last.bin',context_rows[-1]);save(out/'k_decode.bin',key.cpu());save(out/'v_decode.bin',val.cpu())
        save(out/'context_decode_all.bin',torch.cat(context_rows))
        aa=value('bf16_large_k_input');bb=value('bf16_large_k_weight')
        save(out/'bf16_large_k.bin',F.linear(aa,bb).cpu())
    (out/'reference.json').write_text(json.dumps(dict(probe_identity=digest(root/'probe.json'),
        implementation='independent torch_npu fused RMSNorm and eager SDPA; exact BF16 RoPE cast points',
        hf32=False,files={p.name:digest(p) for p in out.iterdir()}),indent=2)+'\n')


def seed_decode(root,reference_dir):
    rec=validate(root)
    if json.loads((reference_dir/'reference.json').read_text())['probe_identity']!=digest(root/'probe.json'):
        raise ValueError('Independent reference/probe identity mismatch')
    seeded={}
    for name in ('k','v'):
        b=next(v for v in rec['phases']['1']['buffers'] if v['name']==name)
        target=root/'decode/inputs'/f"{b['id']}.bin"
        if target.exists():raise FileExistsError(target)
        target.write_bytes((reference_dir/f'{name}_prefill.bin').read_bytes())
        seeded[target.name]=digest(target)
    (root/'decode_seed.json').write_text(json.dumps(dict(
        scope='independent prefill cache; probe only, not complete generation',
        reference_sha256=digest(reference_dir/'reference.json'),files=seeded),indent=2)+'\n')


def validate(root):
    rec=json.loads((root/'probe.json').read_text())
    for seq,phase in rec['phases'].items():
        folder=root/('prefill' if seq=='64' else 'decode')
        for relative,identity in (('plan',phase['plan_sha256']),('inputs',phase['input_sha256'])):
            if any(digest(folder/relative/name)!=sha for name,sha in identity.items()):raise ValueError('Probe plan/input changed')
    return rec


def compare(root,reference_dir,native,seq,out):
    rec=validate(root);phase=rec['phases'][str(seq)];rows=[]
    identity=json.loads((reference_dir/'reference.json').read_text())
    if identity['probe_identity']!=digest(root/'probe.json') or any(digest(reference_dir/name)!=sha for name,sha in identity['files'].items()):
        raise ValueError('Independent reference identity changed')
    if seq==1:
        seeded=json.loads((root/'decode_seed.json').read_text())
        if seeded['reference_sha256']!=digest(reference_dir/'reference.json') or any(digest(root/'decode/inputs'/n)!=sha for n,sha in seeded['files'].items()):
            raise ValueError('Initial diagnostic KV changed')
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text())
    for b in phase['buffers']:
        name=b['name'];expected=None
        if name=='context':expected=reference_dir/('context_prefill.bin' if seq==64 else 'context_decode_last.bin')
        elif name in ('k','v'):expected=reference_dir/(name+('_prefill.bin' if seq==64 else '_decode.bin'))
        elif name.startswith('bf16_output'):expected=reference_dir/'bf16_large_k.bin'
        if expected is None:continue
        actual=read(native/f"{b['id']}.bin",b['shape']);gold=read(expected,b['shape']);diff=(actual.float()-gold.float()).abs()
        tolerance=policy['body']['gemm' if name.startswith('bf16_output') else 'attention' if name=='context' else 'rope']
        failed=int((diff>tolerance['atol']+tolerance['rtol']*gold.float().abs()).sum())
        nrmse=float(diff.square().mean().sqrt()/gold.float().square().mean().sqrt().clamp_min(1e-12))
        nonfinite=int((~torch.isfinite(actual)).sum())
        rows.append(dict(name=name,elements=gold.numel(),bitwise_different=int((actual!=gold).sum()),max_abs=float(diff.max()),
            nrmse=nrmse,nonfinite=nonfinite,failed_elements=failed,
            frozen_body_policy_pass=not failed and not nonfinite and nrmse<=tolerance.get('nrmse_max',float('inf')),
            native_sha256=digest(native/f"{b['id']}.bin")))
    trace=torch.from_file(str(native/'trace.bin'),shared=False,dtype=torch.int64,size=phase['task_count']*8).reshape(-1,8)
    taskdata=(root/('prefill' if seq==64 else 'decode')/'plan/tasks.bin').read_bytes()
    expected_tasks=list(struct.iter_unpack('<7I',taskdata));epoch=0 if seq==64 else 3
    trace_ok=all(tuple(int(v) for v in trace[i,:3])==(t[2],t[0],t[1]) and int(trace[i,5])==epoch and trace[i,4]>=trace[i,3] for i,t in enumerate(expected_tasks))
    launches=[json.loads(v) for v in (native/'launches.jsonl').read_text().splitlines()]
    passes=trace_ok and len(launches)==(1 if seq==64 else 4) and not any(r['bitwise_different'] or r['nonfinite'] for r in rows)
    result=dict(schema='tilemega.as2.mix_attention_probe_numeric.v1',status='pass' if passes else 'fail',tilemega_pipeline=False,
        scope='exact Body/protocol diagnostic; not whole-model correctness or performance',seq=seq,groups=G,rows=rows,
        frozen_body_policy_pass=all(r['frozen_body_policy_pass'] for r in rows),
        policy_sha256=digest(REPO/'configs/ascend/as2_correctness_policy.json'),
        task_trace_pass=trace_ok,actual_launches=len(launches),owner_tasks=phase['task_count'],source_probe=str(root))
    out.write_text(json.dumps(result,indent=2)+'\n')
    return result


def summarize(root,reference_dir,out):
    results=[];stable=True
    for seq,label in ((64,'prefill'),(1,'decode')):
        baseline=None
        for trial in range(1,6):
            native=root/f'{label}_p{trial}b';job=RUNS/f'jobs/mix_fp32_{label}_p{trial}_v4b/job.json'
            receipt=json.loads(job.read_text())
            if receipt['status']!='passed' or receipt['physical_device_id']!=1:raise ValueError('Missing completed single-card process')
            result=compare(root,reference_dir,native,seq,root/f'{label}_p{trial}_numeric.json')
            hashes={p.name:digest(p) for p in native.glob('*.bin') if p.name!='trace.bin'}
            if baseline is None:baseline=hashes
            equal=hashes==baseline;stable=stable and equal and result['task_trace_pass']
            results.append(dict(seq=seq,trial=trial,pid=receipt['pid'],elapsed_s=receipt['elapsed_s'],
                same_algorithm_payload_control_bitwise=equal,strict_reference_status=result['status'],
                frozen_body_policy_pass=result['frozen_body_policy_pass'],
                bitwise_differences={r['name']:r['bitwise_different'] for r in result['rows']},
                evidence=str(root/f'{label}_p{trial}_numeric.json')))
    if len({r['pid'] for r in results})!=10:raise ValueError('Require ten genuinely fresh phase processes')
    summary=dict(schema='tilemega.as2.mix_attention_c3.v1',tilemega_pipeline=False,scope='MIX Body/protocol only',
        protocol_repeatability='pass' if stable else 'fail',processes_per_phase=5,phase_count=2,
        frozen_body_policy_pass=all(r['frozen_body_policy_pass'] for r in results),
        strict_reference_status='pass' if all(r['strict_reference_status']=='pass' for r in results) else 'fail',rows=results)
    out.write_text(json.dumps(summary,indent=2)+'\n');return summary


def norm_diagnose(root,out):
    import torch_npu
    torch.npu.set_device(0);validate(root)
    qkv=read(root/'prefill/inputs/0.bin',(64,8,4,D))
    x=qkv[37,:,:2].reshape(16,D).contiguous();gamma=torch.ones(D,dtype=torch.bfloat16)
    out.mkdir();save(out/'input.bin',x);save(out/'rhs.bin',gamma)
    with torch.inference_mode():
        outputs=torch_npu.npu_rms_norm(x.to('npu:0'),gamma.to('npu:0'),epsilon=1e-6)
        expected=outputs[0].cpu();save(out/'expected.bin',expected)
        if len(outputs)>1:save(out/'rstd_npu.bin',outputs[1].cpu().float())
    rstd=(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt()
    save(out/'expected_cpu.bin',(x.float()*rstd).bfloat16())
    save(out/'rstd_cpu.bin',rstd)
    (out/'identity.json').write_text(json.dumps(dict(probe_identity=digest(root/'probe.json'),
        scope='same actual query operands at prefill position37, including divergent head6',rows=16,width=D,
        reference='independent torch_npu.npu_rms_norm',files={p.name:digest(p) for p in out.iterdir()}),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source',type=Path,required=True);prep.add_argument('--rope',type=Path,required=True)
    for command in ('reference','seed-decode','compare','summarize-c3','norm-diagnose'):sub.add_parser(command)
    for parser in sub.choices.values():
        parser.add_argument('--root',type=Path,required=True)
        if parser.prog.split()[-1] in ('reference','compare','summarize-c3','norm-diagnose'):parser.add_argument('--out',type=Path,required=True)
        if parser.prog.split()[-1] in ('seed-decode','compare','summarize-c3'):parser.add_argument('--reference',type=Path,required=True)
    sub.choices['compare'].add_argument('--native',type=Path,required=True);sub.choices['compare'].add_argument('--seq',type=int,choices=(1,64),required=True)
    a=p.parse_args();torch.set_num_threads(1);root=controlled(a.root,[RUNS])
    if hasattr(a,'out'):
        a.out=controlled(a.out,[RUNS])
        if a.out.exists():raise FileExistsError(a.out)
    if a.command=='prepare':prepare(a.source,a.rope,root)
    elif a.command=='reference':reference(root,a.out)
    elif a.command=='norm-diagnose':norm_diagnose(root,a.out)
    elif a.command=='seed-decode':seed_decode(root,controlled(a.reference,[RUNS]))
    else:
        result=summarize(root,controlled(a.reference,[RUNS]),a.out) if a.command=='summarize-c3' else compare(root,controlled(a.reference,[RUNS]),controlled(a.native,[RUNS]),a.seq,a.out)
        print(json.dumps(result))
        if result.get('status')=='fail' or result.get('protocol_repeatability')=='fail':raise SystemExit(1)


if __name__=='__main__':main()
