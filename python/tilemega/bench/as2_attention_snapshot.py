"""Replay a saved real attention operand; diagnostic queues never admit C-0."""
import argparse
import json
import math
import struct
from pathlib import Path
import torch
from tilemega.bench.as2_models import RUNS, WORK, controlled
from tilemega.bench.as2_native import digest

ABSENT=(1<<64)-1
G,D,CAP=8,128,1088


def read(path,shape,dtype=torch.bfloat16):
    n=math.prod(shape)
    if path.stat().st_size!=n*torch.empty((),dtype=dtype).element_size():raise ValueError('Tensor extent mismatch')
    return torch.from_file(str(path),shared=False,size=n,dtype=dtype).reshape(shape)


def save(path,value):
    path.write_bytes(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())


def validate(case):
    r=json.loads((case/'case.json').read_text())
    for folder,rows in [('plan',r['plan_sha256']),('inputs',r['input_sha256'])]:
        if any(digest(case/folder/n)!=sha for n,sha in rows.items()):raise ValueError('Probe identity changed')
    if digest(case/'expected_context.bin')!=r['expected_context_sha256']:raise ValueError('Observed output changed')
    return r


def prepare(a):
    binding=a.binding;source=a.reference;capture=source/'last_forward'
    identity=json.loads((binding/'request.json').read_text());r=json.loads((source/'result.json').read_text())
    cap=json.loads((capture/'capture.json').read_text());files={v['name']:v for v in cap['files']}
    for name in ('model_id','batch','config_sha256','weight_manifest_sha256','input_sha256'):
        if identity[name]!=r[name]:raise ValueError('Paired identity mismatch: '+name)
    if r['model_id']!='qwen3_0_6b' or r['batch']!=1 or not r.get('diagnostic_capture_last_forward') or r['norm']!='fused' or r['attention']!='sdpa' or cap['seq']!=1:
        raise ValueError('Require an independently captured Qwen B1 SDPA/fused decode operand')
    if any(digest(capture/v['file'])!=v['sha256'] for v in files.values()):raise ValueError('Capture changed')
    original=list(struct.iter_unpack('<17Q16I',(binding/'decode/stages.bin').read_bytes()))
    stages=[v for v in original if v[17]==13]
    if not 0<=a.layer<len(stages):raise ValueError('Invalid layer index')
    s=list(stages[a.layer])
    if (s[19],s[23],s[24],s[25],s[26],s[27],s[31])!=(8,8,128,2,1088,2,5) or s[10:16]!=[ABSENT]*6:
        raise ValueError('Unvalidated stage geometry/operand domain')
    if a.softmax_fold:s[31]=6
    weightroot=WORK/'weights'/r['model_id'];manifest=json.loads((weightroot/'manifest.json').read_text())
    if digest(weightroot/'manifest.json')!=r['weight_manifest_sha256']:raise ValueError('Weight identity mismatch')
    weights={v['name']:v for v in manifest['tensors']};past=cap['past']
    out=a.out;out.mkdir();(out/'plan').mkdir();(out/'inputs').mkdir();buffers=[];offset=0
    def allocation(name,shape,dtype=torch.bfloat16,value=None):
        nonlocal offset
        size=math.prod(shape)*torch.empty((),dtype=dtype).element_size();index=len(buffers)
        row=dict(id=index,name=name,offset=offset,bytes=size,allocated=(size+1023)//512*512,shape=list(shape),dtype=str(dtype),required=value is not None)
        buffers.append(row);offset+=row['allocated']
        if value is not None:save(out/'inputs'/f'{index}.bin',value)
        return row['offset']
    def observed(name):
        v=files[f'l{a.layer}.{name}'];return read(capture/v['file'],v['shape'])
    q=allocation('qkv',(8,4,D),value=observed('qkv').reshape(8,4,D))
    kv=[]
    for side in ('k','v'):
        prior=torch.zeros((8,CAP,D),dtype=torch.bfloat16)
        prior[:,:past]=observed('kv_'+side)[0,:,:past]
        kv.append(allocation(side,(8,CAP,D),value=prior))
    constants=[]
    for name in ('cos','sin'):
        row=next(v for v in identity['input_buffers'] if v['name']=='serving.rope_'+name)
        file=binding/'inputs'/row['file']
        if digest(file)!=row['sha256']:raise ValueError('RoPE input changed')
        constants.append(allocation(name,(CAP,D),value=read(file,(CAP,D))))
    gamma=[]
    for name in ('q','k'):
        row=weights[f'model.layers.{a.layer}.self_attn.{name}_norm.weight'];file=weightroot/row['file']
        if digest(file)!=row['sha256']:raise ValueError('QK scale changed')
        gamma.append(allocation(name+'_gamma',(D,),value=read(file,(D,))))
    context=allocation('context',(16,D));workspace=allocation('workspace',(G,2<<20),torch.uint8)
    original_buffers={b['offset_bytes']:b for b in identity['buffers']}
    for i in (8,9):
        if s[i]!=ABSENT:
            auxiliary=original_buffers[s[i]]
            if auxiliary['required_input'] or auxiliary['bytes']>1<<20:raise ValueError('Unvalidated auxiliary workspace')
            s[i]=allocation(auxiliary['name'],(auxiliary['bytes'],),torch.uint8)
    s[:8]=[q,*kv,*constants,*gamma,context];s[16]=workspace
    tasks=[(0,g,g,1,0,0,1) for g in range(G)];queues=[[g] for g in range(G)]
    for g in range(G):queues.extend(([g],[]))
    flat=[];offsets=[0]
    for queue in queues:flat+=queue;offsets.append(len(flat))
    plan=out/'plan';(plan/'metadata.bin').write_bytes(struct.pack('<12I3Q',4,G,3*G,8,1,0,16,2,1,1,CAP,1,offset,131072,2000000000))
    (plan/'stages.bin').write_bytes(struct.pack('<17Q16I',*s))
    (plan/'tasks.bin').write_bytes(b''.join(struct.pack('<7I',*t) for t in tasks));(plan/'waits.bin').write_bytes(b'')
    (plan/'queue.bin').write_bytes(struct.pack('<16I',*flat));(plan/'offsets.bin').write_bytes(struct.pack('<25I',*offsets))
    (plan/'allocations.bin').write_bytes(b''.join(struct.pack('<3Q2I',v['offset'],v['bytes'],v['allocated'],int(v['required']),0) for v in buffers))
    save(out/'expected_context.bin',observed('context').reshape(16,D))
    result=dict(schema='tilemega.as2.observed_attention.v1',tilemega_pipeline=False,
        scope='isolated Body replay with diagnostic queues; not a model or performance admission',
        model_id=r['model_id'],layer=a.layer,past=past,source_reference=str(source),source_binding=str(binding),
        reference_sha256=digest(source/'result.json'),capture_sha256=digest(capture/'capture.json'),
        binding_sha256=digest(binding/'request.json'),original_stage_sha256=digest(binding/'decode/stages.bin'),
        config_sha256=r['config_sha256'],weight_manifest_sha256=r['weight_manifest_sha256'],
        epsilon=struct.unpack('<f',struct.pack('<I',s[32]))[0],softmax_fold=a.softmax_fold,
        buffers=buffers,arena_bytes=offset,expected_context_sha256=digest(out/'expected_context.bin'),
        plan_sha256={p.name:digest(p) for p in plan.iterdir()},input_sha256={p.name:digest(p) for p in (out/'inputs').iterdir()})
    (out/'case.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('buffers','plan_sha256','input_sha256')}))


def reference(a):
    import torch_npu
    import torch.nn.functional as F
    r=validate(a.case);buffers={v['name']:v for v in r['buffers']};past=r['past']
    torch.npu.set_device(0)
    if torch.npu.device_count()!=1 or torch.npu.matmul.allow_hf32:raise ValueError('Require single device and full FP32 matmul')
    def load(name):
        v=buffers[name];return read(a.case/'inputs'/f"{v['id']}.bin",v['shape']).npu()
    a.out.mkdir()
    with torch.inference_mode():
        qkv=load('qkv');cos=load('cos')[past];sin=load('sin')[past]
        def rotate(x,gamma):
            x=torch_npu.npu_rms_norm(x,gamma,r.get('epsilon',1e-6))[0]
            return x*cos+torch.cat((-x[...,64:],x[...,:64]),-1)*sin
        query=rotate(qkv[:,:2].reshape(1,1,16,D),load('q_gamma')).reshape(1,16,1,D)
        key=load('k');value=load('v');key[:,past]=rotate(qkv[:,2].reshape(1,1,8,D),load('k_gamma')).reshape(8,D)
        value[:,past]=qkv[:,3]
        expanded_key=key.repeat_interleave(2,0)[None];expanded_value=value.repeat_interleave(2,0)[None]
        mask=torch.zeros((1,1,1,CAP),dtype=torch.bfloat16,device='npu:0')
        mask[...,past+1:]=torch.finfo(torch.bfloat16).min
        sdpa=F.scaled_dot_product_attention(query,expanded_key,expanded_value,attn_mask=mask).reshape(16,D)
        scale=128**(-.25);qf=query.float()*scale;kf=expanded_key.float()*scale
        scores=qf@kf.transpose(-1,-2);masked=scores+mask.float();prob=masked.softmax(-1)
        maximum=masked.amax(-1,keepdim=True);exponent=(masked-maximum).exp();denominator=exponent.sum(-1,keepdim=True)
        explicit_prob=exponent/denominator
        pv=prob@expanded_value.float();manual=pv.bfloat16().reshape(16,D)
        outputs={'q_scaled':qf.reshape(8,2,D),'k_scaled':(key.float()*scale),'v_fp32':value.float(),
            'scores':scores.reshape(8,2,CAP),'probabilities':prob.reshape(8,2,CAP),'pv_fp32':pv.reshape(8,2,D),
            'context':manual,'sdpa_context':sdpa,'k_after':key,'v_after':value,
            'exp_fp32':exponent.reshape(8,2,CAP),'maximum_fp32':maximum.reshape(8,2),
            'sum_exp_fp32':denominator.reshape(8,2)}
        for name,tensor in outputs.items():save(a.out/(name+'.bin'),tensor)
        gold=read(a.case/'expected_context.bin',(16,D))
        checks=dict(manual_vs_observed=int((manual.cpu()!=gold).sum()),sdpa_vs_observed=int((sdpa.cpu()!=gold).sum()),manual_vs_sdpa=int((manual!=sdpa).sum().cpu()))
    record=dict(case_sha256=digest(a.case/'case.json'),tilemega_pipeline=False,scope='independent torch_npu same-input replay; no performance admission',
        status='pass' if not any(checks.values()) else 'fail',checks=checks,
        explicit_probability_different_from_fused=int((explicit_prob!=prob).sum().cpu()),
        hf32=False,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        files={p.name:digest(p) for p in a.out.iterdir()})
    (a.out/'reference.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    return 0 if record['status']=='pass' else 1


def compare(a):
    r=validate(a.case);ref=json.loads((a.reference/'reference.json').read_text())
    reused=False
    if ref['case_sha256']!=digest(a.case/'case.json'):
        if a.reference_case is None:raise ValueError('Reference belongs to a different case')
        prior=validate(a.reference_case)
        if ref['case_sha256']!=digest(a.reference_case/'case.json'):raise ValueError('Wrong reference source case')
        for name in ('model_id','layer','past','config_sha256','weight_manifest_sha256','input_sha256','expected_context_sha256'):
            if prior[name]!=r[name]:raise ValueError('Reference reuse changes operands: '+name)
        if struct.pack('<f',prior.get('epsilon',1e-6))!=struct.pack('<f',r.get('epsilon',1e-6)):
            raise ValueError('Reference reuse changes epsilon')
        reused=True
    if any(digest(a.reference/n)!=sha for n,sha in ref['files'].items()):raise ValueError('Independent replay identity changed')
    native=read(a.native/'attention_workspace.bin',(8,524288),torch.float32);rows=[]
    fields=[('q_scaled',0,2*D,(8,2,D)),('k_scaled',65536,CAP*D,(8,CAP,D)),('v_fp32',622592,CAP*D,(8,CAP,D)),
        ('scores',1179648,2*CAP,(8,2,CAP)),('probabilities',1474560,2*CAP,(8,2,CAP)),('pv_fp32',1769472,2*D,(8,2,D))]
    for name,offset,count,shape in fields:
        actual=native[:,offset//4:offset//4+count].reshape(shape);gold=read(a.reference/(name+'.bin'),shape,torch.float32)
        if name=='scores':actual=actual[...,:r['past']+1];gold=gold[...,:r['past']+1]
        different=(actual!=gold);diff=(actual-gold).abs()
        rows.append(dict(name=name,elements=actual.numel(),bitwise_different=int(different.sum()),max_abs=float(diff.max()),first_coordinates=different.nonzero()[:16].tolist()))
    extra=[]
    if json.loads((a.native/'result.json').read_text()).get('softmax_statistics'):
        for name,offset,count,shape in [('exp_fp32',1802240,2*CAP,(8,2,CAP))]:
            actual=native[:,offset//4:offset//4+count].reshape(shape);gold=read(a.reference/(name+'.bin'),shape,torch.float32)
            extra.append(dict(name=name,bitwise_different=int((actual!=gold).sum()),max_abs=float((actual-gold).abs().max())))
        for name,index in [('maximum_fp32',0),('sum_exp_fp32',8)]:
            actual=native[:,2080768//4:2080768//4+32].reshape(8,2,16)[:,:,index]
            gold=read(a.reference/(name+'.bin'),(8,2),torch.float32)
            extra.append(dict(name=name,bitwise_different=int((actual!=gold).sum()),actual=actual.tolist(),reference=gold.tolist()))
    context=next(v for v in r['buffers'] if v['name']=='context')
    actual=read(a.native/f"{context['id']}.bin",(16,D));gold=read(a.case/'expected_context.bin',(16,D))
    result=dict(case_sha256=digest(a.case/'case.json'),tilemega_pipeline=False,status='diagnostic_complete',reference_replay_status=ref['status'],
        independent_reference_reused_for_identical_operands=reused,
        rows=rows,softmax_diagnostics=extra,context_bitwise_different=int((actual!=gold).sum()),context_first_coordinates=(actual!=gold).nonzero()[:16].tolist())
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for name in ('binding','reference','out'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--layer',type=int,required=True)
    prep.add_argument('--softmax-fold',action='store_true')
    replay=sub.add_parser('reference')
    for name in ('case','out'):replay.add_argument('--'+name,type=Path,required=True)
    comp=sub.add_parser('compare')
    for name in ('case','native','reference','out'):comp.add_argument('--'+name,type=Path,required=True)
    comp.add_argument('--reference-case',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    for name in ('case','binding','native','reference','reference_case','out'):
        if getattr(a,name,None) is not None:setattr(a,name,controlled(getattr(a,name),[RUNS]))
    if a.out.exists():raise FileExistsError(a.out)
    return {'prepare':prepare,'reference':reference,'compare':compare}[a.command](a)


if __name__=='__main__':raise SystemExit(main())
