"""Diagnose local Body errors separately from differences accumulated across layers."""
import argparse
import json
from pathlib import Path
import struct
import torch
import torch.nn.functional as F
from tilemega.bench.as2_models import REPO, RUNS, WORK, controlled
from tilemega.bench.as2_compare_native import metrics


def reference_prefill(binding,reference,out,layers,native=False,precision=0):
    receipt=json.loads((binding/'request.json').read_text())
    if native:
        ref=dict(seq=64,model_id=receipt['model_id'],weight_manifest_sha256=receipt['weight_manifest_sha256'],
            files=[dict(name=b['name'],file=f'{b["id"]}.bin') for b in receipt['buffers']])
    else:ref=json.loads((reference/'reference.json').read_text())
    if ref['seq']!=64 or ref['model_id']!=receipt['model_id'] or ref['weight_manifest_sha256']!=receipt['weight_manifest_sha256']:
        raise ValueError('Require the independent same-weight prefill reference')
    manifest=json.loads((WORK/'weights'/receipt['model_id']/'manifest.json').read_text());cfg=manifest['config']
    b=receipt['batch'];g=cfg['num_key_value_heads'];ratio=cfg['num_attention_heads']//g;d=cfg['head_dim'];h=cfg['hidden_size'];f=cfg['intermediate_size']
    files={r['name']:r for r in ref['files']};inputs={r['name']:r for r in receipt['input_buffers']}
    def read(path,shape):return torch.from_file(str(path),shared=False,size=__import__('math').prod(shape),dtype=torch.bfloat16).reshape(shape)
    def value(name,shape):return read(reference/files[name]['file'],shape)
    def weight(name,shape):return read(WORK/'weights'/receipt['model_id']/(name+'.bin'),shape)
    def norm(x,gamma):
        y=x.float();return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+cfg['rms_norm_eps'])).bfloat16()*gamma
    cos=read(binding/'inputs'/inputs['serving.rope_cos']['file'],(1088,d))[:64]
    sin=read(binding/'inputs'/inputs['serving.rope_sin']['file'],(1088,d))[:64]
    def rotate(x):return x*cos[None,None,None,:,:]+torch.cat((-x[...,d//2:],x[...,:d//2]),-1)*sin[None,None,None,:,:]
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text())['body'];rows=[]
    for i in layers:
        prefix=f'model.layers.{i}.';n1=value(f'l{i}.norm1',(b,64,h));qkv=value(f'l{i}.qkv',(b,64,g,ratio+2,d))
        q,k,v=[F.linear(n1,weight(prefix+'self_attn.'+side+'_proj.weight',(g*(ratio if side=='q' else 1)*d,h))) for side in ('q','k','v')]
        expected=torch.cat((q.reshape(b,64,g,ratio,d),k.reshape(b,64,g,1,d),v.reshape(b,64,g,1,d)),3)
        rows.append(dict(layer=i,operation='prefill_qkv_same_norm1_vs_cpu',**metrics(qkv,expected,policy['gemm'])))
        q=qkv[:,:,:,:ratio,:].permute(0,2,3,1,4);k=qkv[:,:,:,ratio,:].permute(0,2,1,3)[:,:,None,:,:]
        if cfg['model_type']=='qwen3':
            q=norm(q,weight(prefix+'self_attn.q_norm.weight',(d,)));k=norm(k,weight(prefix+'self_attn.k_norm.weight',(d,)))
        q,k=rotate(q),rotate(k);v=qkv[:,:,:,ratio+1,:].permute(0,2,1,3)[:,:,None,:,:]
        if precision==3:
            root=(d**-0.5)**0.5
            scores=torch.matmul(q.float()*root,(k.float()*root).transpose(-1,-2))
        else:scores=torch.matmul(q if precision==0 else q.float(),
            k.transpose(-1,-2) if precision==0 else k.float().transpose(-1,-2))*(d**-0.5)
        mask=torch.ones(64,64,dtype=torch.bool).triu(1);scores.masked_fill_(mask,torch.finfo(torch.bfloat16).min)
        if precision==3:scores=F.pad(scores,(0,1088-64),value=torch.finfo(torch.bfloat16).min)
        probability=F.softmax(scores.float(),dim=-1)[...,:64]
        if precision in (0,2):probability=probability.bfloat16()
        context=torch.matmul(probability,v if precision in (0,2) else v.float()).bfloat16().permute(0,3,1,2,4).reshape(b,64,g*ratio*d)
        rows.append(dict(layer=i,operation='prefill_attention_same_qkv_vs_cpu',**metrics(value(f'l{i}.context',context.shape),context,policy['attention'])))
        n2=value(f'l{i}.norm2',(b,64,h));gate=F.linear(n2,weight(prefix+'mlp.gate_proj.weight',(f,h)));up=F.linear(n2,weight(prefix+'mlp.up_proj.weight',(f,h)))
        rows.append(dict(layer=i,operation='prefill_swiglu_same_norm2_vs_cpu',**metrics(value(f'l{i}.act',(b,64,f)),F.silu(gate)*up,policy['swiglu'])))
    result=dict(schema='tilemega.as2.same_input_reference.v1',status='pass' if all(r['passed'] for r in rows) else 'fail',
        scope='native_prefill_local_operator_against_cpu_diagnostic' if native else 'independent_npu_eager_local_operator_against_cpu_diagnostic',
        attention_precision=precision,model_id=receipt['model_id'],rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','native','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--past',type=int,required=True)
    p.add_argument('--layers',type=int,nargs='+',default=[0,1,19,27])
    p.add_argument('--reference-prefill',action='store_true',help='Inspect independent eager operators using their own inputs')
    p.add_argument('--native-prefill',action='store_true',help='Inspect native prefill operators using their own inputs')
    p.add_argument('--attention-precision',type=int,choices=(0,1,2,3),default=0)
    a=p.parse_args();torch.set_num_threads(1)
    binding,native,out=(controlled(getattr(a,n),[RUNS]) for n in ('binding','native','out'))
    if out.exists():raise FileExistsError(out)
    if a.reference_prefill or a.native_prefill:return reference_prefill(binding,native,out,a.layers,a.native_prefill,a.attention_precision)
    receipt=json.loads((binding/'request.json').read_text());model=receipt['model_id']
    manifest=json.loads((WORK/'weights'/model/'manifest.json').read_text());cfg=manifest['config']
    if not 0<=a.past<1087 or any(not 0<=i<cfg['num_hidden_layers'] for i in a.layers):raise ValueError('Invalid diagnostic coordinates')
    buffers={b['name']:b for b in receipt['buffers']};inputs={b['name']:b for b in receipt['input_buffers']}
    def read(path,shape):
        return torch.from_file(str(path),shared=False,size=__import__('math').prod(shape),dtype=torch.bfloat16).reshape(shape)
    def value(name,shape):return read(native/f'{buffers[name]["id"]}.bin',shape)
    def packed(name,shape):return read(binding/'inputs'/inputs[name]['file'],shape)
    def weight(name,shape):return read(WORK/'weights'/model/(name+'.bin'),shape)
    batch=receipt['batch'];h=cfg['hidden_size'];g=cfg['num_key_value_heads'];ratio=cfg['num_attention_heads']//g;d=cfg['head_dim'];f=cfg['intermediate_size']
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text())['body']
    cos=packed('serving.rope_cos',(1088,d))[a.past];sin=packed('serving.rope_sin',(1088,d))[a.past]
    def norm(x,gamma):
        y=x.float();y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+cfg['rms_norm_eps'])
        return y.bfloat16()*gamma
    stages=list(struct.iter_unpack('<17Q16I',(binding/'decode/stages.bin').read_bytes()))
    attention=[i for i,s in enumerate(stages) if s[17]==13];rows=[]
    residual=[s for s in stages if s[17]==0 and s[18]==1]
    if len(residual)!=2*cfg['num_hidden_layers']:raise ValueError('Unexpected residual graph')
    by_offset={b['offset_bytes']:b for b in receipt['buffers']}
    def raw(stage):
        b=by_offset[stage[16]];return read(native/f'{b["id"]}.bin',(batch,h))
    token_path=native/'tokens.bin'
    tokens=torch.from_file(str(token_path),shared=False,size=batch*1088,dtype=torch.int32).reshape(batch,1088)
    hidden=F.embedding(tokens[:,a.past].long(),weight('model.embed_tokens.weight',(cfg['vocab_size'],h)))
    for i in range(cfg['num_hidden_layers']):
        if i in a.layers:
            expected=norm(hidden,weight(f'model.layers.{i}.input_layernorm.weight',(h,)))
            rows.append(dict(layer=i,operation='norm1_same_reconstructed_residual',
                **metrics(value(f'l{i}.norm1',(batch,h)),expected,policy['rmsnorm'])))
        hidden=hidden+raw(residual[2*i])
        if i in a.layers:
            expected=norm(hidden,weight(f'model.layers.{i}.post_attention_layernorm.weight',(h,)))
            rows.append(dict(layer=i,operation='norm2_same_reconstructed_residual',
                **metrics(value(f'l{i}.norm2',(batch,h)),expected,policy['rmsnorm'])))
        hidden=hidden+raw(residual[2*i+1])
    for i in a.layers:
        prefix=f'model.layers.{i}.';width=g*(ratio+2)*d
        n1=value(f'l{i}.norm1',(batch,h));actual=value(f'l{i}.qkv',(batch,width))
        w=packed(f'l{i}.qkv.weight',(width,h));expected=F.linear(n1,w)
        rows.append(dict(layer=i,operation='qkv_projection_same_norm1',**metrics(actual,expected,policy['gemm'])))
        key=actual.reshape(batch,g,ratio+2,d)[:,:,ratio,:]
        if cfg['model_type']=='qwen3':key=norm(key,weight(prefix+'self_attn.k_norm.weight',(d,)))
        rotated=torch.cat((-key[...,d//2:],key[...,:d//2]),-1)
        expected_key=key*cos+rotated*sin
        actual_key=read(native/f'kv_k_s{attention[i]}.bin',(batch,g,d))
        rows.append(dict(layer=i,operation='key_norm_rope_same_qkv',**metrics(actual_key,expected_key,policy['rope'])))
        history=native/f'kv_k_history_s{attention[i]}.bin'
        if history.exists():
            q=actual.reshape(batch,g,ratio+2,d)[:,:,:ratio,:]
            if cfg['model_type']=='qwen3':q=norm(q,weight(prefix+'self_attn.q_norm.weight',(d,)))
            q=q*cos+torch.cat((-q[...,d//2:],q[...,:d//2]),-1)*sin
            keys=read(history,(batch,g,a.past+1,d));values=read(native/f'kv_v_history_s{attention[i]}.bin',(batch,g,a.past+1,d))
            scores=(torch.matmul(q,keys.transpose(-1,-2))*(d**-0.5))
            probability=F.softmax(scores.float(),dim=-1).bfloat16()
            context=torch.matmul(probability,values).reshape(batch,-1)
            rows.append(dict(layer=i,operation='attention_same_query_and_full_kv',
                **metrics(value(f'l{i}.context',(batch,g*ratio*d)),context,policy['attention'])))
        n2=value(f'l{i}.norm2',(batch,h));act=value(f'l{i}.act',(batch,f))
        gate=F.linear(n2,weight(prefix+'mlp.gate_proj.weight',(f,h)))
        up=F.linear(n2,weight(prefix+'mlp.up_proj.weight',(f,h)))
        expected_act=F.silu(gate)*up
        rows.append(dict(layer=i,operation='gate_up_swiglu_same_norm2',**metrics(act,expected_act,policy['swiglu'])))
    result=dict(schema='tilemega.as2.same_input_diagnostic.v1',model_id=model,batch=batch,past=a.past,
        status='pass' if all(r['passed'] for r in rows) else 'fail',scope='CPU_same_operand_local_math_diagnostic_not_full_model_admission',rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
