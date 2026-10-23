"""Independent HF decode with an explicit zero-KV diagnostic initial state."""
from __future__ import annotations
import argparse
import json
import struct
from pathlib import Path
import torch
import torch_npu
from transformers.cache_utils import StaticCache
from tilemega.bench.as2_eager import load_model, digest
from tilemega.bench.as2_models import RUNS, controlled, DENSE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=DENSE,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('npu:0','cpu'),default='npu:0')
    p.add_argument('--attention',choices=('sdpa','eager'),default='sdpa')
    p.add_argument('--norm',choices=('hf','fused'),default='hf')
    p.add_argument('--save-norm-input-layers',type=int,nargs='*',default=[],help='Diagnostic pre-Norm2 operands, stored separately from compared model outputs')
    p.add_argument('--capture-layers',type=int,nargs='+',help='Bound intermediate storage to these layers; still execute the entire model')
    a=p.parse_args();plan=controlled(a.plan,[RUNS]);inputs=controlled(a.inputs,[RUNS]);out=controlled(a.out,[RUNS])
    receipt=json.loads((inputs/'inputs.json').read_text());buffers=json.loads((plan/'buffers.json').read_text())['buffers']
    if receipt['model_id']!=a.model or receipt['initial_kv'] not in ('all_zero_diagnostic_state_not_prefill','empty_cache_for_prefill') or digest(plan/'metadata.bin')!=receipt['metadata_sha256']:
        raise ValueError('Diagnostic input/plan identity mismatch')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if a.device=='npu:0':torch.npu.set_device(0)
    model,ids,manifest,weight_hash=load_model(a.model,a.attention,a.norm,device=a.device)
    if weight_hash!=receipt['weight_manifest_sha256'] or manifest['config_sha256']!=receipt['config_sha256']:
        raise ValueError('Paired weight/config identity mismatch')
    batch,past,seq=receipt['batch'],receipt['past'],receipt.get('seq',1);cfg=manifest['config'];saved={};projections={};norm_inputs={}
    selected=set(range(cfg['num_hidden_layers'])) if a.capture_layers is None else set(a.capture_layers)
    if not selected or any(i<0 or i>=cfg['num_hidden_layers'] for i in selected):raise ValueError('Invalid diagnostic capture layer')
    if any(i<0 or i>=cfg['num_hidden_layers'] for i in a.save_norm_input_layers):raise ValueError('Invalid diagnostic norm layer')
    def save(name):
        def hook(module,args,result):saved[name]=result.detach().clone()
        return hook
    def before(name):
        def hook(module,args):saved[name]=args[0].detach().clone()
        return hook
    for i,layer in enumerate(model.model.layers):
        if i not in selected:continue
        layer.input_layernorm.register_forward_hook(save(f'l{i}.norm1'))
        layer.post_attention_layernorm.register_forward_hook(save(f'l{i}.norm2'))
        if i in a.save_norm_input_layers:
            def norm_input(module,args,i=i):norm_inputs[i]=args[0].detach().clone()
            layer.post_attention_layernorm.register_forward_pre_hook(norm_input)
        layer.self_attn.o_proj.register_forward_pre_hook(before(f'l{i}.context'))
        layer.mlp.down_proj.register_forward_pre_hook(before(f'l{i}.act'))
        layer.self_attn.o_proj.register_forward_hook(save(f'l{i}.o.raw'))
        layer.mlp.down_proj.register_forward_hook(save(f'l{i}.down.raw'))
        for kind in ('q','k','v'):
            def capture(module,args,result,i=i,kind=kind):projections[i,kind]=result.detach().clone()
            getattr(layer.self_attn,kind+'_proj').register_forward_hook(capture)
    model.model.norm.register_forward_hook(save('final.normalized'))
    cache=StaticCache(config=model.config,max_batch_size=batch,max_cache_len=1088,
        device=torch.device(a.device),dtype=torch.bfloat16)
    cache.reset()
    token=(ids[:batch,-1:] if seq==1 else ids[:batch]).to(a.device)
    position=torch.arange(past,past+seq,dtype=torch.int64,device=a.device)
    mask=torch.zeros(batch,1088,dtype=torch.int64,device=a.device);mask[:,:past+seq]=1
    with torch.inference_mode():
        result=model(input_ids=token,attention_mask=mask,position_ids=position.reshape(1,seq).expand(batch,-1),
            cache_position=position,past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1)
        logits=result.logits[:,0].bfloat16();chosen=logits.argmax(-1).to(torch.int32)
        g,d=cfg['num_key_value_heads'],cfg['head_dim'];ratio=cfg['num_attention_heads']//g
        for i in sorted(selected):
            q,k,v=(projections[i,t] for t in ('q','k','v'))
            saved[f'l{i}.qkv']=torch.cat((q.reshape(batch*seq,g,ratio,d),k.reshape(batch*seq,g,1,d),v.reshape(batch*seq,g,1,d)),2).reshape(batch*seq,-1)
        saved['final.normalized']=saved['final.normalized'][:,-1,:]
    if a.device=='npu:0':torch.npu.synchronize()
    native_stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
    residual=[s for s in native_stages if s[17] in (0,6) and s[18]==1]
    if len(residual)!=2*cfg['num_hidden_layers']:raise ValueError('Residual projection layer identity mismatch')
    raw_names={s[5] if s[17]==6 else s[16]:f'l{i//2}.{("o" if i%2==0 else "down")}.raw' for i,s in enumerate(residual)}
    files=[]
    for b in buffers:
        value=saved.get(b['name'])
        if b['offset_bytes'] in raw_names:value=saved.get(raw_names[b['offset_bytes']])
        if b['name'].startswith('native.gemm.raw.'):
            # Logical names remain stable when split projection inserts stages.
            stages=struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes())
            if any(s[17]==0 and s[18]==3 and s[16]==b['offset_bytes'] for s in stages):value=logits
        if value is None:continue
        cpu=value.cpu().contiguous().bfloat16()
        if cpu.numel()*2!=b['bytes']:raise ValueError('Reference/native buffer shape mismatch: '+b['name'])
        path=out/(str(b['id'])+'.bin');path.write_bytes(cpu.view(torch.uint8).numpy().tobytes())
        files.append(dict(id=b['id'],name=b['name'],file=path.name,sha256=digest(path),elements=cpu.numel()))
    (out/'token.bin').write_bytes(chosen.cpu().numpy().tobytes())
    attention_stages=[i for i,s in enumerate(native_stages) if s[17]==13]
    if len(attention_stages)!=cfg['num_hidden_layers']:raise ValueError('KV layer table mismatch')
    for layer,stage in enumerate(attention_stages):
        for name,storage in (('k',cache.key_cache),('v',cache.value_cache)):
            value=storage[layer][:,:,past+seq-1,:].cpu().contiguous()
            path=out/f'kv_{name}_s{stage}.bin';path.write_bytes(value.view(torch.uint8).numpy().tobytes())
            files.append(dict(file=path.name,name=f'layer{layer}.kv_{name}',elements=value.numel(),dtype='bf16',sha256=digest(path)))
    extra=[]
    for i,value in norm_inputs.items():
        cpu=value.cpu().contiguous().bfloat16();path=out/f'norm2_input_l{i}.bin'
        path.write_bytes(cpu.view(torch.uint8).numpy().tobytes())
        extra.append(dict(layer=i,name=f'l{i}.norm2.input',file=path.name,shape=list(cpu.shape),dtype='bf16',sha256=digest(path)))
    report=dict(schema='tilemega.as2.native_reference.v1',status='reference_executed_not_compared',
        model_id=a.model,batch=batch,past=past,seq=seq,decode_only=seq==1,initial_kv=receipt['initial_kv'],
        config_sha256=receipt['config_sha256'],weight_manifest_sha256=weight_hash,input_sha256=receipt['input_sha256'],
        implementation=f'pinned_transformers_{a.attention}_{a.norm}_norm',device=a.device,files=files,
        diagnostic_norm_inputs=extra,token_ids=chosen.cpu().tolist())
    report['captured_layers']=sorted(selected)
    report['scope']='complete model execution; selected intermediate captures, not full intermediate admission'
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))

if __name__=='__main__':main()
