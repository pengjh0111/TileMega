"""Check an actual generated forward MoE region without latency measurement.

Synthetic weights exercise the graph path independently of checkpoint I/O.
Real weights and captured hidden states are required for the single-layer gate.
The native internal-buffer accessor exists only in the validation artifact.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import torch
from tilemega.serving.plan import FORWARD, PlanLibrary
from tilemega.serving.weights import _packed_gpu, load_weights


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def leaf(recipe):
    while isinstance(recipe.get('source'),dict):
        recipe=recipe['source']
    return recipe


def router_output(bridge,gate_name):
    nodes={node['name']:node for node in bridge['nodes']}
    parameters={item['name']:item['target'] for item in bridge['signature']['inputs']
        if item['kind']=='PARAMETER'}
    layouts={'aten.t.default','aten.transpose.int','aten.permute.default',
        'aten.view.default','aten.reshape.default','aten.clone.default','aten.contiguous.default'}
    def parameter(name):
        seen=set()
        while name not in seen:
            seen.add(name)
            if name in parameters:return parameters[name]
            node=nodes[name]
            if node['target'] not in layouts or not node['inputs']:return None
            name=node['inputs'][0]
        return None
    matches=[node['name'] for node in bridge['nodes'] if
        node['target'] in ('aten.linear.default','aten.mm.default','aten.matmul.default') and
        len(node['inputs'])>=2 and parameter(node['inputs'][1])==gate_name]
    if len(matches)!=1:raise ValueError('region must have one router projection')
    return matches[0]


def internal(library,plan,name,shape,dtype,*,batch=1):
    index=next(i for i,b in enumerate(library.buffers) if b.name==name)
    accessor=library.lib.tm_dm_debug_buffer
    accessor.argtypes=[C.c_void_p,C.c_uint32,C.POINTER(C.c_uint64)]
    accessor.restype=C.c_void_p
    meta=(C.c_uint64*18)();pointer=accessor(plan.handle,index,meta)
    count=library.buffers[index].elements_constant+batch*library.buffers[index].elements_per_batch
    value=torch.empty(shape,device='cuda',dtype=dtype)
    if not pointer or count!=value.numel():
        raise ValueError('native routing buffer extent differs from the bound region')
    runtime=C.CDLL('/usr/local/cuda/lib64/libcudart.so')
    runtime.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
    runtime.cudaMemcpy.restype=C.c_int
    if runtime.cudaMemcpy(value.data_ptr(),pointer,value.numel()*value.element_size(),3):
        raise RuntimeError('native routing buffer copy failed')
    return value


@torch.inference_mode()
def check(library_path,bridge_path,checkpoint=None,hidden_path=None,sequence=None):
    torch.set_num_threads(4);torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False
    identity_path=Path(str(library_path)+'.identity.json')
    identity=json.loads(identity_path.read_text())
    if identity['binary_sha256']!=sha(library_path):
        raise ValueError('library identity mismatch')
    library=PlanLibrary(library_path)
    if library.info.phase!=FORWARD or library.info.capacity or library.info.batch_lo!=1:
        raise ValueError('MoE region requires a stateless batch-one forward plan')
    bridge=json.loads(Path(bridge_path).read_text())
    names={n['name']:n for n in bridge['nodes']}
    call=next(n for n in bridge['nodes'] if 'moe_experts' in n['target'])
    xname=next(i['name'] for i in bridge['signature']['inputs'] if i['kind']=='USER_INPUT')
    yname=bridge['signature']['outputs'][0]['name']
    topname=call['inputs'][1]
    recipes={b.name:json.loads(b.pack_json) for b in library.buffers if b.pack_json}
    experts=next(leaf(r) for r in recipes.values() if leaf(r)['kind']=='expert_stack')
    e,h,i,u=(int(experts[k]) for k in ('experts','hidden','intermediate','u'))
    t=library.info.seq;k=8
    # The custom-op input shape declares K; it need not be eight in fixtures.
    k=int(names[topname]['shape'][-1])
    prefix=experts['prefix'];layer=int(prefix.split('.layers.')[1].split('.')[0])
    norm_name=prefix.replace('mlp.experts.','post_attention_layernorm.weight')
    gate_name=prefix.replace('experts.','gate.weight')
    logit_name=router_output(bridge,gate_name)
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock,Qwen3MoeRMSNorm
    config=Qwen3MoeConfig(hidden_size=h,moe_intermediate_size=i,num_experts=e,
        num_experts_per_tok=k,norm_topk_prob=True,rms_norm_eps=1e-6)
    config._experts_implementation='grouped_mm'
    if checkpoint is not None:
        if hidden_path is None:
            raise ValueError('real checkpoint checks require captured HF hidden states')
        from safetensors.torch import load_file
        from tilemega.moe.routing_profile import IndexedCheckpoint,load_hf_layer
        config=Qwen3MoeConfig.from_pretrained(checkpoint,local_files_only=True)
        config._experts_implementation='grouped_mm'
        if (config.num_experts,config.hidden_size,config.moe_intermediate_size)!=(e,h,i):
            raise ValueError('checkpoint geometry differs from the generated region')
        source=IndexedCheckpoint(checkpoint)
        block=load_hf_layer(source,config,layer,'cuda')
        mlp,norm=block.mlp,block.post_attention_layernorm
        tensors={}
        captured=load_file(str(hidden_path))
        candidates=sorted(key for key in captured if key.endswith('.hidden'))
        key=sequence+'.hidden' if sequence else candidates[0]
        x=captured[key][:t].to('cuda')
        provenance=dict(checkpoint=source.identity(),hidden_sha256=sha(hidden_path),sequence=key)
    else:
        mlp=Qwen3MoeSparseMoeBlock(config).to(device='cuda',dtype=torch.bfloat16)
        norm=Qwen3MoeRMSNorm(h,config.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
        norm.weight.copy_(1+torch.randn_like(norm.weight)*.02)
        # Keep the expert contribution visible after BF16 residual rounding.
        for parameter in mlp.parameters():parameter.copy_(torch.randn_like(parameter)*(.7/h**.5))
        sources={norm_name:norm.weight,gate_name:mlp.gate.weight}
        for expert in range(e):
            gate,up=mlp.experts.gate_up_proj[expert].chunk(2,0)
            for part,value in [('gate',gate),('up',up),('down',mlp.experts.down_proj[expert])]:
                sources[prefix+f'{expert}.{part}_proj.weight']=value
        tensors={name:_packed_gpu(recipe,sources.__getitem__).contiguous() for name,recipe in recipes.items()}
        x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
        provenance=dict(seed=20261009,synthetic=True)
    if x.shape!=(t,h) or x.dtype!=torch.bfloat16:
        raise ValueError('captured input differs from the bound region')
    normalized=norm(x);hf_logits,hf_weights,hf_indices=mlp.gate(normalized)
    reference=(x+mlp(normalized.unsqueeze(0)).squeeze(0)).to(torch.bfloat16)
    if checkpoint is not None:
        # References and packed native weights need not coexist on the GPU.
        del normalized,mlp,norm,block
        torch.cuda.empty_cache()
        tensors=load_weights(checkpoint,library,device='cuda')
    contribution=(reference.float()-x.float()).abs()
    if checkpoint is None and torch.all(contribution<=1.6e-2+1.6e-2*reference.float().abs()):
        raise ValueError('synthetic expert contribution is too small to detect a bypass')
    output=torch.empty_like(x);tensors.update({xname:x,yname:output})
    cases=[];first=None
    with library.create(1,{name:value.data_ptr() for name,value in tensors.items()},0) as plan:
        plan.set_steps([0])
        for epoch in range(3):
            for mode in (1,2):
                output.fill_(float('nan'));plan.launch(0,mode,torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                actual_idx=internal(library,plan,topname,(t,k),torch.int32).long()
                actual_logits=internal(library,plan,logit_name,(t,e),torch.bfloat16)
                same=actual_idx.sort(-1).values.eq(hf_indices.sort(-1).values).all(-1)
                intersection=(actual_idx[:,:,None]==hf_indices[:,None,:]).any(-1).sum(-1)
                error=(output.float()-reference.float()).abs()
                output_ok=bool(torch.all((error<=1.6e-2+1.6e-2*reference.float().abs())[same]))
                boundary=hf_logits.gather(1,hf_indices).float().min(-1).values
                rounded_boundary=boundary.to(torch.bfloat16)
                ulp=torch.maximum(*[(torch.nextafter(rounded_boundary,
                    torch.full_like(rounded_boundary,direction)).float()-boundary).abs()
                    for direction in (float('inf'),float('-inf'))])
                changed=(actual_idx[:,:,None]!=hf_indices[:,None,:]).all(-1)
                added_gap=(hf_logits.float().gather(1,actual_idx)-boundary[:,None]).abs()
                # Both entering and departing experts must lie at the HF boundary.
                removed=(hf_indices[:,:,None]!=actual_idx[:,None,:]).all(-1)
                removed_gap=(hf_logits.float().gather(1,hf_indices)-boundary[:,None]).abs()
                routing_ok=bool(torch.all((added_gap<=2*ulp[:,None])|~changed) and
                    torch.all((removed_gap<=2*ulp[:,None])|~removed))
                bits=first is None or torch.equal(first,output)
                if first is None:first=output.clone()
                case=dict(epoch=epoch,mode=mode,passed=output_ok and routing_ok and bits,
                    max_error=error.max().item(),set_equal_fraction=same.float().mean().item(),
                    routing_intersection_fraction=intersection.float().mean().item()/k,
                    router_max_error=(actual_logits.float()-hf_logits.float()).abs().max().item(),
                    same_binary_bits=bits)
                cases.append(case)
    return dict(evidence='verified',passed=all(c['passed'] for c in cases),
        scope='real-weight MoE region numerical gate' if checkpoint else 'synthetic generated MoE region; no real-weight gate',
        artifact_id=identity['artifact_id'],bridge_sha256=sha(bridge_path),
        provenance=provenance,expert_contribution_max=contribution.max().item(),cases=cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--bridge',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--hidden',type=Path)
    parser.add_argument('--sequence');args=parser.parse_args()
    result=check(args.library,args.bridge,args.checkpoint,args.hidden,args.sequence)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('evidence','passed','scope','artifact_id','cases')}),flush=True)
    raise SystemExit(0 if result['passed'] else 1)
