"""Check a real MoE deployment's streaming weight load under a memory cap.

This validates packing and allocation, not generated-model execution. Both
phase descriptors deliberately request the same weight recipes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tilemega.moe.checkpoints import index_check
from tilemega.serving.plan import Buffer
from tilemega.serving.weights import Checkpoint, load_weights


def unpack_pages(value, n, k, tn, tk):
    index=torch.arange(tn*tk)
    row,col=index//tk,index%tk
    if tk<64:
        atom=(row%8)*tk+col
        address=(row//8)*(8*tk)+(atom ^ (((atom//64)%(tk//8))*8))
    else:
        address=((col//64)*(tn//8)+row//8)*512+(row%8)*64+(((col//8)%8) ^ (row%8))*8+col%8
    logical=value.reshape(n//tn,k//tk,tn*tk)[...,address]
    return logical.reshape(n//tn,k//tk,tn,tk).permute(0,2,1,3).reshape(n,k)


def check(directory, *, pages=True, tile_n=64, tile_k=64, cap_bytes=24*1024**3):
    torch.set_num_threads(4)
    directory=Path(directory);config=json.loads((directory/'config.json').read_text())
    verified=index_check(directory);checkpoint=Checkpoint(directory)
    h,i,l=int(config['hidden_size']),int(config['moe_intermediate_size']),int(config['num_hidden_layers'])
    e=int(config.get('num_experts',config.get('num_local_experts',0)))
    device=torch.device('cuda',torch.cuda.current_device());total=torch.cuda.get_device_properties(device).total_memory
    if not 0<cap_bytes<=total:
        raise ValueError('memory cap exceeds the current device')
    torch.cuda.set_per_process_memory_fraction(cap_bytes/total,device)
    torch.cuda.reset_peak_memory_stats(device)
    buffers=[];expert_names=set()
    def add(name,recipe,elements):
        buffers.append(Buffer(name,1,0,elements,0,json.dumps(recipe,sort_keys=True)))
    for layer in range(l):
        prefix=f'model.layers.{layer}.mlp.experts.'
        for part,width,inner in [('gate_up',2*i,h),('down',h,i)]:
            recipe=dict(kind='expert_stack',prefix=prefix,part=part,experts=e,hidden=h,intermediate=i,u=16)
            if pages:
                recipe=dict(kind='tile_pages',source=recipe,tile_n=tile_n,tile_k=tile_k)
            add(prefix+part,recipe,e*width*inner)
        expert_names.update(prefix+f'{expert}.{kind}_proj.weight' for expert in range(e)
                            for kind in ('gate','up','down'))
    for name,entry in verified['tensors'].items():
        if name not in expert_names:
            add(name,dict(kind='alias',source=name),entry['bytes']//2)
    phase=SimpleNamespace(buffers=buffers)
    weights=load_weights(directory,phase,phase,device=device)
    torch.cuda.synchronize()
    peak=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
    if peak['reserved']>cap_bytes:
        raise AssertionError('streaming loader exceeded the specified memory cap')
    checked=0
    for layer in range(l):
        prefix=f'model.layers.{layer}.mlp.experts.'
        for expert in range(e):
            stem=prefix+f'{expert}.'
            gate,up=(checkpoint.tensor(stem+kind+'_proj.weight') for kind in ('gate','up'))
            expected_gate=torch.cat([part for first in range(0,i,16)
                for part in (gate[first:first+16],up[first:first+16])])
            down=checkpoint.tensor(stem+'down_proj.weight')
            for part,n,k,expected in [('gate_up',2*i,h,expected_gate),('down',h,i,down)]:
                actual=weights[prefix+part].reshape(e,-1)[expert].cpu()
                actual=unpack_pages(actual,n,k,tile_n,tile_k) if pages else actual.reshape(n,k)
                if not torch.equal(actual,expected):
                    raise AssertionError('expert weight differs from original checkpoint: '+stem+part)
                checked+=1
    for name,entry in verified['tensors'].items():
        if name not in expert_names and not torch.equal(weights[name].cpu(),checkpoint.tensor(name)):
            raise AssertionError('non-expert weight differs from checkpoint: '+name)
    return dict(evidence='verified',passed=True,scope='real checkpoint weight packing/allocation only',
        checkpoint_index_sha256=verified['index_sha256'],checkpoint_config_sha256=verified['config_sha256'],
        checkpoint_bytes=verified['total_bytes'],layers=l,experts=e,shared_phases=2,
        pages=pages,tile_n=tile_n if pages else None,tile_k=tile_k if pages else None,
        cap_bytes=cap_bytes,peak_bytes=peak,expert_matrices_checked=checked,
        deployment_bytes=sum(t.numel()*t.element_size() for t in weights.values()),
        gpu_name=torch.cuda.get_device_properties(device).name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--layout',choices=('dense','pages'),default='pages')
    parser.add_argument('--cap-bytes',type=int,default=24*1024**3)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=check(args.checkpoint,pages=args.layout=='pages',cap_bytes=args.cap_bytes)
    result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
