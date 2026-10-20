"""Pinned AS-2 configurations, streaming random tensors and serving exports."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
import torch

ROOT=Path('/home/d50064638/junhuip')
REPO=ROOT/'TileMega'
WORK=ROOT/'tilemega-as2'
RUNS=REPO/'runs/as2'
SEED=20261016
GENERATOR='tilemega-as2-name-normal-v1-torch-2.7.1-quantum65536'
DENSE=('qwen3_0_6b','llama3_2_1b','llama3_2_3b','qwen3_1_7b','llama3_1_8b')
DEFERRED='qwen3_30b_a3b'


def config(model):
    from tilemega.serving.model_registry import load_model,verify_config
    return verify_config(load_model(model),REPO/'configs/ascend/as2_models'/(model+'.json'))


def describe(model):
    from transformers import AutoConfig,AutoModelForCausalLM
    cfg=config(model)
    hf=AutoConfig.for_model(cfg['model_type'],**{k:v for k,v in cfg.items() if k!='model_type'})
    with torch.device('meta'):
        module=AutoModelForCausalLM.from_config(hf,torch_dtype=torch.bfloat16)
    module.tie_weights()
    canonical={};aliases={};tensors=[]
    for name,p in module.named_parameters(remove_duplicate=False):
        if id(p) in canonical:
            aliases[name]=canonical[id(p)]
        else:
            canonical[id(p)]=name
            tensors.append(dict(name=name,shape=list(p.shape),dtype='bf16',bytes=p.numel()*2))
    return dict(model_id=model,config=cfg,config_sha256=hashlib.sha256(
        (REPO/'configs/ascend/as2_models'/(model+'.json')).read_bytes()).hexdigest(),
        unique_weight_bytes=sum(t['bytes'] for t in tensors),tensors=tensors,aliases=aliases,
        seed=SEED,generator=GENERATOR,torch_version=torch.__version__,evidence='verified-run_meta_only')


def tensor_chunks(model,tensor,seed=SEED):
    if not torch.__version__.startswith('2.7.1'):
        raise ValueError('The frozen random generator requires torch 2.7.1')
    name=tensor['name'];count=math.prod(tensor['shape']);cfg=config(model)
    key=f'{GENERATOR}:{model}:{name}:{seed}'.encode()
    generator=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8],'little')&((1<<63)-1))
    for first in range(0,count,65536):
        n=min(65536,count-first)
        if name.endswith('weight') and 'norm' in name:
            value=torch.ones(n,dtype=torch.bfloat16)
        elif name.endswith('bias'):
            value=torch.zeros(n,dtype=torch.bfloat16)
        else:
            value=(torch.randn(n,generator=generator,dtype=torch.float32)*cfg['initializer_range']).bfloat16()
        yield memoryview(value.view(torch.uint8).numpy())


def controlled(path,parents):
    path=path.absolute()
    if path.resolve()!=path or not any(path!=p and path.is_relative_to(p) for p in parents):
        raise ValueError('Use a canonical AS-2 artifact path')
    return path


def inventory(out):
    out=controlled(out,[RUNS]);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for model in (*DENSE,DEFERRED):
        desc=describe(model);cfg=desc['config']
        (out/(model+'.json')).write_text(json.dumps(desc,indent=2)+'\n')
        for batch in (1,16):
            kv=2*cfg['num_hidden_layers']*batch*cfg['num_key_value_heads']*1088*cfg['head_dim']*2
            rows.append(dict(model=model,batch=batch,isl=64,osl=1024,dtype='bf16',capacity=1088,
                layers=cfg['num_hidden_layers'],hidden=cfg['hidden_size'],q_width=cfg['num_attention_heads']*cfg['head_dim'],
                gqa=cfg['num_attention_heads']//cfg['num_key_value_heads'],unique_weight_bytes=desc['unique_weight_bytes'],
                dense_kv_bytes=kv,packed_weight_upper_bytes=desc['unique_weight_bytes'],runtime_reserve_bytes=2<<30,
                peak_preflight_partial_bytes=2*desc['unique_weight_bytes']+kv+(2<<30),
                budget_note='activation/plan workspace to be added from solved Plan; not an allocation permission',
                config_sha256=desc['config_sha256'],status='not_started',
                scope='deferred_by_user' if model==DEFERRED else 'active_dense',tpot_ms=None,speedup=None))
    (out/'main_cells.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(status='meta_inventory_only',models=6,active_dense_cells=10,deferred_moe_cells=2,
        weight_bytes={model:next(r['unique_weight_bytes'] for r in rows if r['model']==model) for model in (*DENSE,DEFERRED)})))


def export(model,phase,out):
    if model not in DENSE:
        raise ValueError('MoE implementation is deferred by the user')
    out=controlled(out,[RUNS]);
    if out.exists():raise FileExistsError(out)
    from tilemega.serving.export import export as serving_export
    from tilemega.export_bridge import serialize
    cfg=config(model)
    receipt=serving_export(cfg,phase,1088,out,64,meta_weights=True,cpu_threads=1,seed=SEED)
    program=torch.export.load(out/'exported_program.pt2')
    (out/'export.json').write_text(json.dumps(serialize(program),indent=2)+'\n')
    receipt.update(model_id=model,seed=SEED,stage='export_only_not_device_support',
        config_sha256=hashlib.sha256((REPO/'configs/ascend/as2_models'/(model+'.json')).read_bytes()).hexdigest())
    (out/'as2_export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


def weights(model,out):
    if model not in DENSE:raise ValueError('MoE is deferred')
    out=controlled(out,[WORK/'weights'])
    if out!=WORK/'weights'/model:
        raise ValueError('One canonical weight directory per model; candidates must reference it')
    desc=describe(model)
    policy=json.loads((REPO/'configs/bench/as2_disk_policy.json').read_text())
    usage=shutil.disk_usage(ROOT)
    owned=[RUNS,*(WORK/n for n in ('build','cache','tmp','weights'))]
    occupied=sum(int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0]) for p in owned)
    if usage.free/usage.total<policy['stop_below_free_fraction'] or occupied+desc['unique_weight_bytes']>policy['project_generated_soft_limit_bytes']:
        raise RuntimeError('blocked-resource: canonical weight generation exceeds the AS-2 disk policy')
    out.mkdir(parents=True,exist_ok=False);desc['status']='in_progress'
    manifest=out/'manifest.json';manifest.write_text(json.dumps(desc,indent=2)+'\n')
    for tensor in desc['tensors']:
        name=tensor['name'];path=out/(name+'.bin');digest=hashlib.sha256()
        with path.open('xb') as stream:
            for chunk in tensor_chunks(model,tensor):stream.write(chunk);digest.update(chunk)
        assert path.stat().st_size==tensor['bytes']
        tensor.update(file=path.name,sha256=digest.hexdigest())
    token_generator=torch.Generator().manual_seed(SEED)
    tokens=torch.randint(0,desc['config']['vocab_size'],(16,64),generator=token_generator,dtype=torch.int32)
    data=tokens.view(torch.uint8).numpy().tobytes();(out/'input_ids.bin').write_bytes(data)
    desc.update(status='complete',input_ids=dict(file='input_ids.bin',shape=[16,64],dtype='int32',sha256=hashlib.sha256(data).hexdigest(),b1='first row of B16'))
    manifest.write_text(json.dumps(desc,indent=2)+'\n');print(json.dumps(dict(status='complete',model=model,unique_weight_bytes=desc['unique_weight_bytes'],out=str(out))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    inv=sub.add_parser('inventory');inv.add_argument('--out',type=Path,required=True)
    exp=sub.add_parser('export');exp.add_argument('--model',choices=DENSE,required=True);exp.add_argument('--phase',choices=['prefill','decode'],required=True);exp.add_argument('--out',type=Path,required=True)
    weight=sub.add_parser('weights');weight.add_argument('--model',choices=DENSE,required=True);weight.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='inventory':inventory(args.out)
    elif args.command=='export':export(args.model,args.phase,args.out)
    else:weights(args.model,args.out)


if __name__=='__main__':main()
