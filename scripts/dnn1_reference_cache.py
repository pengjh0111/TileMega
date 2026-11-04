#!/usr/bin/env python3
"""Build an immutable deterministic CPU oracle for NAFNet random-input checks."""
from pathlib import Path
import argparse,copy,hashlib,json,os,time
import torch
from safetensors.torch import load_file,save_file
from tilemega.dnn.reference_cache import RECIPE,sha

def build(export,bridge,inputs,batch,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU reference construction requires CUDA_VISIBLE_DEVICES=''")
    manifest=json.loads((export/'manifest.json').read_text())
    if manifest['model']!='nafnet' or batch not in (1,2,4,8,16):
        raise ValueError('reference cache supports the DNN1 NAFNet batch domain')
    if manifest['artifacts']['exported_program.pt2']!=sha(export/'exported_program.pt2') or manifest['checkpoint']['sha256']!=sha(export/'checkpoint/model.safetensors'):
        raise ValueError('export or checkpoint differs from its manifest')
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.deterministic=True
    program=torch.export.load(export/'exported_program.pt2');module=program.module();oracle=copy.deepcopy(module).float()
    for node in oracle.graph.nodes:
        promote=lambda v:torch.float32 if v is torch.bfloat16 else v
        node.args=torch.utils._pytree.tree_map(promote,node.args);node.kwargs=torch.utils._pytree.tree_map(promote,node.kwargs)
    oracle.recompile();desc=json.loads(bridge.read_text());names=[x['name'] for x in desc['signature']['inputs'] if x['kind']=='USER_INPUT']
    supplied=load_file(str(inputs),device='cpu')
    if set(supplied)!=set(names) or any(x.shape[0]!=batch for x in supplied.values()):
        raise ValueError('identified inputs do not match the exported signature or batch')
    arguments=[supplied[x] for x in names];refs={}
    with torch.inference_mode():
        for kind,model,args in [('bf16',module,arguments),('fp32',oracle,[x.float() if x.is_floating_point() else x for x in arguments])]:
            positional,kw=torch.utils._pytree.tree_unflatten(args,program.call_spec.in_spec)
            values=torch.utils._pytree.tree_flatten(model(*positional,**kw))[0]
            for i,x in enumerate(v for v in values if isinstance(v,torch.Tensor)):refs[f'{kind}/{i}']=x.contiguous()
    if torch.cuda.is_initialized():raise RuntimeError('CPU reference unexpectedly initialized CUDA')
    save_file(refs,str(out/'reference.safetensors'))
    paths={'exported_program':export/'exported_program.pt2','checkpoint':export/'checkpoint/model.safetensors','manifest':export/'manifest.json','bridge':bridge,'inputs':inputs}
    value=dict(schema=1,batch=batch,model='nafnet',cpu_only=True,cuda_initialized=False,recipe=RECIPE,
        recipe_sha256=hashlib.sha256(json.dumps(RECIPE,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        torch={'version':torch.__version__,'init_sha256':sha(torch.__file__),'libtorch_cpu_sha256':sha(Path(torch.__file__).resolve().parent/'lib/libtorch_cpu.so')},
        sources={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
        references={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()} for k,v in refs.items()},
        reference_file_sha256=sha(out/'reference.safetensors'),builder_sha256=sha(__file__),wall_s=time.time()-start)
    (out/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export','bridge','inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch',type=int,required=True);a=p.parse_args();r=build(a.export,a.bridge,a.inputs,a.batch,a.out)
    print(json.dumps(dict(status='built',batch=a.batch,wall_s=r['wall_s'],reference_file_sha256=r['reference_file_sha256'])))
if __name__=='__main__':main()
