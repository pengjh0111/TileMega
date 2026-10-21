"""Verify an offline position-only RoPE table using the pinned independent HF module."""
import argparse
import inspect
import json
from pathlib import Path
import torch
import torch_npu
import transformers
from transformers import AutoConfig
from tilemega.bench.as2_models import RUNS,REPO,DENSE,controlled
from tilemega.bench.as2_native import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=DENSE,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=controlled(a.out,[RUNS]);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.npu.set_device(0)
    path=REPO/'configs/ascend/as2_models'/(a.model+'.json');cfg=json.loads(path.read_text())
    config=AutoConfig.for_model(cfg['model_type'],**{k:v for k,v in cfg.items() if k!='model_type'})
    if cfg['model_type']=='llama':
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as Rotary
    elif cfg['model_type']=='qwen3':
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as Rotary
    else:raise ValueError('Dense rotary contract is not registered')
    dummy=torch.zeros((1,1,cfg['hidden_size']),device='npu:0',dtype=torch.bfloat16)
    module=Rotary(config,device=torch.device('npu:0'))
    cpu_module=Rotary(config,device=torch.device('cpu'))
    positions=torch.arange(1088,device='npu:0',dtype=torch.int64)[None,:]
    with torch.inference_mode():
        cos,sin=module(dummy,positions)
        cpu_cos,cpu_sin=cpu_module(dummy.cpu(),positions.cpu())
        cos,sin=cos[0].cpu().contiguous(),sin[0].cpu().contiguous()
        checks=[]
        for past,length in [(0,64),*( (p,1) for p in (63,64,65,66,127,128,129,1023,1024,1086))]:
            part=torch.arange(past,past+length,device='npu:0',dtype=torch.int64)[None,:]
            c,s=module(dummy,part)
            checks.append(dict(past=past,seq=length,cos_bitwise=torch.equal(c[0].cpu(),cos[past:past+length]),sin_bitwise=torch.equal(s[0].cpu(),sin[past:past+length])))
    torch.npu.synchronize()
    if not all(c['cos_bitwise'] and c['sin_bitwise'] for c in checks):raise ValueError('Position-table results depend on query shape')
    files=[];differences=[]
    for name,t,cpu in [('cos',cos,cpu_cos[0]),('sin',sin,cpu_sin[0])]:
        file=out/(name+'.bin');file.write_bytes(t.view(torch.uint8).numpy().tobytes())
        files.append(dict(name=name,file=file.name,shape=list(t.shape),dtype='bf16',sha256=digest(file)))
        diff=(t.float()-cpu.float()).abs()
        differences.append(dict(name=name,bitwise_different=int((t!=cpu).sum()),max_abs=float(diff.max()),prefill_different=int((t[:64]!=cpu[:64]).sum())))
    source=Path(inspect.getsourcefile(Rotary))
    result=dict(schema='tilemega.as2.rope_table.v1',status='pass',model_id=a.model,config_sha256=digest(path),
        capacity=1088,head_dim=cfg['head_dim'],dtype='bf16',source='pinned_independent_HF_RotaryEmbedding_on_selected_NPU',
        scope='position/config constants only; no model weights/activations/answers',
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,transformers_version=transformers.__version__,
        source_file=str(source),source_sha256=digest(source),shape_invariance_checks=checks,
        cpu_differences=differences,files=files,allow_matmul_hf32=torch.npu.matmul.allow_hf32)
    (out/'table.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
