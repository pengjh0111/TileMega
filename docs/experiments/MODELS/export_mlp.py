#!/usr/bin/env python3
"""Export existing SwiGLU TaskBody coverage at the published model dimensions.

These are independent post-normalization MLP regions, one per original layer.
Boundary inputs are explicit random tensors. This is NOT a full-model timing:
attention, normalization, RoPE, embedding and final head are excluded. No
missing operator is replaced with an approximate implementation.
"""
import argparse,ctypes,hashlib,json,subprocess,sys,time
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
class Region(nn.Module):
 def __init__(self,h,i):
  super().__init__();self.gate_proj=nn.Linear(h,i,bias=False);self.up_proj=nn.Linear(h,i,bias=False);self.down_proj=nn.Linear(i,h,bias=False)
 def forward(self,x,residual):return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))+residual
class Regions(nn.Module):
 def __init__(self,l,h,i):super().__init__();self.layers=nn.ModuleList([Region(h,i) for _ in range(l)])
 def execute(self,args):return tuple(layer(args[2*j],args[2*j+1]) for j,layer in enumerate(self.layers))
def write(p,t):
 t=t.detach().cpu().contiguous();raw=(ctypes.c_char*(t.numel()*t.element_size())).from_address(t.data_ptr());p.write_bytes(memoryview(raw))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',choices=['llama','qwen'],default='llama');ap.add_argument('--seq',type=int,default=4);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20260916)
 source=HERE/'sources'/('llama_config_public_copy.json' if a.model=='llama' else 'qwen_config.json');config=json.loads(source.read_text())
 h,i,l=[config[k] for k in ('hidden_size','intermediate_size','num_hidden_layers')]
 a.out.mkdir(parents=True,exist_ok=True);fixture=a.out/'fixture';fixture.mkdir(exist_ok=True)
 model=Regions(l,h,i).eval().to(torch.bfloat16)
 names=[name for j in range(l) for name in (f'normalized_{j}',f'residual_{j}')]
 namespace={};exec('def forward(self, '+', '.join(names)+'):\n return self.execute(('+', '.join(names)+',))\n',namespace)
 object.__setattr__(model,'forward',namespace['forward'].__get__(model,type(model)))
 args=tuple(torch.randn(1,a.seq,h,dtype=torch.bfloat16) for _ in names)
 seq=torch.export.Dim('seq_len',min=1,max=128)
 program=torch.export.export(model,args,dynamic_shapes=tuple({1:seq} for _ in names),strict=True)
 torch.export.save(program,a.out/'exported_program.pt2')
 with torch.no_grad():outputs=program.module()(*args)
 users=[x for x in program.graph_signature.input_specs if x.kind.name=='USER_INPUT']
 for spec,value in zip(users,args):write(fixture/f'input_{spec.arg.name}.bin',value)
 for name,value in program.state_dict.items():write(fixture/('state_'+name.replace('.','_')+'.bin'),value)
 for j,value in enumerate(outputs):write(fixture/f'reference_{j}.bin',value)
 manifest=dict(seq=a.seq,past=3,layers=l,hidden=h,intermediate=i,dtype='torch.bfloat16',golden_device='cpu',torch_version=torch.__version__,scope=__doc__,config_source=str(source),config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parameters=sum(t.numel() for t in program.state_dict.values()),outputs=len(outputs))
 (fixture/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
