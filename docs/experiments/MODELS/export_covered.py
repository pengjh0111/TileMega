#!/usr/bin/env python3
"""Export all currently implemented Llama regions across explicit semantic cuts.

Cuts: token embedding, both per-layer RMSNorms, FP32 RoPE, final RMSNorm.
Their outputs are inputs; Q/K projection outputs remain checked outputs.
The residual chain, V/cache/attention/O chain and SwiGLU chain remain connected.
No missing operation is approximated or implemented by this exporter.
"""
import argparse,ctypes,hashlib,json,math
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
class Layer(nn.Module):
 def __init__(self,c):
  super().__init__();h=c['hidden_size'];i=c['intermediate_size'];self.heads=c['num_attention_heads'];self.kv=c['num_key_value_heads'];self.dim=c.get('head_dim',h//self.heads);self.h=h
  self.q_proj=nn.Linear(h,self.heads*self.dim,bias=False);self.k_proj=nn.Linear(h,self.kv*self.dim,bias=False);self.v_proj=nn.Linear(h,self.kv*self.dim,bias=False);self.o_proj=nn.Linear(self.heads*self.dim,h,bias=False)
  self.gate_proj=nn.Linear(h,i,bias=False);self.up_proj=nn.Linear(h,i,bias=False);self.down_proj=nn.Linear(i,h,bias=False)
 def forward(self,x,n1,qr,kr,pk,pv,n2):
  batch,seq,_=x.shape;past=pk.shape[2]
  q,k,v=self.q_proj(n1),self.k_proj(n1),self.v_proj(n1)
  fullk=torch.cat((pk,kr.view(batch,seq,self.kv,self.dim).transpose(1,2)),dim=2)
  fullv=torch.cat((pv,v.view(batch,seq,self.kv,self.dim).transpose(1,2)),dim=2)
  aq=qr.view(batch,seq,self.heads,self.dim).transpose(1,2)
  ak=torch.repeat_interleave(fullk,self.heads//self.kv,dim=1);av=torch.repeat_interleave(fullv,self.heads//self.kv,dim=1)
  score=torch.matmul(aq,ak.transpose(-1,-2))/math.sqrt(self.dim)
  mask=torch.arange(past+seq)[None,:]>torch.arange(past,past+seq)[:,None]
  prob=torch.softmax(score.masked_fill(mask,float('-inf')).float(),dim=-1).to(x.dtype)
  context=torch.matmul(prob,av).transpose(1,2).contiguous().view(batch,seq,self.heads*self.dim)
  x=x+self.o_proj(context)
  x=x+self.down_proj(torch.nn.functional.silu(self.gate_proj(n2))*self.up_proj(n2))
  return x,q,k,fullk,fullv
class Covered(nn.Module):
 def __init__(self,c):
  super().__init__();self.layers=nn.ModuleList([Layer(c) for _ in range(c['num_hidden_layers'])]);self.lm_head=nn.Linear(c['hidden_size'],c['vocab_size'],bias=False)
 def execute(self,hidden,final_norm,cuts):
  outputs=[]
  for j,layer in enumerate(self.layers):
   hidden,q,k,fk,fv=layer(hidden,*cuts[6*j:6*j+6]);outputs.extend((q,k,fk,fv))
  return (hidden,self.lm_head(final_norm),*outputs)
def write(p,t):
 t=t.detach().cpu().contiguous();raw=(ctypes.c_char*(t.numel()*t.element_size())).from_address(t.data_ptr());p.write_bytes(memoryview(raw))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seq',type=int,default=4);ap.add_argument('--past',type=int,default=3);ap.add_argument('--test-small',action='store_true');a=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20260918)
 source=HERE/'sources/llama_config_public_copy.json';config=json.loads(source.read_text())
 if a.test_small:config.update(hidden_size=128,intermediate_size=256,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=1,vocab_size=256,head_dim=32)
 h=config['hidden_size'];kv=config['num_key_value_heads'];dim=config.get('head_dim',h//config['num_attention_heads']);layers=config['num_hidden_layers']
 model=Covered(config).eval().to(torch.bfloat16)
 names=[name for j in range(layers) for name in (f'norm1_{j}',f'rotated_q{j}',f'rotated_k{j}',f'past_k{j}',f'past_v{j}',f'norm2_{j}')]
 namespace={};exec('def forward(self,hidden,final_norm,'+', '.join(names)+'):\n return self.execute(hidden,final_norm,('+', '.join(names)+',))\n',namespace)
 object.__setattr__(model,'forward',namespace['forward'].__get__(model,type(model)))
 def rand(*shape):return torch.randn(*shape,dtype=torch.bfloat16)
 args=[rand(1,a.seq,h),rand(1,a.seq,h)]
 for j in range(layers):args.extend((rand(1,a.seq,h),rand(1,a.seq,h),rand(1,a.seq,kv*dim),rand(1,kv,a.past,dim),rand(1,kv,a.past,dim),rand(1,a.seq,h)))
 seq=torch.export.Dim('seq_len',min=1,max=128);past=torch.export.Dim('past_len',min=0,max=512)
 dynamic=[{1:seq},{1:seq}]
 for j in range(layers):dynamic.extend(({1:seq},{1:seq},{1:seq},{2:past},{2:past},{1:seq}))
 a.out.mkdir(parents=True,exist_ok=True);fixture=a.out/'fixture';fixture.mkdir(exist_ok=True)
 program=torch.export.export(model,tuple(args),dynamic_shapes=tuple(dynamic),strict=True);torch.export.save(program,a.out/'exported_program.pt2')
 with torch.no_grad():outputs=program.module()(*args)
 users=[s for s in program.graph_signature.input_specs if s.kind.name=='USER_INPUT']
 for spec,value in zip(users,args):write(fixture/f'input_{spec.arg.name}.bin',value)
 for name,value in program.state_dict.items():write(fixture/('state_'+name.replace('.','_')+'.bin'),value)
 for j,value in enumerate(outputs):write(fixture/f'reference_{j}.bin',value)
 manifest=dict(seq=a.seq,past=a.past,config=config,config_source=str(source),config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope=__doc__,small_test=a.test_small,outputs=len(outputs),inputs=len(args),golden_device='cpu',dtype='torch.bfloat16',torch_version=torch.__version__)
 (fixture/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
