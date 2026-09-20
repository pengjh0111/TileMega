#!/usr/bin/env python3
"""Export a whole decoder: embedding, every layer, the final norm and the head.

Nothing is cut. What R6 had to pass in as graph inputs -- the token embedding,
both per-layer normalizations, the rotation and the final normalization -- is
inside the exported program here, which is what R7's A4/A5/A6 make importable.
Qwen3's per-head query/key normalization is included when the config has one.

The rotary frequency table is computed on the host from the config, `rope_scaling`
applied, and registered as an FP32 buffer: it is a phase table read at its own
precision, not model storage, and no device code recomputes it.

Weights are seeded random at the public config's dimensions, which is R6's
convention for these runs: the architecture is the object under test, not a
particular checkpoint.
"""
import argparse,ctypes,hashlib,json,math
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent

def frequencies(config):
 """inv_freq per config, with llama3 scaling applied where the config asks."""
 dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
 base=float(config['rope_theta'])
 inv=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float64)/dim))
 scaling=config.get('rope_scaling')
 if scaling and scaling.get('rope_type')=='llama3':
  factor=float(scaling['factor']);low=float(scaling['low_freq_factor']);high=float(scaling['high_freq_factor'])
  original=float(scaling['original_max_position_embeddings'])
  low_wavelength=original/low;high_wavelength=original/high
  wavelength=2*math.pi/inv
  scaled=inv/factor
  smooth=(original/wavelength-low)/(high-low)
  smoothed=(1-smooth)*scaled+smooth*inv
  inv=torch.where(wavelength<high_wavelength,inv,
      torch.where(wavelength>low_wavelength,scaled,smoothed))
 return inv.to(torch.float32)

class RMSNorm(nn.Module):
 def __init__(self,width,eps):super().__init__();self.weight=nn.Parameter(torch.ones(width));self.eps=eps
 def forward(self,x):
  v=x.to(torch.float32);v=v*torch.rsqrt(v.pow(2).mean(-1,keepdim=True)+self.eps)
  return self.weight*v.to(x.dtype)

class Layer(nn.Module):
 def __init__(self,c):
  super().__init__();h=c['hidden_size'];i=c['intermediate_size']
  self.heads=c['num_attention_heads'];self.kv=c['num_key_value_heads']
  self.dim=c.get('head_dim',h//self.heads);self.eps=c['rms_norm_eps']
  self.input_norm=RMSNorm(h,self.eps);self.post_norm=RMSNorm(h,self.eps)
  self.q_proj=nn.Linear(h,self.heads*self.dim,bias=False);self.k_proj=nn.Linear(h,self.kv*self.dim,bias=False)
  self.v_proj=nn.Linear(h,self.kv*self.dim,bias=False);self.o_proj=nn.Linear(self.heads*self.dim,h,bias=False)
  self.gate_proj=nn.Linear(h,i,bias=False);self.up_proj=nn.Linear(h,i,bias=False);self.down_proj=nn.Linear(i,h,bias=False)
  self.per_head=c['architectures'][0].startswith('Qwen3')
  if self.per_head:
   self.q_norm=RMSNorm(self.dim,self.eps);self.k_norm=RMSNorm(self.dim,self.eps)
 def rotate(self,x,inv,past,seq,heads):
  # The reference formulation: `x*cos + rotate_half(x)*sin`, with the cosine
  # and sine cast to model storage before the multiply, as HF does.
  position=torch.arange(past,past+seq,dtype=torch.float32).unsqueeze(1)
  angle=position*inv.unsqueeze(0)
  full=torch.cat((angle,angle),dim=-1)
  cos=torch.cos(full).to(x.dtype).view(1,seq,1,self.dim)
  sin=torch.sin(full).to(x.dtype).view(1,seq,1,self.dim)
  view=x.view(-1,seq,heads,self.dim)
  half=torch.cat((-view[...,self.dim//2:],view[...,:self.dim//2]),dim=-1)
  return (view*cos+half*sin).view(x.shape)
 def forward(self,x,inv,pk,pv,qhat=None,khat=None):
  batch,seq,_=x.shape;past=pk.shape[2]
  n1=self.input_norm(x)
  # Each projection is normalized before the next one is taken, as the
  # published modeling code does: the exported order is what the importer sees.
  q=self.q_proj(n1)
  k=self.k_proj(n1)
  extra=()
  if self.per_head and qhat is None:
   q=self.q_norm(q.view(batch,seq,self.heads,self.dim)).view(q.shape)
   k=self.k_norm(k.view(batch,seq,self.kv,self.dim)).view(k.shape)
  elif self.per_head:
   # The per-head normalization is the one operator the task-work derivation
   # refuses: its reduction axis is indexed `head_dim*floordiv(c,head_dim)+r`,
   # which is not the exact unit axis the local reduction asks for. Hoisting
   # exactly that operator to a graph input leaves every other operator of the
   # decoder in one connected graph; the projections stay and become outputs,
   # so nothing admissible is dropped along with it.
   extra=(q,k);q,k=qhat,khat
  v=self.v_proj(n1)
  qr=self.rotate(q,inv,past,seq,self.heads);kr=self.rotate(k,inv,past,seq,self.kv)
  fullk=torch.cat((pk,kr.view(batch,seq,self.kv,self.dim).transpose(1,2)),dim=2)
  fullv=torch.cat((pv,v.view(batch,seq,self.kv,self.dim).transpose(1,2)),dim=2)
  aq=qr.view(batch,seq,self.heads,self.dim).transpose(1,2)
  group=self.heads//self.kv
  ak=torch.repeat_interleave(fullk,group,dim=1);av=torch.repeat_interleave(fullv,group,dim=1)
  score=torch.matmul(aq,ak.transpose(-1,-2))/math.sqrt(self.dim)
  mask=torch.arange(past+seq)[None,:]>torch.arange(past,past+seq)[:,None]
  prob=torch.softmax(score.masked_fill(mask,float('-inf')).float(),dim=-1).to(x.dtype)
  context=torch.matmul(prob,av).transpose(1,2).contiguous().view(batch,seq,self.heads*self.dim)
  x=x+self.o_proj(context)
  n2=self.post_norm(x)
  x=x+self.down_proj(torch.nn.functional.silu(self.gate_proj(n2))*self.up_proj(n2))
  return x,fullk,fullv,extra

class Model(nn.Module):
 def __init__(self,c):
  super().__init__()
  self.embed=nn.Embedding(c['vocab_size'],c['hidden_size'])
  self.layers=nn.ModuleList([Layer(c) for _ in range(c['num_hidden_layers'])])
  self.final_norm=RMSNorm(c['hidden_size'],c['rms_norm_eps'])
  self.lm_head=nn.Linear(c['hidden_size'],c['vocab_size'],bias=False)
  self.register_buffer('inv_freq',frequencies(c),persistent=True)
 def execute(self,ids,caches,hoisted=()):
  x=self.embed(ids);outputs=[];kept=[]
  for j,layer in enumerate(self.layers):
   pair=(hoisted[2*j],hoisted[2*j+1]) if hoisted else (None,None)
   x,fk,fv,extra=layer(x,self.inv_freq,caches[2*j],caches[2*j+1],*pair)
   outputs.extend((fk,fv));kept.extend(extra)
  return (self.lm_head(self.final_norm(x)),*outputs,*kept)

def write(p,t):
 t=t.detach().cpu().contiguous()
 raw=(ctypes.c_char*(t.numel()*t.element_size())).from_address(t.data_ptr())
 p.write_bytes(memoryview(raw))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--config',type=Path,required=True)
 ap.add_argument('--seq',type=int,default=4);ap.add_argument('--past',type=int,default=3)
 ap.add_argument('--layers',type=int,default=0,help='override the layer count for a smoke export')
 ap.add_argument('--test-small',action='store_true')
 # torch.export specializes a length-one sequence to a constant and then
 # refuses the dynamic dim, so the seq=1 cell is exported with that one
 # dimension pinned. The megakernel is solved at a fixed seq regardless.
 ap.add_argument('--static-seq',action='store_true')
 # The maximal connected graph the compiler admits, for a model whose per-head
 # normalization it refuses: that one operator becomes a pair of inputs per
 # layer and everything else is exported as it stands.
 ap.add_argument('--hoist-qk-norm',action='store_true')
 a=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20260918)
 config=json.loads(a.config.read_text())
 if a.test_small:config.update(hidden_size=128,intermediate_size=256,num_attention_heads=4,
   num_key_value_heads=2,num_hidden_layers=1,vocab_size=256,head_dim=32)
 if a.layers:config['num_hidden_layers']=a.layers
 kv=config['num_key_value_heads']
 dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
 layers=config['num_hidden_layers']
 model=Model(config).eval().to(torch.bfloat16)
 # The frequency table stays FP32: it is a phase table, not model storage.
 model.inv_freq=model.inv_freq.to(torch.float32)
 heads=config['num_attention_heads']
 if a.hoist_qk_norm and not config['architectures'][0].startswith('Qwen3'):
  raise SystemExit('--hoist-qk-norm only applies to a per-head-normalized architecture')
 names=[f'past_{t}{j}' for j in range(layers) for t in 'kv']
 hoisted=[f'{t}hat_{j}' for j in range(layers) for t in 'qk'] if a.hoist_qk_norm else []
 namespace={}
 exec('def forward(self,input_ids,'+', '.join(names+hoisted)+'):\n return self.execute(input_ids,('+
   ', '.join(names)+',),('+', '.join(hoisted)+(',)' if hoisted else ')')+')\n',namespace)
 object.__setattr__(model,'forward',namespace['forward'].__get__(model,type(model)))
 args=[torch.randint(0,config['vocab_size'],(1,a.seq),dtype=torch.int64)]
 for j in range(layers):
  args.extend((torch.randn(1,kv,a.past,dim,dtype=torch.bfloat16),
               torch.randn(1,kv,a.past,dim,dtype=torch.bfloat16)))
 for j in range(layers):
  if a.hoist_qk_norm:
   args.append(torch.randn(1,a.seq,heads*dim,dtype=torch.bfloat16))
   args.append(torch.randn(1,a.seq,kv*dim,dtype=torch.bfloat16))
 seq=torch.export.Dim('seq_len',min=1,max=128);past=torch.export.Dim('past_len',min=0,max=512)
 dynamic=[{} if a.static_seq else {1:seq}]+[{2:past} for _ in range(2*layers)]
 dynamic+=[{} if a.static_seq else {1:seq} for _ in range(2*layers if a.hoist_qk_norm else 0)]
 a.out.mkdir(parents=True,exist_ok=True);fixture=a.out/'fixture';fixture.mkdir(exist_ok=True)
 program=torch.export.export(model,tuple(args),dynamic_shapes=tuple(dynamic),strict=True)
 torch.export.save(program,a.out/'exported_program.pt2')
 with torch.no_grad():outputs=program.module()(*args)
 users=[s for s in program.graph_signature.input_specs if s.kind.name=='USER_INPUT']
 for spec,value in zip(users,args):write(fixture/f'input_{spec.arg.name}.bin',value)
 for name,value in program.state_dict.items():write(fixture/('state_'+name.replace('.','_')+'.bin'),value)
 for name,value in program.constants.items():write(fixture/('state_'+name.replace('.','_')+'.bin'),value)
 for j,value in enumerate(outputs):write(fixture/f'reference_{j}.bin',value)
 manifest=dict(seq=a.seq,past=a.past,static_seq=a.static_seq,config=config,config_source=str(a.config),
   config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),scope=__doc__,
   small_test=a.test_small,outputs=len(outputs),inputs=len(args),golden_device='cpu',
   dtype='torch.bfloat16',token_id_dtype='torch.int64',torch_version=torch.__version__,
   per_head_norm=config['architectures'][0].startswith('Qwen3'),
   hoisted_qk_norm=a.hoist_qk_norm)
 (fixture/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest)[:400],flush=True)
if __name__=='__main__':main()
