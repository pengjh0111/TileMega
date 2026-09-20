#!/usr/bin/env python3
"""Export the maximal connected Qwen3 graph, across the same cuts A-a uses.

Qwen3's decoder layer is Llama's plus a per-head RMSNorm on the Q and K
projections. That normalization is the one operator the task-work derivation
refuses (`lib/Analysis/TaskWork.cpp:279-283`: its reduction axis is indexed
`128*floordiv(c, 128) + 1*r`, not the exact unit axis the local reduction
asks for), and it sits between the projection and RoPE -- which R6 already
cuts. So the A-a cut list carries over unchanged: cutting RoPE cuts the
per-head normalization with it, and the connected remainder is the same
residual / V-cache-attention-O / SwiGLU chain that the Llama graph keeps.

The layer module itself is imported from `MODELS/export_covered.py` rather
than restated, so the two graphs are the same code on two configurations.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[3]
COVERED=REPO/'docs/experiments/MODELS/export_covered.py'
# A plain import, not a loader trick: dynamo re-imports the defining module by
# name while it traces the layer, so the module has to be findable on the path.
sys.path.insert(0,str(COVERED.parent));import export_covered as covered

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--config',type=Path,default=REPO/'docs/experiments/MODELS/sources/qwen_config.json')
 ap.add_argument('--seq',type=int,default=4);ap.add_argument('--past',type=int,default=3)
 ap.add_argument('--layers',type=int,default=0)
 a=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20260918)
 config=json.loads(a.config.read_text())
 if a.layers:config=dict(config,num_hidden_layers=a.layers)
 h=config['hidden_size'];kv=config['num_key_value_heads']
 dim=config.get('head_dim',h//config['num_attention_heads']);layers=config['num_hidden_layers']
 model=covered.Covered(config).eval().to(torch.bfloat16)
 names=[name for j in range(layers)
        for name in (f'norm1_{j}',f'rotated_q{j}',f'rotated_k{j}',f'past_k{j}',f'past_v{j}',f'norm2_{j}')]
 namespace={};exec('def forward(self,hidden,final_norm,'+', '.join(names)+
                   '):\n return self.execute(hidden,final_norm,('+', '.join(names)+',))\n',namespace)
 object.__setattr__(model,'forward',namespace['forward'].__get__(model,type(model)))
 def rand(*shape):return torch.randn(*shape,dtype=torch.bfloat16)
 heads=config['num_attention_heads']
 args=[rand(1,a.seq,h),rand(1,a.seq,h)]
 for j in range(layers):
  args.extend((rand(1,a.seq,h),rand(1,a.seq,heads*dim),rand(1,a.seq,kv*dim),
               rand(1,kv,a.past,dim),rand(1,kv,a.past,dim),rand(1,a.seq,h)))
 seq=torch.export.Dim('seq_len',min=1,max=128);past=torch.export.Dim('past_len',min=0,max=512)
 dynamic=[{1:seq},{1:seq}]
 for j in range(layers):dynamic.extend(({1:seq},{1:seq},{1:seq},{2:past},{2:past},{1:seq}))
 a.out.mkdir(parents=True,exist_ok=True);fixture=a.out/'fixture';fixture.mkdir(exist_ok=True)
 program=torch.export.export(model,tuple(args),dynamic_shapes=tuple(dynamic),strict=True)
 torch.export.save(program,a.out/'exported_program.pt2')
 with torch.no_grad():outputs=program.module()(*args)
 users=[s for s in program.graph_signature.input_specs if s.kind.name=='USER_INPUT']
 for spec_,value in zip(users,args):covered.write(fixture/f'input_{spec_.arg.name}.bin',value)
 for name,value in program.state_dict.items():
  covered.write(fixture/('state_'+name.replace('.','_')+'.bin'),value)
 for j,value in enumerate(outputs):covered.write(fixture/f'reference_{j}.bin',value)
 manifest=dict(seq=a.seq,past=a.past,config=config,config_source=str(a.config),
  config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),scope=__doc__,
  layer_module=str(COVERED),outputs=len(outputs),inputs=len(args),
  golden_device='cpu',dtype='torch.bfloat16',torch_version=torch.__version__)
 (fixture/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
