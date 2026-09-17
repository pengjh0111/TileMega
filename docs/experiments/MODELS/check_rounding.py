#!/usr/bin/env python3
"""Independent FP64 diagnostic at the first connected V projection.

This never rewrites the frozen CPU golden or changes any acceptance tolerance.
"""
import json,sys
from pathlib import Path
import torch
here=Path(__file__).resolve().parent;root=here/'covered_llama_admitted2/diagnostic'
torch.set_num_threads(4)
def data(index,shape):return torch.frombuffer(bytearray((root/f'buffer_{index}.bin').read_bytes()),dtype=torch.bfloat16).reshape(shape)
x=data(0,(4,2048));weight=data(5,(512,2048));gpu=data(6,(4,512))
cpu=torch.nn.functional.linear(x,weight);exact=torch.nn.functional.linear(x.double(),weight.double()).to(torch.bfloat16)
rows=[]
for t,j in torch.nonzero(gpu!=cpu).tolist():
 rows.append(dict(token=t,column=j,gpu=float(gpu[t,j]),cpu=float(cpu[t,j]),fp64_rounded=float(exact[t,j]),fp64_accumulator=float(torch.dot(x[t].double(),weight[j].double()))))
result=dict(scope=__doc__,gpu_vs_cpu=int((gpu!=cpu).sum()),gpu_vs_fp64=int((gpu!=exact).sum()),cpu_vs_fp64=int((cpu!=exact).sum()),different_elements=rows)
(root/'first_v_rounding.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
program=torch.export.load(here/'covered_llama/exported_program.pt2')
class Check(torch.fx.Interpreter):
 def run_node(self,n):
  value=super().run_node(n)
  if n.name=='matmul_1':
   args,_=self.fetch_args_kwargs_from_env(n)
   v=data(11,(1,8,7,64)).repeat_interleave(4,dim=1)
   conditional=torch.matmul(args[0],v).transpose(1,2).contiguous().view(-1)
   observed=data(13,(8192,))
   row=dict(context_differences_with_original_cpu_v=int((observed!=value.transpose(1,2).contiguous().view(-1)).sum()),context_differences_after_substituting_gpu_v=int((observed!=conditional).sum()))
   (root/'first_context_isolation.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
   raise StopIteration
  return value
try:
 with torch.no_grad():Check(program.module()).run(*program.example_inputs[0])
except StopIteration:pass
