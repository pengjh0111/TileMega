#!/usr/bin/env python3
"""Compare existing TaskBody buffers against the original exported CPU graph."""
import argparse,csv,re
from pathlib import Path
import torch
ap=argparse.ArgumentParser();ap.add_argument('export',type=Path);ap.add_argument('source',type=Path);ap.add_argument('dump',type=Path);a=ap.parse_args()
torch.set_num_threads(4)
program=torch.export.load(a.export);rows=[]
block=a.source.read_text().split('constexpr BufferDesc kBuffers[] = {')[1].split('\n};')[0]
names={}
for i,line in enumerate(block.strip().splitlines()):
 m=re.search(r'\{"([^"]+)".*BufferSource::kZero',line)
 if m:names[m[1]]=i
class Compare(torch.fx.Interpreter):
 def run_node(self,n):
  value=super().run_node(n)
  if n.name in names and isinstance(value,torch.Tensor):
   p=a.dump/f'buffer_{names[n.name]}.bin'
   gpu=torch.frombuffer(bytearray(p.read_bytes()),dtype=torch.bfloat16).float()
   # Attention's context buffer is token-major; FX matmul is head-major.
   aligned=value.transpose(1,2) if value.dim()==4 and str(n.target)=='aten.matmul.default' else value
   cpu=aligned.detach().contiguous().view(-1).float()
   if len(cpu)==len(gpu):
    delta=(gpu-cpu).abs();bad=delta>0.016+0.016*cpu.abs()
    rows.append(dict(node=n.name,buffer=names[n.name],target=str(n.target),shape=str(tuple(value.shape)),elements=len(cpu),different=int((delta!=0).sum()),mismatch=int(bad.sum()),max_abs=float(delta.max()),gpu389=float(gpu[389]) if len(gpu)>389 else '',cpu389=float(cpu[389]) if len(cpu)>389 else ''))
  return value
with torch.no_grad():Compare(program.module()).run(*program.example_inputs[0])
with (a.dump/'cpu_buffers_aligned.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
for r in rows:
 if r['target']=='aten.add.Tensor':print(r,flush=True)
