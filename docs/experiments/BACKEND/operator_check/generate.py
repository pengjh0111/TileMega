#!/usr/bin/env python3
"""A-a: PyTorch inputs and references for each rewritten TaskBody.

The shapes are the anchored models' own: Llama-3.2-1B's hidden width and head
geometry for RMSNorm and attention, Qwen3-1.7B's head_dim for the per-head
norm. Everything is bf16 in and out, which is the models' storage dtype; the
references apply the same rounding points the exporters do, so a difference
here is the body's, not a dtype artifact.
"""
import argparse,ctypes,json,math
from pathlib import Path
import torch

def write(path,t):
 t=t.detach().cpu().contiguous()
 raw=(ctypes.c_char*(t.numel()*t.element_size())).from_address(t.data_ptr())
 Path(path).write_bytes(memoryview(raw))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--tokens',type=int,default=4);ap.add_argument('--hidden',type=int,default=2048)
 ap.add_argument('--heads',type=int,default=32);ap.add_argument('--kv-heads',type=int,default=8)
 ap.add_argument('--head-dim',type=int,default=64);ap.add_argument('--past',type=int,default=3)
 ap.add_argument('--eps',type=float,default=1e-5)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(20260921)
 total=a.past+a.tokens
 (a.out/'shapes.txt').write_text(
   f"{a.tokens} {a.hidden} {a.heads} {a.kv_heads} {a.head_dim} {a.past}\n")

 # RMSNorm: x * rsqrt(mean(x^2)+eps) rounded to bf16, then weighted.
 x=torch.randn(a.tokens,a.hidden,dtype=torch.bfloat16)
 w=torch.randn(a.hidden,dtype=torch.bfloat16)
 scale=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+a.eps)
 ref=(x.float()*scale).to(torch.bfloat16).float()*w.float()
 write(a.out/'rmsnorm_in.bin',x);write(a.out/'rmsnorm_w.bin',w)
 write(a.out/'rmsnorm_ref.bin',ref.to(torch.bfloat16))

 # QK-norm: the same row body over one head.
 qx=torch.randn(a.tokens*a.heads,a.head_dim,dtype=torch.bfloat16)
 qw=torch.randn(a.head_dim,dtype=torch.bfloat16)
 qscale=torch.rsqrt(qx.float().pow(2).mean(-1,keepdim=True)+a.eps)
 qref=(qx.float()*qscale).to(torch.bfloat16).float()*qw.float()
 write(a.out/'qknorm_in.bin',qx);write(a.out/'qknorm_w.bin',qw)
 write(a.out/'qknorm_ref.bin',qref.to(torch.bfloat16))

 # Attention: the body's own layout -- q is [token, head, dim], k and v are
 # [kv_head, total, dim] -- and the causal mask the body applies.
 q=torch.randn(a.tokens,a.heads,a.head_dim,dtype=torch.bfloat16)
 k=torch.randn(a.kv_heads,total,a.head_dim,dtype=torch.bfloat16)
 v=torch.randn(a.kv_heads,total,a.head_dim,dtype=torch.bfloat16)
 write(a.out/'attn_q.bin',q);write(a.out/'attn_k.bin',k);write(a.out/'attn_v.bin',v)
 group=a.heads//a.kv_heads
 context=torch.zeros(a.tokens,a.heads,a.head_dim,dtype=torch.float32)
 for token in range(a.tokens):
  for head in range(a.heads):
   kv=head//group
   scores=torch.full((total,),float('-inf'))
   for pos in range(total):
    if pos<=a.past+token:
     s=torch.dot(q[token,head].float(),k[kv,pos].float())/math.sqrt(a.head_dim)
     scores[pos]=s.to(torch.bfloat16).float()   # the body rounds the score
   m=scores.max()
   e=torch.exp(scores-m)
   p=(e/e.sum()).to(torch.bfloat16).float()     # and rounds the probability
   context[token,head]=(p.unsqueeze(-1)*v[kv].float()).sum(0)
 write(a.out/'attention_ref.bin',context.to(torch.bfloat16))
 print(json.dumps(dict(tokens=a.tokens,hidden=a.hidden,heads=a.heads,
   kv_heads=a.kv_heads,head_dim=a.head_dim,past=a.past,total=total,eps=a.eps)))
if __name__=='__main__':main()
