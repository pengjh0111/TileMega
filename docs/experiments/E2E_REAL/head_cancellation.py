#!/usr/bin/env python3
"""Size the cancellation in the language-model head against the golden output.

The whole-decoder run disagrees with the CPU golden on a few of its 128256
logits and on nothing else.  This reads the buffers the harness dumps with
TILEMEGA_DUMP_BUFFERS and prices, for every offending logit, the absolute mass
of the 2048 products the head sums against the value it sums to.  Dependency
free so it runs under the system interpreter, which has no numpy; 146 rows of
2048 elements do not need one.
"""
import argparse,array,random,sys
from pathlib import Path

def bf16(path,offset=0,count=None):
 a=array.array('H')
 with open(path,'rb') as f:
  f.seek(offset);a.frombytes(f.read(-1 if count is None else count*2))
 return array.array('f',array.array('I',(x<<16 for x in a)).tobytes())

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--buffers',type=Path,required=True,help='TILEMEGA_DUMP_BUFFERS directory')
 ap.add_argument('--reference',type=Path,required=True,help='golden logits, fixture/reference_0.bin')
 ap.add_argument('--logits',type=int,default=374);ap.add_argument('--hidden',type=int,default=372)
 ap.add_argument('--weight',type=int,default=375)
 ap.add_argument('--vocab',type=int,default=128256);ap.add_argument('--width',type=int,default=2048)
 a=ap.parse_args()
 logit=bf16(a.buffers/f'buffer_{a.logits}.bin');ref=bf16(a.reference)
 hidden=bf16(a.buffers/f'buffer_{a.hidden}.bin')
 if len(logit)!=len(ref):raise SystemExit('dumped logits and golden differ in length')
 positions=len(logit)//a.vocab
 # The harness' own bound, quoted from ModelHarness.cuh; this script reports
 # what exceeds it, it does not choose it.
 bad=[(k//a.vocab,k%a.vocab,abs(logit[k]-ref[k])) for k in range(len(logit))
      if abs(logit[k]-ref[k])>1.6e-2+1.6e-2*abs(ref[k])]
 print(f'HEAD_DIFF positions={positions} vocab={a.vocab} mismatch={len(bad)} '
       f'max_abs={max((d for _,_,d in bad),default=0):.9g}')
 print('HEAD_DIFF per_position='+','.join(str(sum(1 for p,_,_ in bad if p==q)) for q in range(positions)))
 def priced(p,j):
  w=bf16(a.buffers/f'buffer_{a.weight}.bin',offset=j*a.width*2,count=a.width)
  total=mass=0.0
  for i in range(a.width):
   term=hidden[p*a.width+i]*w[i];total+=term;mass+=abs(term)
  return total,mass
 ratios=[]
 for rank,(p,j,d) in enumerate(sorted(bad,key=lambda x:-x[2])):
  total,mass=priced(p,j);ratios.append(mass/max(abs(total),1e-12))
  if rank<5:
   print(f'HEAD_ELEM position={p} token={j} gpu={logit[p*a.vocab+j]:.9g} '
         f'golden={ref[p*a.vocab+j]:.9g} recomputed_fp64={total:.9g} '
         f'sum_abs_terms={mass:.9g} cancellation={ratios[-1]:.6g}')
 ratios.sort()
 if ratios:
  print(f'HEAD_CANCELLATION offending={len(ratios)} min={ratios[0]:.6g} '
        f'median={ratios[len(ratios)//2]:.6g} max={ratios[-1]:.6g}')
 random.seed(0);sample=[]
 for _ in range(64):
  total,mass=priced(0,random.randrange(a.vocab));sample.append(mass/max(abs(total),1e-12))
 sample.sort()
 print(f'HEAD_CANCELLATION_SAMPLE tokens=64 position=0 median={sample[32]:.6g} max={sample[-1]:.6g}')
if __name__=='__main__':main()
