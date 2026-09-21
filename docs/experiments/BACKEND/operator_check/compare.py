#!/usr/bin/env python3
"""A-a: element-by-element comparison of each rewritten body against PyTorch.

The tolerance is the harness's own (`Compare()` in ModelHarness.cuh):
`1.6e-2 + 1.6e-2*|expected|` on the BF16 profile. Nothing here widens it.
Dependency free, so it runs under the system interpreter.
"""
import argparse,array,struct
from pathlib import Path
TOL=1.6e-2

def bf16(path):
 raw=array.array('H');raw.frombytes(Path(path).read_bytes())
 return array.array('f',array.array('I',(x<<16 for x in raw)).tobytes())

def compare(name,got,want):
 assert len(got)==len(want),(name,len(got),len(want))
 bad=[(i,want[i],got[i]) for i in range(len(want))
      if abs(got[i]-want[i])>TOL+TOL*abs(want[i])]
 worst=max(((abs(g-e),i,e,g) for i,e,g in ((i,want[i],got[i]) for i in range(len(want)))),
           default=(0,0,0,0))
 return dict(operator=name,elements=len(want),mismatch=len(bad),
             max_abs=worst[0],at=worst[1],expected=worst[2],device=worst[3])

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--dir',type=Path,required=True)
 ap.add_argument('--out',type=Path)
 a=ap.parse_args()
 rows=[compare(n,bf16(a.dir/f'{n}_device.bin'),bf16(a.dir/f'{n}_ref.bin'))
       for n in ('rmsnorm','qknorm','attention')]
 header='operator\telements\tmismatch\tmax_abs\tat\texpected\tdevice'
 lines=[header]+['\t'.join(str(r[k]) for k in
        ('operator','elements','mismatch','max_abs','at','expected','device')) for r in rows]
 text='\n'.join(lines)+'\n'
 if a.out:a.out.write_text(text)
 print(text,end='')
 failed=[r['operator'] for r in rows if r['mismatch']]
 print('A-a RESULT operators',len(rows),'passing',len(rows)-len(failed),
       'failing',','.join(failed) or '-')
 raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
