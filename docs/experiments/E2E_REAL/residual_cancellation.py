#!/usr/bin/env python3
"""Price the golden-versus-device difference on one whole-model output.

The maximal connected Qwen3 graph runs with all three levels bit-identical
(`E2E_HASH`) and disagrees only with the CPU golden, on the output that carries
28 layers of residual accumulation. This reads the buffer the harness dumps
with `TILEMEGA_DUMP_BUFFERS` next to that output's golden file and reports, for
every element outside `Compare()`'s tolerance, how large the value is that the
difference sits on -- the question being whether the offenders are the small
values (cancellation) or the large ones (a wrong computation).

Dependency free so it runs under the system interpreter.
"""
import argparse,array,statistics
from pathlib import Path
# ModelHarness.cuh Compare(): absolute plus relative, one constant for both.
TOL=1.6e-2

def bf16(path):
 raw=array.array('H');raw.frombytes(Path(path).read_bytes())
 return array.array('f',array.array('I',(x<<16 for x in raw)).tobytes())

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--buffers',type=Path,required=True,help='TILEMEGA_DUMP_BUFFERS directory')
 ap.add_argument('--reference',type=Path,required=True,help='fixture/reference_<index>.bin')
 ap.add_argument('--buffer',type=int,required=True,help='buffer id the harness named')
 ap.add_argument('--show',type=int,default=10)
 a=ap.parse_args()
 got=bf16(a.buffers/f'buffer_{a.buffer}.bin');want=bf16(a.reference)
 if len(got)!=len(want):
  raise SystemExit(f'dump has {len(got)} elements, golden has {len(want)}')
 bad=[(i,want[i],got[i]) for i in range(len(want))
      if abs(got[i]-want[i])>TOL+TOL*abs(want[i])]
 mag=sorted(abs(v) for v in want)
 badmag=sorted(abs(e) for _,e,_ in bad)
 print(f'ELEMENTS total={len(want)} outside_tolerance={len(bad)} '
       f'tolerance={TOL}+{TOL}*|expected|')
 q=lambda s,f:s[min(len(s)-1,int(f*len(s)))] if s else float('nan')
 print(f'MAGNITUDE all |expected| median={statistics.median(mag):.6g} '
       f'p90={q(mag,0.9):.6g} max={mag[-1]:.6g}')
 if bad:
  print(f'MAGNITUDE offending |expected| median={statistics.median(badmag):.6g} '
        f'p90={q(badmag,0.9):.6g} max={badmag[-1]:.6g}')
  worst=sorted(bad,key=lambda r:-abs(r[2]-r[1]))[:a.show]
  print('index\texpected\tdevice\tabs_diff\trel_diff')
  for i,e,g in worst:
   print(f'{i}\t{e:.8g}\t{g:.8g}\t{abs(g-e):.6g}\t{abs(g-e)/abs(e) if e else float("inf"):.6g}')
if __name__=='__main__':main()
