#!/usr/bin/env python3
"""Same-geometry split1/16 phase probes from the numerically admitted sweep.

These are diagnostic snapshots, not an additional selected performance arm.
Golden tolerances and source geometry remain unchanged.
"""
import argparse,time
from pathlib import Path
from measure import build,run,HERE
from search import table

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','run']);ap.add_argument('--arch',default='sm_89');a=ap.parse_args();c=HERE/'raw/real_s4';session=str(time.time_ns())
 for i,r in enumerate(table(c/'numeric_expansion.tsv')):
  if r['split'] not in ('1','16'):continue
  arm='operand_'+r['key'];spec=dict(r,source=str(c/'plans'/r['key']/'eft.cu'),placement_macro='0')
  if a.action=='build':
   if build(c,'real',arm,spec,a.arch,phase=True):raise RuntimeError('operand probe compile')
  else:run(c,'real',4,arm+'_phase',c/'operand_probe'/r['key'],0,i,session,phase=True)
if __name__=='__main__':main()
