#!/usr/bin/env python3
"""Generate local, unmaterialized sources for the R6 calibration design.

These five geometry probes calibrate the model; they do not select a production
configuration. All inputs come from --input-root on the target machine. No
worker/slot table is copied from another GPU.
"""
import argparse,json,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
GEOMETRIES=[(32,16,16,2),(32,16,32,2),(32,16,64,2),(64,128,16,2),(128,128,16,3)]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);manifest={}
 for m in a.models:
  manifest[m]={}
  for tm,tn,tk,stages in GEOMETRIES:
   arm=f'uniform_m{tm}n{tn}k{tk}s{stages}';folder=out/m/arm;folder.mkdir(parents=True,exist_ok=True);source=folder/'model.cu'
   if source.exists():raise RuntimeError('refusing overwrite '+str(source))
   plan=dict(schema='tilemega.runtime_variants.v1',variants=[dict(seq_begin=1,seq_end=2048,rope_tile_per_block=True,kv_tile_per_block=True,activation_tile_per_block=True,combiner_tile_per_block=True,uniform=dict(tile_m=tm,tile_n=tn,tile_k=tk,stages=stages,split_k=1))]);(folder/'geometry.json').write_text(json.dumps(plan,indent=2)+'\n')
   command=[str(REPO/'build-portable/tools/tilemega-compile'),str(a.input_root.resolve()/'export'/f'{m}.json'),str(source),'--variants',str(folder/'geometry.json')];meta=dict(command=command,started_ns=time.time_ns())
   with (folder/'generate.log').open('w') as f:r=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
   meta['exit_code']=r.returncode;(folder/'command.json').write_text(json.dumps(meta,indent=2)+'\n');r.check_returncode()
   manifest[m][arm]=dict(source=str(source),placement_macro='5',kappa='1',residency='0',m=str(tm),n=str(tn),k=str(tk),stages=str(stages),split='1',extra=['TRACE_KLOOP=1'])
 (out/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
