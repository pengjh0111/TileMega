#!/usr/bin/env python3
"""Materialize a fixed searched configuration for A/B/kW and stages ablations."""
import argparse,hashlib,json,pathlib,re,subprocess,time
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]

def point(model,seq,key,out,pure,width,fixture,actual=0,stages=None):
 old=E.parent/'SOLVER_V2/legacy_r8_domain'/f'{model}_s{seq}'
 export=json.loads((old/'solve.json').read_text())['command'][1]
 if stages is not None:key=re.sub(r's\d+k',f's{stages}k',key)
 kappa=int(re.search(r'kappa=(\d+)',key)[1]);residency=int(re.search(r'residency=(\d+)',key)[1])
 if actual:key=re.sub(r'residency=\d+',f'residency={min(residency,actual)}',key);residency=min(residency,actual)
 out.mkdir(parents=True,exist_ok=True);prefix=out/'selected';binary=ROOT/'build-portable/tools/tilemega-flow-point'
 cmd=[str(binary),export,str(E/'fit/target.json'),str(seq),'3',str(kappa),str(residency),str(width),key,str(prefix),str(E.parent/'SIMULATOR/hop_ns.tsv'),str(fixture),str(int(pure)),str(actual)]
 if (out/'materialize.log').exists():raise RuntimeError('refusing overwrite '+str(out))
 record=dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());(out/'command.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
 with (out/'materialize.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 record.update(exit=code,wall_seconds=time.monotonic()-start);(out/'exit.json').write_text(json.dumps(record,indent=2)+'\n')
 if code:raise RuntimeError('fixed-point materialization failed '+str(out))
 return prefix.with_suffix('.cu')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=['llama','qwen3']);p.add_argument('seq',type=int);p.add_argument('key');p.add_argument('out',type=pathlib.Path)
 p.add_argument('--pure',action='store_true');p.add_argument('--width',default='8');p.add_argument('--actual',type=int,default=0);p.add_argument('--stages',type=int);a=p.parse_args()
 fixture=json.loads((E/'floor'/f'{a.model}_s{a.seq}.command.json').read_text())[5]
 point(a.model,a.seq,a.key,a.out,a.pure,a.width,fixture,a.actual,a.stages)
