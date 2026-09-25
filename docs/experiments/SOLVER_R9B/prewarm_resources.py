#!/usr/bin/env python3
"""Revalidate cached wrapper shapes after host target-schema changes; no GPU kernels run."""
import concurrent.futures,json,pathlib,re,subprocess,time
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2];cache=pathlib.Path('/root/r9_work/variant_resources');out=E/'resources';out.mkdir(exist_ok=True)
shapes=set()
for p in cache.glob('*/probe.cu'):
 text=p.read_text()
 if '#define TILEMEGA_MODEL_BF16 1' not in text:continue
 if 'probe_RMSNorm' in text:shapes.add(None);continue
 matches=[re.search(r'#define TILEMEGA_GEMM_'+part+r' (\d+)',text) for part in ('TILE_M','TILE_N','TILE_K','STAGES')]
 if all(matches):shapes.add(tuple(int(m[1]) for m in matches))
def probe(shape):
 tag='nongemm' if shape is None else '_'.join(map(str,shape));cmd=['python3',str(root/'tools/probe_variant.py'),'--cache',str(cache),'--output',str(out/(tag+'.json')),'--arch','sm_89','--dtype','bf16']
 cmd+=['--nongemm'] if shape is None else ['--tile',','.join(map(str,shape))]
 (out/(tag+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n');start=time.monotonic()
 with (out/(tag+'.log')).open('w') as log:code=subprocess.call(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 (out/(tag+'.exit.json')).write_text(json.dumps(dict(exit=code,wall_seconds=time.monotonic()-start))+'\n');print(tag,code,flush=True)
 return code
print('shapes',len(shapes),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(probe,sorted(shapes,key=str)))
raise SystemExit(any(codes))
