#!/usr/bin/env python3
"""Replay fixed legacy geometries to prove local release optimization identity.

CPU prediction replay, not new GPU performance. The eight archived flow
predictions are the pre-optimization output for identical target/CG/theta.
"""
import concurrent.futures,hashlib,json,pathlib,re,shutil,subprocess,time,csv
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
binary=pathlib.Path('/root/r9b_work/local-release-audit')
if not binary.exists():shutil.copy2(ROOT/'build-portable/tools/tilemega-flow-audit',binary)
def work(cell):
 out=E/'local_release_identity'/cell;out.mkdir(parents=True,exist_ok=True)
 cmd=json.loads((E/'flow_final'/cell/'command.json').read_text())['command'][:7];cmd[0]=str(binary)
 (out/'command.json').write_text(json.dumps(dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
 start=time.monotonic()
 with (out/'run.log').open('x') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 text=(out/'run.log').read_text();old=next(csv.DictReader((E/'flow_final'/cell/'home.flow.tsv').open(),delimiter='\t'))
 fields={'T':r'FLOW ns=([\d.eE+-]+)','synchronization':r'DECOMPOSE sync=([\d.eE+-]+)','fixed':r'DECOMPOSE.* fixed=([\d.eE+-]+)','contention':r'DECOMPOSE.* contention=([\d.eE+-]+)','chain_delay':r'DECOMPOSE.* chain=([\d.eE+-]+)','pg_upper_bound':r'DECOMPOSE.* pg=([\d.eE+-]+)'}
 values={k:float(re.search(pattern,text)[1]) for k,pattern in fields.items()} if code==0 else {}
 equal=code==0 and all(v.hex()==float(old[k]).hex() for k,v in values.items())
 record=dict(exit=code,seconds=time.monotonic()-start,bit_equal=equal,values=values,baseline={k:old[k] for k in fields})
 (out/'exit.json').write_text(json.dumps(record,indent=2)+'\n');print(cell,record,flush=True);return equal
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(work,[f'{m}_s{s}' for m in ('llama','qwen3') for s in (1,4,16,64)]))
if not all(results):raise SystemExit(1)
