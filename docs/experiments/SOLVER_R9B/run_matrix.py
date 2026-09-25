#!/usr/bin/env python3
"""R9b full-domain matrix: independent single-thread searches, queued GPU tests.

Two cells may prepare concurrently; each solver obeys --search-jobs 1. GPU
execution is serialized by measure.py's lock and external-utilization admission.
Products: matrix/<model>_s<seq>/solve.*, selected.cu.{search,timing,top3}.tsv,
per-candidate .measurement/ with ten fresh process logs, and measured winner CG.
No domain restriction, pass reduction, or missed cell is silently accepted.
"""
import concurrent.futures,json,pathlib,subprocess,sys,time
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2]
def solve(cell):
 model,seq=cell;out=E/'matrix'/f'{model}_s{seq}';out.mkdir(parents=True,exist_ok=True)
 with (out/'runner.log').open('x') as log:
  code=subprocess.call([sys.executable,str(E/'run_search.py'),model,str(seq)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 if code:return dict(model=model,seq=seq,stage='solve_failed',exit=code)
 return dict(model=model,seq=seq,stage='solved',exit=0)
def measure(item):
 cell=f'{item["model"]}_s{item["seq"]}';out=E/'matrix'/cell
 fixture=json.loads((E/'floor'/f'{cell}.command.json').read_text())[5]
 with (out/'measurement_runner.log').open('x') as log:
  code=subprocess.call([sys.executable,str(E/'measure.py'),'--source',str(out/'selected.cu'),'--fixture',fixture,'--top3'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 return dict(**{k:v for k,v in item.items() if k not in ('stage','exit')},stage='measured' if code==0 else 'measurement_failed',exit=code)
def main():
 state=[];pending=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as cpu,concurrent.futures.ThreadPoolExecutor(max_workers=1) as gpu:
  futures=[cpu.submit(solve,(m,s)) for s in (1,4,16,64) for m in ('llama','qwen3')]
  for future in concurrent.futures.as_completed(futures):
   item=future.result();state.append(item);print(json.dumps(item),flush=True)
   if item['stage']=='solved':pending.append(gpu.submit(measure,item))
   (E/'matrix_progress.json').write_text(json.dumps(state,indent=2)+'\n')
  for future in concurrent.futures.as_completed(pending):
   item=future.result();state.append(item);print(json.dumps(item),flush=True)
   (E/'matrix_progress.json').write_text(json.dumps(state,indent=2)+'\n')
 if any('failed' in item['stage'] for item in state):raise SystemExit(1)
if __name__=='__main__':main()
