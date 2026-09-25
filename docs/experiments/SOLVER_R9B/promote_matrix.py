#!/usr/bin/env python3
"""Start only the three recorded queued cells; retain the original GPU queue.

Each search remains single-threaded, full-domain, two starts, at most P=3.
run_search.py locks each cell; the existing coordinator later reuses its exact
command's completed output and remains the only owner of GPU measurements.
"""
import concurrent.futures,json,pathlib,subprocess,sys,time
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
CELLS=[('llama',64),('qwen3',16),('qwen3',64)]
def launch(cell):
 model,seq=cell;out=E/'matrix'/f'{model}_s{seq}';out.mkdir(parents=True,exist_ok=True)
 if (out/'solve.log').exists():return dict(cell=f'{model}_s{seq}',status='already started; coordinator owns it')
 with (out/'promotion_runner.log').open('x') as log:
  code=subprocess.call([sys.executable,str(E/'run_search.py'),model,str(seq)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 return dict(cell=f'{model}_s{seq}',exit=code)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(launch,cell) for cell in CELLS];done=[]
 for future in concurrent.futures.as_completed(futures):
  done.append(future.result());print(json.dumps(done[-1]),flush=True)
  (E/'promotion_progress.json').write_text(json.dumps(done,indent=2)+'\n')
if any(r.get('exit',0) for r in done):raise SystemExit(1)
