from pathlib import Path
import subprocess,concurrent.futures,json,time
root=Path('/root/r10_work/implementation_completion');evidence=root/'docs/experiments/SERVING_R10/closure_r11';checks=evidence/'seed_checks';work=Path('/root/r10_work/current_plans')
# Do not replace seed plans with solved plans until all integration checks pass.
for model in ['llama','qwen3']:
 for batch in [1,16]:
  p=checks/f'{model}_B{batch}'/'hf_check.json'
  while not p.exists():time.sleep(5)
  if not json.loads(p.read_text())['pass']:raise SystemExit(f'failed seed C-1: {p}')
  if not json.loads((p.parent/'mode_check.json').read_text())['pass']:raise SystemExit(f'failed seed modes: {p}')
def run(item):
 model,phase=item;cmd=['python3',str(root/'docs/experiments/SERVING_R10/run_plan_matrix.py'),'--models',model,'--phases',phase,'--work',str(work),'--evidence',str(evidence/'current_plans'),'--compiler','/root/r10_work/completion_build/tools/tilemega-compile']
 # Build preflight serializes through its own tiny completed Ninja graph; the
 # compilation has already completed before these unchanged-source chains.
 start=time.monotonic()
 with (evidence/f'{model}_{phase}.stdout').open('w') as a,(evidence/f'{model}_{phase}.stderr').open('w') as b:r=subprocess.run(cmd,cwd=root,stdout=a,stderr=b)
 (evidence/f'{model}_{phase}.command.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start},indent=2)+'\n')
 return r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:r=list(pool.map(run,[(m,p) for m in ['llama','qwen3'] for p in ['prefill','decode']]))
raise SystemExit(any(r))
