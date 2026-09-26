from pathlib import Path
import os,subprocess,time,json,fcntl
root=Path('/root/r10_work/implementation_completion');out=root/'docs/experiments/SERVING_R10/closure_r11/seed_checks';out.mkdir(parents=True,exist_ok=True);work=Path('/root/r10_work/completion_seeds');env=os.environ.copy();env['PYTHONPATH']=str(root/'python');python='/root/venvs/tilemega-torch213-cu126/bin/python'
for model in ['llama','qwen3']:
 for b in [1,16]:
  suffix='_B16' if b==16 else ''; cell=out/f'{model}_B{b}';cell.mkdir(exist_ok=True)
  paths=[work/f'{model}_{phase}{suffix}.so' for phase in ['prefill','decode']]
  for path in paths:
   metadata=root/'docs/experiments/SERVING_R10/closure_r11/seeds'/f'{path.stem}.build.json'
   while not metadata.exists():time.sleep(2)
   if json.loads(metadata.read_text())['returncode']:raise RuntimeError(metadata)
  checkpoint='/root/models/'+('llama3_2_1b' if model=='llama' else 'qwen3_1_7b');ids=root/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
  mode=[python,'-m','tilemega.serving.check_modes','--model',checkpoint,'--prefill-so',str(paths[0]),'--decode-so',str(paths[1]),'--prompt-ids',str(ids),'--batch',str(b),'--out',str(cell)]
  hf=[python,'-m','tilemega.serving.hf_check','--model',checkpoint,'--prompt-ids',str(ids),'--generated',str(cell/'tokens_L2.json'),'--out',str(cell/'hf_check.json'),'--skip-free-greedy']
  for name,cmd in [('modes',mode),('hf',hf)]:
   with Path('/root/r10_work/serving_gpu.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    start=time.monotonic()
    with (cell/(name+'.stdout')).open('w') as stdout,(cell/(name+'.stderr')).open('w') as stderr:
     r=subprocess.run(cmd,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=900)
   (cell/(name+'.command.json')).write_text(json.dumps({'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start},indent=2)+'\n')
   print(model,b,name,r.returncode,flush=True)
   if r.returncode:raise SystemExit(r.returncode)
