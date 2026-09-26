from pathlib import Path
import subprocess,shlex,json,time,concurrent.futures,shutil
root=Path('/root/r10_work/implementation_completion'); out=root/'docs/experiments/SERVING_R10/closure_r11/seeds';work=Path('/root/r10_work/completion_seeds');out.mkdir(parents=True,exist_ok=True);work.mkdir(exist_ok=True)
def run(cell):
 old=Path('/root/r10_work')/(cell+'_seed.so'); cmd=shlex.split(Path(str(old)+'.build_command.txt').read_text());cmd=[x.replace('/root/TileMega',str(root)) for x in cmd];new=work/(cell+'.so');src=Path(str(new)+'.cu');shutil.copyfile(str(old)+'.cu',src);cmd=[str(src) if x==str(old)+'.cu' else x for x in cmd];cmd[cmd.index('-o')+1]=str(new)
 start=time.monotonic()
 with (out/(cell+'.build.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 (out/(cell+'.build.json')).write_text(json.dumps({'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start,'geometry_source':str(old)+'.cu','kernel_source':'aac391c3a'},indent=2)+'\n');print(cell,r.returncode,flush=True);return r.returncode
cells=[f'{m}_{p}'+('_B16' if b==16 else '') for m in ['llama','qwen3'] for b in [1,16] for p in ['prefill','decode']]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(run,cells))
raise SystemExit(any(results))
